"""Transformer state-dict adapter.

Applies a state_dict to a transformer-like caramba model using a schema that
describes external key naming conventions and policies that describe how to
initialize parameters that do not exist in the teacher checkpoint.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor, nn

from caramba.adapter.schema import SchemaLoader, StateDictSchema
from caramba.config.layer import AttentionMode
from caramba.initializers.dba.base import DBAInitializer
from caramba.initializers.dba.random import DBARandom
from caramba.initializers.dba.svd import DBASVD
from caramba.layer.attention import AttentionLayer
from caramba.layer.linear import LinearLayer
from caramba.layer.rms_norm import RMSNormLayer
from caramba.layer.swiglu import SwiGLULayer
from caramba.loader.state_reader import StateReader
from caramba.model.embedder import Embedder


class AdapterStateDictTransformer:
    """Schema-driven state-dict adapter for transformer-like models."""

    def __init__(self, *, schema: StateDictSchema, dba_initializer: DBAInitializer) -> None:
        self.schema = schema
        self.dba_initializer = dba_initializer

    @classmethod
    def llama(cls, *, dba_init: str = "svd") -> "AdapterStateDictTransformer":
        """Build the transformer adapter with the built-in Llama schema."""

        schema_path = Path(__file__).resolve().parent.parent / "schema" / "llama.yml"
        schema = SchemaLoader().load(path=str(schema_path))
        init = cls.buildDbaInitializer(dba_init=str(dba_init))
        return cls(schema=schema, dba_initializer=init)

    @staticmethod
    def buildDbaInitializer(*, dba_init: str) -> DBAInitializer:
        """Build a DBA initializer policy."""

        s = str(dba_init).lower().strip()
        if s == "svd":
            return DBASVD()
        if s == "random":
            return DBARandom()
        raise ValueError(f"Unsupported dba_init={dba_init!r} (expected 'svd' or 'random')")

    def apply(self, *, model: nn.Module, state_dict: dict[str, Tensor]) -> None:
        """Apply state_dict weights to the model."""

        state = StateReader(state_dict)
        embedder = self.findEmbedder(model=model)
        head = self.findHead(model=model)
        attn = self.findAttn(model=model)
        mlp = self.findMlp(model=model)
        norms = self.findNorms(model=model)
        self.validateCounts(attn=attn, mlp=mlp, norms=norms)

        self.loadEmbedder(state=state, embedder=embedder)
        self.loadBlocks(state=state, attn=attn, mlp=mlp, norms=norms)
        self.loadFinalNorm(state=state, norms=norms)
        self.loadHead(state=state, head=head)

    def key(self, *parts: str) -> str:
        """Build a full key from schema parts."""

        return ".".join([str(self.schema.prefix)] + [p for p in parts if str(p)])

    def layerPrefix(self, *, i: int) -> str:
        """Return the per-layer prefix."""

        return str(self.schema.block.path).format(i=int(i))

    def loadEmbedder(self, *, state: StateReader, embedder: Embedder) -> None:
        """Load token embedding weights."""

        if embedder.token_embedding is None:
            raise ValueError("Embedder has no token_embedding")
        k = self.key(self.schema.embedder.tokens_weight)
        embedder.token_embedding.weight.data.copy_(state.get(k))

    def loadBlocks(
        self,
        *,
        state: StateReader,
        attn: list[AttentionLayer],
        mlp: list[SwiGLULayer],
        norms: list[RMSNormLayer],
    ) -> None:
        """Load all transformer blocks."""

        for i, (att, m) in enumerate(zip(attn, mlp)):
            self.loadBlock(state=state, i=int(i), attn=att, mlp=m, norms=norms)

    def loadBlock(
        self,
        *,
        state: StateReader,
        i: int,
        attn: AttentionLayer,
        mlp: SwiGLULayer,
        norms: list[RMSNormLayer],
    ) -> None:
        """Load a single transformer block."""

        layer_prefix = self.layerPrefix(i=int(i))
        self.loadBlockNorms(state=state, i=int(i), layer_prefix=layer_prefix, norms=norms)
        self.loadAttention(state=state, layer_prefix=layer_prefix, attn=attn)
        self.loadMlp(state=state, layer_prefix=layer_prefix, mlp=mlp)

    def loadBlockNorms(
        self, *, state: StateReader, i: int, layer_prefix: str, norms: list[RMSNormLayer]
    ) -> None:
        """Load per-block RMSNorm weights."""

        n0 = norms[2 * int(i)]
        n1 = norms[2 * int(i) + 1]
        self.copyNorm(state=state, layer=n0, key=self.key(layer_prefix, self.schema.block.input_norm_weight))
        self.copyNorm(state=state, layer=n1, key=self.key(layer_prefix, self.schema.block.post_attn_norm_weight))

    def loadFinalNorm(self, *, state: StateReader, norms: list[RMSNormLayer]) -> None:
        """Load final RMSNorm."""

        self.copyNorm(state=state, layer=norms[-1], key=self.key(self.schema.final_norm_weight))

    def copyNorm(self, *, state: StateReader, layer: RMSNormLayer, key: str) -> None:
        """Copy norm weight."""

        if layer.weight is None:
            raise ValueError("RMSNormLayer is missing weight parameter")
        layer.weight.data.copy_(state.get(str(key)))

    def loadHead(self, *, state: StateReader, head: LinearLayer | None) -> None:
        """Load explicit LM head weight if present."""

        if head is None:
            return
        if self.schema.head is None:
            raise ValueError("Schema has no head but model has a head")
        head.linear.weight.data.copy_(state.get(self.key(self.schema.head.weight)))

    def loadAttention(self, *, state: StateReader, layer_prefix: str, attn: AttentionLayer) -> None:
        """Load attention weights."""

        attn_prefix = self.key(layer_prefix, self.schema.attention.path)
        q = state.get(self.key(attn_prefix, self.schema.attention.q_weight))
        k = state.get(self.key(attn_prefix, self.schema.attention.k_weight))
        v = state.get(self.key(attn_prefix, self.schema.attention.v_weight))
        o = state.get(self.key(attn_prefix, self.schema.attention.o_weight))

        if attn.mode == AttentionMode.DECOUPLED:
            self.loadAttentionDecoupled(attn=attn, q=q, k=k, v=v, o=o, attn_prefix=attn_prefix)
            return

        self.loadAttentionStandard(attn=attn, q=q, k=k, v=v, o=o)

    def loadAttentionStandard(
        self, *, attn: AttentionLayer, q: Tensor, k: Tensor, v: Tensor, o: Tensor
    ) -> None:
        """Load standard attention by packing QKV."""

        qkv_proj = getattr(attn, "qkv_proj", None)
        out_proj = getattr(attn, "out_proj", None)
        if not isinstance(qkv_proj, nn.Linear) or not isinstance(out_proj, nn.Linear):
            raise TypeError("Standard attention missing qkv_proj/out_proj Linear modules")
        qkv_proj.weight.data.copy_(torch.cat([q, k, v], dim=0))
        out_proj.weight.data.copy_(o)

    def loadAttentionDecoupled(
        self,
        *,
        attn: AttentionLayer,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        o: Tensor,
        attn_prefix: str,
    ) -> None:
        """Load decoupled attention using DBA initializer for Q/K and copying V/O."""

        if attn.q_sem is None or attn.k_sem is None:
            raise ValueError("DBA attention missing semantic projections (q_sem/k_sem)")
        if attn.q_geo is None or attn.k_geo is None:
            raise ValueError("DBA attention missing geometric projections (q_geo/k_geo)")
        if attn.v_proj is None or attn.out_proj is None:
            raise ValueError("DBA attention missing v_proj/out_proj")

        sem_dim = int(attn.q_sem.out_features)
        geo_dim = int(attn.q_geo.out_features)

        self.copyVO(attn=attn, v=v, o=o)
        self.dba_initializer.initialize(
            sem_weight=attn.q_sem.weight,
            geo_weight=attn.q_geo.weight,
            teacher_weight=q,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            seed=f"{attn_prefix}.q",
        )
        self.dba_initializer.initialize(
            sem_weight=attn.k_sem.weight,
            geo_weight=attn.k_geo.weight,
            teacher_weight=k,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            seed=f"{attn_prefix}.k",
        )
        if attn.decoupled_gate_logit is not None:
            attn.decoupled_gate_logit.data.zero_()

    def copyVO(self, *, attn: AttentionLayer, v: Tensor, o: Tensor) -> None:
        """Copy V and O weights with truncation/padding."""

        v_out = int(attn.v_proj.out_features)  # type: ignore[union-attr]
        if int(v.size(0)) == v_out:
            attn.v_proj.weight.data.copy_(v)  # type: ignore[union-attr]
        else:
            n = min(int(v.size(0)), v_out)
            attn.v_proj.weight.data[:n, :].copy_(v[:n, :])  # type: ignore[union-attr]

        o_in = int(attn.out_proj.in_features)  # type: ignore[union-attr]
        if int(o.size(1)) == o_in:
            attn.out_proj.weight.data.copy_(o)  # type: ignore[union-attr]
        else:
            n = min(int(o.size(1)), o_in)
            attn.out_proj.weight.data[:, :n].copy_(o[:, :n])  # type: ignore[union-attr]

    def loadMlp(self, *, state: StateReader, layer_prefix: str, mlp: SwiGLULayer) -> None:
        """Load MLP weights, including fused gate+up."""

        mlp_prefix = self.key(layer_prefix, self.schema.mlp.path)
        gate_w = state.get(self.key(mlp_prefix, self.schema.mlp.gate_weight))
        up_w = state.get(self.key(mlp_prefix, self.schema.mlp.up_weight))
        mlp.w_gate_up.weight.data.copy_(torch.cat([gate_w, up_w], dim=0))

        if mlp.w_gate_up.bias is not None:
            if self.schema.mlp.gate_bias is None or self.schema.mlp.up_bias is None:
                raise ValueError("Schema missing gate/up bias keys but model expects bias")
            gate_b = state.get(self.key(mlp_prefix, self.schema.mlp.gate_bias))
            up_b = state.get(self.key(mlp_prefix, self.schema.mlp.up_bias))
            mlp.w_gate_up.bias.data.copy_(torch.cat([gate_b, up_b], dim=0))

        mlp.w_down.weight.data.copy_(state.get(self.key(mlp_prefix, self.schema.mlp.down_weight)))
        if mlp.w_down.bias is not None:
            if self.schema.mlp.down_bias is None:
                raise ValueError("Schema missing down bias key but model expects bias")
            mlp.w_down.bias.data.copy_(state.get(self.key(mlp_prefix, self.schema.mlp.down_bias)))

    def validateCounts(
        self,
        *,
        attn: list[AttentionLayer],
        mlp: list[SwiGLULayer],
        norms: list[RMSNormLayer],
    ) -> None:
        """Validate module counts match expected transformer structure."""

        if len(attn) != len(mlp):
            raise ValueError(f"Attention/MLP count mismatch: {len(attn)} vs {len(mlp)}")
        expected_norms = 2 * len(attn) + 1
        if len(norms) != expected_norms:
            raise ValueError(f"Expected {expected_norms} norms, got {len(norms)}")

    def findEmbedder(self, *, model: nn.Module) -> Embedder:
        """Find Embedder module."""

        for _name, m in model.named_modules():
            if isinstance(m, Embedder):
                return m
        raise ValueError("No Embedder found in model")

    def findHead(self, *, model: nn.Module) -> LinearLayer | None:
        """Find an explicit head if present."""

        heads = [m for _name, m in model.named_modules() if isinstance(m, LinearLayer)]
        return heads[-1] if heads else None

    def findAttn(self, *, model: nn.Module) -> list[AttentionLayer]:
        """Find attention layers."""

        return [m for _name, m in model.named_modules() if isinstance(m, AttentionLayer)]

    def findMlp(self, *, model: nn.Module) -> list[SwiGLULayer]:
        """Find MLP layers."""

        return [m for _name, m in model.named_modules() if isinstance(m, SwiGLULayer)]

    def findNorms(self, *, model: nn.Module) -> list[RMSNormLayer]:
        """Find RMSNorm layers."""

        return [m for _name, m in model.named_modules() if isinstance(m, RMSNormLayer)]

