"""Transformer state-dict adapter.

Applies a state_dict to a transformer-like caramba model using a schema that
describes external key naming conventions and policies that describe how to
initialize parameters that do not exist in the teacher checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast
import torch
from torch import Tensor, nn

from adapter.schema import SchemaLoader, StateDictSchema
from config.layer import AttentionMode
from initializers.dba.base import DBAInitializer
from initializers.dba.fresh import DBAFresh, init_fresh_linear
from initializers.dba.dba_random import DBARandom
from initializers.dba.svd import DBASVD
from layer.attention import AttentionLayer
from layer.linear import LinearLayer
from layer.rms_norm import RMSNormLayer
from layer.swiglu import SwiGLULayer
from loader.state_reader import StateReader
from model.embedder import Embedder


class AdapterStateDictTransformer:
    """Schema-driven state-dict adapter for transformer-like models."""

    def __init__(
        self,
        *,
        schema: StateDictSchema,
        dba_initializer: DBAInitializer,
        gate_init_bias: float | None = None,
        out_proj_init_std: float | None = None,
    ) -> None:
        self.schema = schema
        self.dba_initializer = dba_initializer
        self.gate_init_bias = gate_init_bias
        self.out_proj_init_std = out_proj_init_std

    @classmethod
    def llama(
        cls,
        *,
        dba_init: str = "svd",
        gate_init_bias: float | None = None,
        out_proj_init_std: float | None = None,
    ) -> "AdapterStateDictTransformer":
        """Build the transformer adapter with the built-in Llama schema."""

        schema_path = Path(__file__).resolve().parent.parent / "schema" / "llama.yml"
        schema = SchemaLoader().load(path=str(schema_path))
        init = cls.buildDbaInitializer(dba_init=str(dba_init))
        return cls(
            schema=schema,
            dba_initializer=init,
            gate_init_bias=gate_init_bias,
            out_proj_init_std=out_proj_init_std,
        )

    @staticmethod
    def buildDbaInitializer(*, dba_init: str) -> DBAInitializer:
        """Build a DBA initializer policy.

        Available policies:
            - "svd": Initialize DBA Q/K from SVD decomposition of teacher Q/K (preserves patterns)
            - "random": Random init for Q/K, but still copy V/O from teacher
            - "fresh": Complete random init for all projections (routing hypothesis)
        """

        s = str(dba_init).lower().strip()
        if s == "svd":
            return DBASVD()
        if s == "random":
            return DBARandom()
        if s == "fresh":
            return DBAFresh()
        raise ValueError(f"Unsupported dba_init={dba_init!r} (expected 'svd', 'random', or 'fresh')")

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
        self.loadHead(state=state, head=head, embedder=embedder)

    def key(self, *parts: str) -> str:
        """Build a full key from schema parts."""
        prefix = str(self.schema.prefix)
        cleaned = [str(p) for p in parts if str(p)]
        # Some call sites build intermediate prefixes (e.g. `attn_prefix`) using `key()`
        # and then pass them back into `key()` again. If `cleaned[0]` already includes
        # the schema prefix (e.g. "model.layers.0.self_attn"), avoid duplicating it.
        if cleaned and (cleaned[0] == prefix or cleaned[0].startswith(prefix + ".")):
            return ".".join(cleaned)
        return ".".join([prefix] + cleaned)

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

    def loadHead(self, *, state: StateReader, head: LinearLayer | None, embedder: Embedder) -> None:
        """Load explicit LM head weight if present.

        Notes:
        - Some Llama checkpoints omit `lm_head.weight` because weights are tied to the
          token embedding matrix. In that case we fall back to copying the token
          embedding weights into the head when the model declares an explicit head.
        """

        if head is None:
            return
        if self.schema.head is None:
            raise ValueError("Schema has no head but model has a head")
        primary_key = self.key(self.schema.head.weight)
        try:
            w = state.get(primary_key)
        except (KeyError, ValueError):
            # Fallbacks for common Llama checkpoint conventions.
            fallback_keys = [
                "lm_head.weight",
                "model.lm_head.weight",
                "output.weight",
                "model.output.weight",
            ]
            w2 = None
            for fk in fallback_keys:
                try:
                    w2 = state.get(fk)
                    break
                except (KeyError, ValueError):
                    continue
            if w2 is not None:
                w = w2
            else:
                # Final fallback: tie to embeddings when teacher omits the head.
                tok = getattr(embedder, "token_embedding", None)
                if tok is None or getattr(tok, "weight", None) is None:
                    raise ValueError(f"Missing state_dict key: {primary_key}")
                head.linear.weight.data.copy_(tok.weight.data)
                return
        head.linear.weight.data.copy_(w)

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
        """Load decoupled attention using DBA initializer for Q/K.

        For SVD/random init: copies V/O from teacher, initializes Q/K via policy.
        For fresh init: initializes ALL projections randomly (routing hypothesis).

        Note: For GQA models, Q and K have different dimensions:
        - Q uses n_heads (full head count)
        - K uses n_kv_heads (may be smaller)
        """

        if attn.q_sem is None or attn.k_sem is None:
            raise ValueError("DBA attention missing semantic projections (q_sem/k_sem)")
        if attn.q_geo is None or attn.k_geo is None:
            raise ValueError("DBA attention missing geometric projections (q_geo/k_geo)")
        if attn.v_proj is None or attn.out_proj is None:
            raise ValueError("DBA attention missing v_proj/out_proj")

        # Q dimensions (n_heads based)
        sem_q_dim = int(attn.q_sem.out_features)
        geo_q_dim = int(attn.q_geo.out_features)

        # K dimensions (n_kv_heads based - may be smaller for GQA)
        sem_kv_dim = int(attn.k_sem.out_features)
        geo_kv_dim = int(attn.k_geo.out_features)

        # Fresh mode: initialize ALL projections randomly (full replacement)
        # This is the routing hypothesis approach - don't copy anything from teacher
        if isinstance(self.dba_initializer, DBAFresh):
            # V projection: normal Xavier scale
            init_fresh_linear(attn.v_proj, seed=f"{attn_prefix}.v", suffix="v", scale=1.0)
            # O projection: small scale (GPT-2 style) to avoid disrupting residual stream
            # The pretrained norms expect attention outputs to be in a certain range;
            # starting with small outputs lets the model gradually learn routing.
            init_fresh_linear(attn.out_proj, seed=f"{attn_prefix}.o", suffix="o", scale=0.02)
        else:
            # SVD/random mode: copy V/O from teacher
            self.copyVO(attn=attn, v=v, o=o)
            if self.out_proj_init_std is not None:
                nn.init.normal_(attn.out_proj.weight, mean=0.0, std=self.out_proj_init_std)

        # Initialize Q semantic and geometric projections (uses full n_heads)
        self.dba_initializer.initialize(
            sem_weight=attn.q_sem.weight,
            geo_weight=attn.q_geo.weight,
            teacher_weight=q,
            sem_dim=sem_q_dim,
            geo_dim=geo_q_dim,
            seed=f"{attn_prefix}.q",
        )

        # Initialize K semantic and geometric projections (uses n_kv_heads)
        self.dba_initializer.initialize(
            sem_weight=attn.k_sem.weight,
            geo_weight=attn.k_geo.weight,
            teacher_weight=k,
            sem_dim=sem_kv_dim,
            geo_dim=geo_kv_dim,
            seed=f"{attn_prefix}.k",
        )

        # Reset gate to neutral (0.5 semantic/geometric mix)
        if attn.decoupled_gate_logit is not None:
            if self.gate_init_bias is not None:
                attn.decoupled_gate_logit.data.fill_(self.gate_init_bias)
            else:
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

        # NOTE: We intentionally avoid `isinstance(m, Embedder)` here.
        #
        # The CLI sometimes loads the codebase under both module roots:
        #   - `model.*` / `layer.*`
        #   - `caramba.model.*` / `caramba.layer.*`
        #
        # That can produce duplicate class objects with the same name, causing
        # `isinstance` checks to fail even though the module is correct.
        for _name, m in model.named_modules():
            if type(m).__name__ == "Embedder" and hasattr(m, "token_embedding"):
                return cast(Embedder, m)
        raise ValueError("No Embedder found in model (module path mismatch?)")

    def findHead(self, *, model: nn.Module) -> LinearLayer | None:
        """Find an explicit head if present."""

        heads: list[LinearLayer] = []
        for _name, m in model.named_modules():
            if type(m).__name__ == "LinearLayer" and hasattr(m, "linear"):
                heads.append(cast(LinearLayer, m))
        return heads[-1] if heads else None

    def findAttn(self, *, model: nn.Module) -> list[AttentionLayer]:
        """Find attention layers."""

        attn: list[AttentionLayer] = []
        for _name, m in model.named_modules():
            n = type(m).__name__
            if n in {"AttentionLayer", "StandardAttentionLayer", "DecoupledAttentionLayer"}:
                attn.append(cast(AttentionLayer, m))
                continue
            # Fallback for factory-returned subclasses / alias modules.
            if hasattr(m, "mode") and hasattr(m, "out_proj") and (
                hasattr(m, "qkv_proj") or (hasattr(m, "q_sem") and hasattr(m, "q_geo"))
            ):
                attn.append(cast(AttentionLayer, m))
        return attn

    def findMlp(self, *, model: nn.Module) -> list[SwiGLULayer]:
        """Find MLP layers."""

        mlp: list[SwiGLULayer] = []
        for _name, m in model.named_modules():
            if type(m).__name__ == "SwiGLULayer" and hasattr(m, "w_gate_up") and hasattr(m, "w_down"):
                mlp.append(cast(SwiGLULayer, m))
        return mlp

    def findNorms(self, *, model: nn.Module) -> list[RMSNormLayer]:
        """Find RMSNorm layers."""

        norms: list[RMSNormLayer] = []
        for _name, m in model.named_modules():
            if type(m).__name__ == "RMSNormLayer" and hasattr(m, "weight") and hasattr(m, "eps"):
                norms.append(cast(RMSNormLayer, m))
        return norms

