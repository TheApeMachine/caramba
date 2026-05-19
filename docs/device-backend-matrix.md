# Device backend matrix (T1.4)

Per-backend registration for **Metal**, **CUDA**, and **XLA** tensor backends under `pkg/backend/device/`. **registered** means at least one `kernels.Default` entry with that `tensor.Location`; it does not assert full `device.Backend` coverage.

Machine-checkable source: `pkg/backend/device/backendaudit/`, validated by `backendaudit_test.go`. Metal registrations load via `load_metal.go` blank import.

CPU dispatch (T1.3): [`cpu-dispatch-matrix.md`](./cpu-dispatch-matrix.md). Combined coverage (T1.5): [`backend-coverage.md`](./backend-coverage.md).

## Backend summary

| Backend | Supported dtypes | Kernel registrations | Unique kernel names | `.metal` / dispatch sources | `*_darwin.go` | `*_stub.go` | Tensor API methods |
|---------|------------------|------------------------:|----------------------:|---------------------------:|--------------:|-------------:|-------------------:|
| metal | F32, BF16, F16, I32, I8, I4, BOOL | 462 | 158 | 24 | 39 | 6 | 33 |
| cuda | F32, BF16, F16, I8, I4, BOOL, F8E4M3, F8E5M2 | 0 | 0 | 0 | 0 | 1 | 9 |
| xla | F64, F32, F16, BF16, F8E4M3, F8E5M2, I64, I32, I16, I8, U64, U32, U16, U8, BOOL | 0 | 0 | 0 | 0 | 1 | 9 |

## Required IR operations — Metal kernel coverage

Maps each `ir.RequiredOperationIDs()` entry to expected kernel name(s) and whether any Metal registration exists.

| Operation ID | Cross-link | Metal kernels |
|--------------|------------|:-------------:|
| `Input` | graph_only | — |
| `Add` | direct | yes |
| `Mul` | direct | yes |
| `Matmul` | direct | yes |
| `ReLU` | direct | yes |
| `LeakyReLU` | direct | yes |
| `GELU` | direct | — |
| `Tanh` | direct | yes |
| `Sigmoid` | direct | yes |
| `SwiGLU` | direct | — |
| `Swish` | direct | yes |
| `SELU` | direct | yes |
| `Fused` | graph_only | — |
| `activation.relu` | direct | yes |
| `activation.leaky_relu` | direct | yes |
| `activation.gelu` | direct | — |
| `activation.tanh` | direct | yes |
| `activation.sigmoid` | direct | yes |
| `activation.swiglu` | direct | — |
| `activation.swish` | direct | yes |
| `activation.selu` | direct | yes |
| `attention.sdpa` | direct | yes |
| `attention.mqa` | composite | yes |
| `attention.gqa` | composite | yes |
| `attention.sliding_window` | direct | yes |
| `masking.apply` | direct | yes |
| `masking.causal` | direct | yes |
| `math.add` | direct | yes |
| `math.mul` | direct | yes |
| `math.matmul` | direct | yes |
| `math.exp` | direct | yes |
| `math.sin` | kernel_registry | yes |
| `math.cos` | kernel_registry | yes |
| `math.log` | direct | yes |
| `math.logsumexp` | kernel_registry | yes |
| `math.softmax` | direct | yes |
| `math.outer` | kernel_registry | yes |
| `math.sign` | kernel_registry | yes |
| `math.inv_sqrt_dim_scale` | kernel_registry | yes |
| `math.dropout` | direct | yes |
| `math.rmsnorm` | direct | yes |
| `math.layernorm` | direct | yes |
| `math.groupnorm` | direct | yes |
| `shape.reshape` | kernel_registry | yes |
| `shape.transpose` | kernel_registry | yes |
| `shape.concat` | kernel_registry | yes |
| `shape.split` | kernel_registry | yes |
| `shape.upsample_nearest2d` | kernel_registry | yes |
| `shape.view_as_heads` | kernel_registry | yes |
| `shape.merge_heads` | kernel_registry | yes |
| `shape.last_token` | kernel_registry | yes |
| `shape.slice` | kernel_registry | — |
| `positional.rope` | direct | yes |
| `positional.alibi` | direct | yes |
| `embedding.token` | direct | yes |
| `convolution.conv1d` | direct | yes |
| `convolution.conv2d` | direct | yes |
| `convolution.conv3d` | direct | yes |
| `convolution.conv_transpose2d` | direct | — |
| `pooling.max_pool2d` | direct | — |
| `pooling.avg_pool2d` | direct | — |
| `pooling.adaptive_avg_pool2d` | direct | — |
| `pooling.adaptive_max_pool2d` | direct | — |
| `projection.linear` | direct | yes |
| `projection.fused_qkv` | composite | yes |
| `hawkes.intensity` | direct | — |
| `hawkes.kernel_matrix` | direct | — |
| `hawkes.log_likelihood` | direct | — |
| `hawkes.simulate` | kernel_registry | — |
| `vsa.bind` | direct | — |
| `vsa.bundle` | direct | — |
| `vsa.similarity` | direct | — |
| `vsa.permute` | direct | — |
| `vsa.inverse_permute` | direct | — |
| `active_inference.belief_update` | direct | — |
| `active_inference.expected_free_energy` | direct | — |
| `active_inference.free_energy` | direct | — |
| `active_inference.precision_weight` | direct | — |
| `predictive_coding.prediction` | direct | — |
| `predictive_coding.prediction_error` | direct | — |
| `predictive_coding.update_representation` | direct | — |
| `predictive_coding.update_weights` | direct | — |
| `markov_blanket.flow_active` | direct | — |
| `markov_blanket.flow_internal` | direct | — |
| `markov_blanket.mutual_information` | direct | — |
| `markov_blanket.partition` | direct | — |
| `causal.backdoor_adjustment` | direct | — |
| `causal.cate` | direct | yes |
| `causal.counterfactual` | direct | yes |
| `causal.dag_markov_factorization` | direct | — |
| `causal.do_calculus` | direct | — |
| `causal.frontdoor_adjustment` | direct | — |
| `causal.iv_estimate` | direct | — |
| `train.loss.mse` | direct | yes |
| `train.loss.cross_entropy` | direct | yes |
| `train.loss.mse_grad` | kernel_registry | — |
| `train.loss.cross_entropy_grad` | kernel_registry | — |
| `train.grad.mse` | kernel_registry | — |
| `train.grad.cross_entropy` | kernel_registry | yes |
| `train.optimizer.adam` | kernel_registry | yes |
| `train.optimizer.adamw` | kernel_registry | yes |
| `train.optimizer.adamax` | kernel_registry | yes |
| `train.optimizer.sgd` | kernel_registry | yes |
| `train.optimizer.lion` | kernel_registry | yes |
| `train.optimizer.rmsprop` | kernel_registry | yes |
| `train.optimizer.hebbian` | kernel_registry | yes |
| `train.optimizer.lars` | kernel_registry | yes |
| `train.optimizer.lamb` | kernel_registry | — |
| `train.optimizer.adagrad` | kernel_registry | yes |
| `train.optimizer.adadelta` | kernel_registry | — |
| `train.optimizer.lbfgs` | kernel_registry | yes |
| `bench.accuracy` | graph_only | — |
| `bench.perplexity` | graph_only | — |
| `bench.f1` | graph_only | — |
| `bench.metric.accuracy` | graph_only | — |
| `bench.metric.perplexity` | graph_only | — |
| `bench.metric.f1` | graph_only | — |
| `model.graft` | graph_only | — |
| `model.freeze` | graph_only | — |

Metal covers **68 / 119** required operation IDs via `kernels.Default`.

## Kernel name index (Metal / CUDA / XLA)

| Kernel name | Metal | CUDA | XLA | Dtype variants (Metal) |
|-------------|:-----:|:----:|:---:|-------------------------:|
| abs | yes | — | — | 3 |
| adagrad_step | yes | — | — | 3 |
| adam_step | yes | — | — | 3 |
| adamax_step | yes | — | — | 3 |
| adamw_step | yes | — | — | 3 |
| adaptive_avg_pool2d | yes | — | — | 3 |
| adaptive_max_pool2d | yes | — | — | 3 |
| add | yes | — | — | 3 |
| alibi_bias | yes | — | — | 3 |
| apply_mask | yes | — | — | 3 |
| argmax | yes | — | — | 3 |
| argmin | yes | — | — | 3 |
| atan2 | yes | — | — | 3 |
| attention | yes | — | — | 3 |
| avg_pool2d | yes | — | — | 3 |
| backdoor_adjustment | yes | — | — | 3 |
| batchnorm_eval | yes | — | — | 3 |
| belief_update | yes | — | — | 3 |
| binary_cross_entropy | yes | — | — | 3 |
| bohmian_velocity | yes | — | — | 3 |
| cate | yes | — | — | 3 |
| causal_mask | yes | — | — | 3 |
| checkpoint_decode_float32 | yes | — | — | 1 |
| checkpoint_encode_float32 | yes | — | — | 1 |
| concat | yes | — | — | 3 |
| conv1d | yes | — | — | 3 |
| conv2d | yes | — | — | 3 |
| conv3d | yes | — | — | 3 |
| conv_transpose2d | yes | — | — | 3 |
| cos | yes | — | — | 3 |
| counterfactual | yes | — | — | 3 |
| cross_entropy | yes | — | — | 3 |
| dag_markov_factorization | yes | — | — | 3 |
| div | yes | — | — | 3 |
| divergence1d | yes | — | — | 3 |
| do_intervene | yes | — | — | 3 |
| dropout | yes | — | — | 3 |
| elu | yes | — | — | 3 |
| embedding_bag | yes | — | — | 3 |
| embedding_lookup | yes | — | — | 3 |
| eq | yes | — | — | 3 |
| exp | yes | — | — | 3 |
| expected_free_energy | yes | — | — | 3 |
| fft1d | yes | — | — | 3 |
| flash_attention | yes | — | — | 3 |
| free_energy | yes | — | — | 3 |
| frontdoor_adjustment | yes | — | — | 3 |
| fused_qkv | yes | — | — | 3 |
| gather | yes | — | — | 3 |
| ge | yes | — | — | 3 |
| grad1d | yes | — | — | 3 |
| greedy_sample | yes | — | — | 3 |
| grouped_query_attention | yes | — | — | 3 |
| groupnorm | yes | — | — | 3 |
| gt | yes | — | — | 3 |
| hardsigmoid | yes | — | — | 3 |
| hardswish | yes | — | — | 3 |
| hawkes_intensity | yes | — | — | 3 |
| hawkes_kernel_matrix | yes | — | — | 3 |
| hawkes_log_likelihood | yes | — | — | 3 |
| hebbian_step | yes | — | — | 3 |
| huber_loss | yes | — | — | 3 |
| ifft1d | yes | — | — | 3 |
| instancenorm | yes | — | — | 3 |
| int4_dequant | yes | — | — | 1 |
| int8_dequant | yes | — | — | 1 |
| int8_quant | yes | — | — | 1 |
| inv_sqrt_dim_scale | yes | — | — | 3 |
| iv_estimate | yes | — | — | 3 |
| kl_divergence | yes | — | — | 3 |
| l1_norm | yes | — | — | 3 |
| l2_norm | yes | — | — | 3 |
| laplacian | yes | — | — | 3 |
| laplacian4 | yes | — | — | 3 |
| lars_step | yes | — | — | 3 |
| last_token | yes | — | — | 3 |
| layernorm | yes | — | — | 3 |
| lbfgs_step | yes | — | — | 3 |
| le | yes | — | — | 3 |
| leaky_relu | yes | — | — | 3 |
| linear | yes | — | — | 3 |
| lion_step | yes | — | — | 3 |
| log | yes | — | — | 3 |
| logsumexp | yes | — | — | 3 |
| lora_apply | yes | — | — | 3 |
| lora_merge | yes | — | — | 3 |
| lt | yes | — | — | 3 |
| madelung_continuity | yes | — | — | 3 |
| mae_loss | yes | — | — | 3 |
| markov_blanket_partition | yes | — | — | 3 |
| markov_flow_active | yes | — | — | 3 |
| markov_flow_internal | yes | — | — | 3 |
| markov_mutual_information | yes | — | — | 3 |
| masked_fill | yes | — | — | 3 |
| matmul | yes | — | — | 3 |
| matmul_add | yes | — | — | 3 |
| max | yes | — | — | 3 |
| max_pool2d | yes | — | — | 3 |
| mean | yes | — | — | 3 |
| merge_heads | yes | — | — | 3 |
| min | yes | — | — | 3 |
| mod | yes | — | — | 3 |
| mse_loss | yes | — | — | 3 |
| mul | yes | — | — | 3 |
| multi_head_attention | yes | — | — | 3 |
| ne | yes | — | — | 3 |
| neg | yes | — | — | 3 |
| outer | yes | — | — | 3 |
| pc_prediction | yes | — | — | 3 |
| pc_prediction_error | yes | — | — | 3 |
| pc_update_representation | yes | — | — | 3 |
| pc_update_weights | yes | — | — | 3 |
| pow | yes | — | — | 3 |
| precision_weight | yes | — | — | 3 |
| prod | yes | — | — | 3 |
| quantum_potential | yes | — | — | 3 |
| recip | yes | — | — | 3 |
| reduce_max | yes | — | — | 3 |
| reduce_min | yes | — | — | 3 |
| relu | yes | — | — | 3 |
| reshape | yes | — | — | 3 |
| rmsnorm | yes | — | — | 3 |
| rmsprop_step | yes | — | — | 3 |
| rope | yes | — | — | 3 |
| rsqrt | yes | — | — | 3 |
| scatter | yes | — | — | 3 |
| selu | yes | — | — | 3 |
| sgd_step | yes | — | — | 3 |
| sigmoid | yes | — | — | 3 |
| sign | yes | — | — | 3 |
| silu | yes | — | — | 3 |
| sin | yes | — | — | 3 |
| sliding_window_attention | yes | — | — | 3 |
| softmax | yes | — | — | 3 |
| softsign | yes | — | — | 3 |
| split2 | yes | — | — | 3 |
| split_heads | yes | — | — | 3 |
| sqrt | yes | — | — | 3 |
| square | yes | — | — | 3 |
| stddev | yes | — | — | 3 |
| sub | yes | — | — | 3 |
| sum | yes | — | — | 3 |
| swish | yes | — | — | 3 |
| tanh | yes | — | — | 3 |
| tokenizer_pack_int32 | yes | — | — | 1 |
| topk_sample | yes | — | — | 3 |
| topp_sample | yes | — | — | 3 |
| transpose | yes | — | — | 3 |
| transpose2d | yes | — | — | 3 |
| upsample_nearest2d | yes | — | — | 3 |
| variance | yes | — | — | 3 |
| view_as_heads | yes | — | — | 3 |
| vsa_bind | yes | — | — | 3 |
| vsa_bundle | yes | — | — | 3 |
| vsa_inverse_permute | yes | — | — | 3 |
| vsa_permute | yes | — | — | 3 |
| weight_freeze_mask | yes | — | — | 3 |
| where | yes | — | — | 3 |

