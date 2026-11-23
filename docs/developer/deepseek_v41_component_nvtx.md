# DeepSeek-V4.1 component profiling

This adapts NVIDIA/Megatron-LM commit
[27e1018dbc](https://github.com/NVIDIA/Megatron-LM/commit/27e1018dbc2606fb9b20e67995d4148b59e0e0e4)
to the historical D/E HybridStack and CSA2 interfaces used by the SFT fork.
Enable `MCORE_COMPONENT_NVTX=1` before importing Megatron. Unset or `0` leaves
component/layer decorators as identity functions. No optimizer, model parameter,
checkpoint layout or distributed collective is added by the instrumentation.

## Labels and ownership

- `microbatch` ranges contain step, original microbatch, rank and explicit
  PP/CP/DP group coordinates. The pipeline schedule already publishes
  `current_microbatch`; the profiler does not create a second counter.
- `layer:<n>` uses the one-based physical D/E sublayer ID. For this V4.1
  composition, `hf=(n-1)//2` identifies the zero-based released HF block.
- `phase=forward`, `phase=recompute` and `phase=backward` identify execution.
  `checkpointed_hybrid_forward` wraps the callable passed to the checkpoint
  engine and snapshots host labels. First execution keeps the caller phase;
  subsequent replay explicitly selects recompute. No-grad alone is not used
  to classify arbitrary model calls.
- `comp:<name>` attributes attention projection/compressor/indexer/native or
  fused attention, mHC mapping/projection/Sinkhorn/aggregation/BDA, and MoE
  router/preprocess/dispatch/experts/combine. Bridge must separately instrument
  its actual Engram hash, lookup, projection and gate methods.
- CP boundary/shared-KV, Engram A2A/count wait, MoE count wait, typed PP and
  replay's CPU/Gloo rendezvous have coarse communication ranges. No range is
  inserted for each query chunk, tensor operation or Sinkhorn iteration.

Backward start/end ranges are opened by pre/post hooks on the actual autograd
nodes created by a component, on their execution thread. Nested components
claim their nodes; layer and microbatch scopes are inclusive. Shared dataclass
state outputs are included in layer attribution so outgoing mHC/CSA edges do
not lose their layer labels. Unused nodes close at graph completion, and
retained graphs can be run again. These scopes describe recorded execution
intervals, not additive GPU costs: concurrent kernels, nested scopes and
communication waits need separate interpretation.

The helper uses PyTorch private graph-task/sequence APIs. Test those APIs and
NVTX behavior when upgrading PyTorch. Its host bookkeeping and hooks add
profiling overhead; compare performance with profiling disabled.

## Capture-window integration

EchoCraft's selected-node Nsight wrapper defaults `MCORE_COMPONENT_NVTX` to `1`
in the target subprocess, preserving an explicit `0`. The Swift callback calls
`set_capture_context(False)` before training, enables it with the absolute step
inside the requested window, and disables it afterwards. It wraps the existing
optimizer step only for profiling and restores that method when capture ends.
Older environments without this module retain their original step profiling.

For PP8/CP16/EP32, verify actual node mapping and sample the same DP/CP position
on nodes `0,4,8,12,16,20,24,28`. A suggested full-step window is 12–13 with a
3000-second upper bound, retaining the experiment's no-checkpoint controls.

## Verification and limits

Direct CPU tests exercise the real PyTorch autograd engine with recorded NVTX
calls: identity when disabled, nested output/gradient/RNG parity, original
microbatch on checkpoint replay, backward on another thread, unused outputs,
retained graphs and exception cleanup. A separate real Nsight CPU smoke checks
the push/pop and start/end events, without substituting the NVTX frontend.

The primary target is the current full/uniform, no-VPP, no-CUDA-Graph V4.1 SFT
recipe. Selective CheckpointWithoutOutput and graph replay phase attribution
are not qualified by these tests. Native sparse attention's internal backward
query recomputation is included in the component backward range, rather than
emitting an event for every query chunk. Communication ranges outside model
execution do not invent absent microbatch/layer coordinates.

CPU tests and CPU traces cannot qualify TE/FlashMLA/CUDA kernel dispatch, GPU
overhead, distributed H200/CP16/PP8 behavior or full-model memory. Validate a
small GPU forward/backward and a two-step collection that continues training
before using the instrumented full model. This adaptation does not change the
existing setup pin or an installed production prefix.
