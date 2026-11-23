# V4.1 Hybrid recompute synchronization

`TransformerConfig.hybrid_recompute_sync` is an opt-in workaround for a
cross-communicator ordering stall observed during V4.1 full recomputation.
It defaults to `False` and does not select an mHC or attention backend.

## Observed failure

With CP16, EP32 and PP8, ranks in one pipeline stage entered adjacent replay
groups at different times. Some ranks started the next group's MoE token-count
AllGather while peers remained in the previous group's CP boundary backward or
compiled mHC backward autotuning. NCCL RAS reported a persistent AllGather
operation-count mismatch on the same 32-rank communicator. Pipeline stages
waiting for that backward then stopped making optimizer progress.

A small CP/EP collective-order reproducer can time out when ranks issue these
operations in opposing orders, including after communicator preconnection.
Small real Hybrid models did not naturally reproduce the original full-model
trigger. The observations establish the stalled dependencies; they do not
establish every step of the initial scheduling divergence.

## Configuration

Enable the option for a static V4.1 HybridStack with uniform full recomputation:

```python
config.hybrid_recompute_sync = True
config.recompute_granularity = "full"
config.recompute_method = "uniform"
config.recompute_num_layers = 2
```

Supply `ProcessGroupCollection.dp_cp_gloo`, using a CPU Gloo group whose members
include every rank in the stack's CP and EP groups. The MPU compatibility
constructor exposes the existing data/context Gloo group when it was created
during model-parallel initialization. It returns `None` when that group was not
created; enabling synchronization then fails during stack construction. No new
communicators are created by this option.

Before each checkpoint replay, ranks perform a CPU AllGather of the first and
last layer numbers. A mismatch raises an error. This host rendezvous avoids
adding a CUDA barrier or device-wide synchronization while other communicator
operations are outstanding. Original no-grad forward and groups without
differentiable checkpoint inputs skip the rendezvous. Tensor arithmetic and
checkpoint boundary tensors are unchanged.

The option requires static CP, uniform full recomputation, and no virtual
pipeline stages or CUDA Graphs. All participating ranks must enable it and
enter compatible replay boundaries. The identity check contains layer numbers,
not microbatch IDs; it is not a general reordering scheduler for arbitrary
autograd graphs. Existing process-group timeouts still apply.

## Validation and limits

- CPU tests check optional group materialization, replay-only placement,
  default-disabled parity, frozen inputs, and mismatched layer identities.
- Eight H200 ranks exercised the real checkpoint, CP halo and EP collective
  paths. A small real PP6 HybridStack with CSA, mHC and MoE passed output,
  FP32-gradient and three CPU AdamW update comparisons in deterministic mode.
- A new-process small-model restore of model, FP32 masters, Adam moments/step,
  RNG and inputs passed a subsequent-update comparison at `rtol=2e-6`,
  `atol=2e-7`. This test used ordinary CPU AdamW. A separate precision-aware
  CPU-offload/DCP test used an earlier barrier candidate without exercising
  Hybrid replay and does not replace full-model checkpoint validation.
- The matching production implementation ran on 32 x 8 H200 with
  TP1/PP8/CP16/EP32/ETP1/DP2, GBS64, MBS1 and a 327680-token limit. It retained
  fused mHC, `torch.compile`, native CSA and frozen FP8 Engram tables. All 256
  ranks completed 51 real precision-aware CPU Adam updates. The run was
  deliberately cancelled at 51/66 steps and saved no checkpoint.

Deterministic mode was used only for small-model numerical comparisons, not
for the full-model run. The 51-step result demonstrates progress under this
recipe; it does not prove arbitrary topology support, full-model cold restore,
or removal of every possible communication stall. The rendezvous adds host
latency and waits for slower ranks. No isolated synchronization speedup is
claimed. With EP16, the existing DP x CP synchronization group would still span
32 ranks under the same ordinary TP/PP/CP/DP layout.
