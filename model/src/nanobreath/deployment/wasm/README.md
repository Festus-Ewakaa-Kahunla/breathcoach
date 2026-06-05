# BreathHead WASM Deployment

Standalone C implementation of the BreathHead inference, parity-tested
against PyTorch.

## Status

- ✅ **`breath_head.c` + `breath_head.h`** — streaming causal inference
  (conv1 k=5, conv2 k=5 dilation=2, linear head + sigmoid). ~200 LOC, no
  dependencies beyond `<math.h>` and `<string.h>`.
- ✅ **`test_breath_head.c`** — parity test harness reading a binary fixture
  produced by `export_test_fixture.py`.
- ✅ **C ↔ PyTorch parity verified** — max abs diff `8.3e-7` over 37 frames
  (well under the 1e-5 tolerance for single-precision float).
- ⏳ **WASM build** — not yet wired (needs emscripten + `build.sh`).
- ⏳ **Integration with NanoPitch joint pipeline** — needs nanopitch.c to
  expose the 384-d concat features (one-line additionto output struct).
- ⏳ **JS bridge** — needs additions to NanoPitch's web demo to load
  breath_head.json alongside model.json.

## How to run the parity test

```bash
# 1. Export the binary test fixture from a trained PyTorch checkpoint
python export_test_fixture.py /path/to/breath_head_best.pth -o /tmp/fixture.bin

# 2. Build the test program
cc -O2 -Wall -Wextra -o /tmp/test_breath_head test_breath_head.c breath_head.c -lm

# 3. Run — should print "PASS: C and PyTorch agree within 1e-5"
/tmp/test_breath_head /tmp/fixture.bin
```

## Streaming convention

`breath_head_process_frame()` is called once per audio frame (every 10 ms).
The caller provides:
- A pointer to the `BreathHeadWeights` (loaded once)
- A `BreathHeadState` (mutable; one per audio stream)
- A 384-float array of NanoPitch's concatenated `conv2_out + gru1 + gru2 + gru3`
  features for the current frame

The function returns one float — the probability that the current frame
contains a breath event. During the first 12 frames (warm-up), it returns
0 because the ring buffers don't yet contain enough valid context.

## Integration with NanoPitch (TODO)

NanoPitch's `nanopitch_process_frame()` computes the 384-d concat vector
internally (at line ~1361 of `nanopitch.c`) but doesn't expose it. To wire
the breath head in:

1. Add `float concat[4 * NC_MAX_LAYER_SIZE];` to `NanoPitchOutput` in
   `nanopitch.h`.
2. Copy `cat` into `out->concat` just before `dense_sigmoid(...)` in
   `nanopitch_process_frame()`.
3. Caller pattern:
   ```c
   nanopitch_process_frame(nanopitch_weights, nanopitch_state, audio_frame, &out);
   float p_breath = breath_head_process_frame(
       breath_weights, breath_state, out.concat);
   ```

That's the only change required in NanoPitch's code. Two lines.
