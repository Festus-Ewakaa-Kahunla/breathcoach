/* =============================================================================
 * BreathHead — Standalone Causal Breath-Event Inference (C)
 * =============================================================================
 *
 * A small (~15K-parameter) causal neural network head that consumes the 384-d
 * concatenated features from NanoPitch and produces a per-frame breath
 * probability in [0, 1].
 *
 * Architecture (matches Python BreathHead in prototypes/model/breath_head.py):
 *
 *     in (384) -> Conv1d(k=5, dilation=1) -> ReLU
 *              -> Conv1d(k=5, dilation=2) -> ReLU
 *              -> Linear(hidden, 1)        -> sigmoid -> p(breath)
 *
 * Causal padding: both convolutions only look at past + current frames.
 *   Conv1 needs 4 frames of history (k-1=4)
 *   Conv2 with dilation=2 needs 8 more frames after conv1 output (k-1)*2=8
 *   Total receptive field: 12 past frames + current = 13 frames = 130 ms
 *
 * Streaming convention:
 *   - Caller pushes one 384-d feature vector per audio frame (every 10 ms)
 *   - We maintain a ring buffer of the last 5 conv1-inputs (for k=5 conv1)
 *     and the last 9 conv1-outputs (for k=5 dilation=2 conv2)
 *   - Output is one float (breath probability) per frame
 *   - During the first 12 frames of a stream, output is undefined (warm-up).
 *
 * Why standalone (no NanoPitch dependency):
 *   Lets us unit-test the breath head's math independently of NanoPitch's
 *   inference. Round-trip parity with PyTorch is verified without dragging
 *   in the entire NanoPitch C code. Integration into the joint pipeline is
 *   a separate step.
 */

#ifndef BREATH_HEAD_H
#define BREATH_HEAD_H

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed architecture constants (must match the trained Python model) */
#define BH_INPUT_SIZE    384  /* NanoPitch concat features per frame */
#define BH_KERNEL_SIZE   5    /* conv1 and conv2 kernel width */
#define BH_DILATION_2    2    /* conv2 dilation */
#define BH_MAX_HIDDEN    32   /* upper bound on `hidden` (we use 8 by default) */

/* Receptive-field sizes — left-padding amounts */
#define BH_PAD_CONV1     (BH_KERNEL_SIZE - 1)              /* 4 frames */
#define BH_PAD_CONV2     ((BH_KERNEL_SIZE - 1) * BH_DILATION_2) /* 8 frames */
#define BH_WARMUP_FRAMES (BH_PAD_CONV1 + BH_PAD_CONV2)      /* 12 frames */

/* Ring buffer sizes */
#define BH_INPUT_BUF_LEN  (BH_PAD_CONV1 + 1)   /* 5 = need k=5 conv input frames */
#define BH_CONV1_BUF_LEN  (BH_PAD_CONV2 + 1)   /* 9 = need k=5 dil=2 -> 9 frames span */

/**
 * BreathHeadWeights — read-only learned parameters.
 *
 * Conv1: weight shape [hidden][BH_INPUT_SIZE][BH_KERNEL_SIZE]
 *   bias  shape [hidden]
 * Conv2: weight shape [hidden][hidden][BH_KERNEL_SIZE]
 *   bias  shape [hidden]
 * Head:  weight shape [1][hidden]
 *   bias  shape [1]
 *
 * Memory layout for each tensor is row-major (C-contiguous), matching how
 * PyTorch's tensor.numpy().flatten() serializes — so the export from
 * export_breath_head.py loads correctly with simple pointer assignment.
 */
typedef struct {
    int hidden;          /* hidden channel count (typically 8) */

    const float *conv1_weight;  /* [hidden * BH_INPUT_SIZE * BH_KERNEL_SIZE] */
    const float *conv1_bias;    /* [hidden] */

    const float *conv2_weight;  /* [hidden * hidden * BH_KERNEL_SIZE] */
    const float *conv2_bias;    /* [hidden] */

    const float *head_weight;   /* [hidden] (linear 1xhidden) */
    const float *head_bias;     /* [1] */
} BreathHeadWeights;

/**
 * BreathHeadState — mutable per-stream inference state.
 *
 * Two ring buffers — one for the conv1 inputs (last 5 cat-feature frames),
 * one for the conv1 outputs (last 9 hidden frames, enough for k=5 dil=2 conv2).
 * Positions are monotonic counters; we use (pos % buffer_len) to wrap.
 *
 * Buffer sizing is conservative for BH_MAX_HIDDEN=32 so the struct can be
 * stack-allocated by callers.
 */
typedef struct {
    float input_buf[BH_INPUT_BUF_LEN][BH_INPUT_SIZE];  /* recent cat features */
    float conv1_buf[BH_CONV1_BUF_LEN][BH_MAX_HIDDEN];  /* recent conv1 outputs */
    int   input_pos;    /* monotonic count of pushed input frames */
    int   conv1_pos;    /* monotonic count of conv1 output frames generated */
    int   frame_count;  /* total frames processed (for warm-up detection) */
} BreathHeadState;

/* ─── Public API ──────────────────────────────────────────────────────── */

/**
 * Bind a flat float buffer (as exported by export_breath_head.py --format binary
 * or by JSON.parse + Float32Array copy) to a BreathHeadWeights struct.
 *
 * The buffer is NOT copied; the weights object holds pointers into it.
 * The caller owns the float buffer and must keep it alive for the lifetime
 * of the weights object.
 *
 * Expected layout (matches export_breath_head.py):
 *   1. conv1_weight  [hidden * BH_INPUT_SIZE * BH_KERNEL_SIZE]
 *   2. conv1_bias    [hidden]
 *   3. conv2_weight  [hidden * hidden * BH_KERNEL_SIZE]
 *   4. conv2_bias    [hidden]
 *   5. head_weight   [hidden]
 *   6. head_bias     [1]
 *
 * @return 0 on success, -1 if hidden > BH_MAX_HIDDEN or buffer too small.
 */
int breath_head_bind_weights(BreathHeadWeights *w,
                             const float *data,
                             int n_floats,
                             int hidden);

/**
 * Initialize a state object to all-zero ring buffers.
 *
 * Call once before streaming; call again to "reset" between streams.
 */
void breath_head_reset_state(BreathHeadState *st);

/**
 * Process one frame: push 384-d input, return breath probability.
 *
 * During the first BH_WARMUP_FRAMES (12) frames the return value is 0
 * (insufficient context). After warm-up, the return value is the sigmoid
 * of the head logit — the probability that the current frame contains a
 * breath event.
 *
 * @param w               Weights (read-only).
 * @param st              State (updated in place).
 * @param concat_features Pointer to BH_INPUT_SIZE (384) floats — the
 *                        concatenated NanoPitch features for this frame.
 * @return                Breath probability in [0, 1], or 0.0 during warm-up.
 */
float breath_head_process_frame(const BreathHeadWeights *w,
                                BreathHeadState *st,
                                const float *concat_features);

#ifdef __cplusplus
}
#endif

#endif /* BREATH_HEAD_H */
