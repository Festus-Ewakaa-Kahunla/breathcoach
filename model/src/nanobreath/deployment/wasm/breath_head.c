/* =============================================================================
 * BreathHead — Standalone Causal Breath-Event Inference (implementation)
 * =============================================================================
 *
 * See breath_head.h for the streaming convention and architecture overview.
 */

#include "breath_head.h"

#include <math.h>
#include <string.h>


/* ─── Small math helpers ──────────────────────────────────────────────── */

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x) {
    /* Numerically stable form */
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}


/* ─── Weight binding ──────────────────────────────────────────────────── */

int breath_head_bind_weights(BreathHeadWeights *w,
                             const float *data,
                             int n_floats,
                             int hidden) {
    if (hidden < 1 || hidden > BH_MAX_HIDDEN) return -1;

    int expected =
        hidden * BH_INPUT_SIZE * BH_KERNEL_SIZE   /* conv1.weight */
      + hidden                                     /* conv1.bias   */
      + hidden * hidden * BH_KERNEL_SIZE           /* conv2.weight */
      + hidden                                     /* conv2.bias   */
      + hidden                                     /* head.weight  */
      + 1;                                          /* head.bias    */
    if (n_floats < expected) return -1;

    const float *p = data;
    w->hidden       = hidden;
    w->conv1_weight = p; p += hidden * BH_INPUT_SIZE * BH_KERNEL_SIZE;
    w->conv1_bias   = p; p += hidden;
    w->conv2_weight = p; p += hidden * hidden * BH_KERNEL_SIZE;
    w->conv2_bias   = p; p += hidden;
    w->head_weight  = p; p += hidden;
    w->head_bias    = p; p += 1;

    (void)p;  /* suppress unused */
    return 0;
}


/* ─── State management ────────────────────────────────────────────────── */

void breath_head_reset_state(BreathHeadState *st) {
    memset(st, 0, sizeof(*st));
}


/* ─── Forward pass ────────────────────────────────────────────────────── */

/* Look up the k-th past input frame in chronological order
 * (k=0 is the oldest of the BH_INPUT_BUF_LEN stored, k=BH_INPUT_BUF_LEN-1
 * is the most recent / current frame). */
static const float *input_at(const BreathHeadState *st, int k) {
    /* After pushing the current frame, input_pos has incremented to the next
     * slot. The current frame lives at (input_pos - 1) % LEN. The k-th frame
     * counted from the oldest end is (input_pos - LEN + k) % LEN. */
    int idx = (st->input_pos - BH_INPUT_BUF_LEN + k) % BH_INPUT_BUF_LEN;
    if (idx < 0) idx += BH_INPUT_BUF_LEN;
    return st->input_buf[idx];
}

/* Same indexing but for the conv1 output ring buffer.
 * k counted in 1-frame steps; the convolution will sample every DILATION_2 frame. */
static const float *conv1_at(const BreathHeadState *st, int k) {
    int idx = (st->conv1_pos - BH_CONV1_BUF_LEN + k) % BH_CONV1_BUF_LEN;
    if (idx < 0) idx += BH_CONV1_BUF_LEN;
    return st->conv1_buf[idx];
}


float breath_head_process_frame(const BreathHeadWeights *w,
                                BreathHeadState *st,
                                const float *concat_features) {
    int H = w->hidden;
    int K = BH_KERNEL_SIZE;

    /* Step 1: push current input frame into the ring buffer */
    int slot_in = st->input_pos % BH_INPUT_BUF_LEN;
    memcpy(st->input_buf[slot_in], concat_features,
           BH_INPUT_SIZE * sizeof(float));
    st->input_pos++;
    st->frame_count++;

    /* Step 2: run conv1 over the latest BH_INPUT_BUF_LEN=5 frames.
     *
     * Causal conv with left-padding: when frame_count < 5, the older slots
     * still hold zeros (memset by reset_state) which is exactly equivalent
     * to zero-padding on the left.
     *
     * conv1_out[o] = bias[o] + sum_k sum_c weight[o][c][k] * input[k][c]
     * where k ranges over 0..K-1 (oldest to newest stored frames).
     */
    float conv1_out[BH_MAX_HIDDEN];
    for (int o = 0; o < H; o++) {
        float sum = w->conv1_bias[o];
        for (int k = 0; k < K; k++) {
            const float *frame = input_at(st, k);
            const float *wrow = &w->conv1_weight[(o * BH_INPUT_SIZE) * K + k];
            /* For one output channel o and one kernel offset k, walk all 384
             * input channels. The weight layout is [o][c][k] flattened
             * row-major, so for fixed o and k we stride by K to jump between
             * input channels. */
            for (int c = 0; c < BH_INPUT_SIZE; c++) {
                sum += wrow[c * K] * frame[c];
            }
        }
        conv1_out[o] = relu(sum);
    }

    /* Step 3: store conv1 output into its ring buffer */
    int slot_c1 = st->conv1_pos % BH_CONV1_BUF_LEN;
    memcpy(st->conv1_buf[slot_c1], conv1_out, H * sizeof(float));
    st->conv1_pos++;

    /* Warm-up gate: until we have BH_WARMUP_FRAMES of valid history,
     * the convolutions are dominated by zero-padding and the output is
     * not meaningful. */
    if (st->frame_count <= BH_WARMUP_FRAMES) {
        return 0.0f;
    }

    /* Step 4: run conv2 with dilation=2 over the latest 9 conv1 frames.
     *
     * conv2 reads at offsets [0, 2, 4, 6, 8] within the 9-frame window.
     * The 9 most recent conv1 frames live at conv1_at(st, 0..8).
     *
     * conv2_out[o] = bias[o] + sum_k sum_c weight[o][c][k] * conv1_in[k*2][c]
     */
    float conv2_out[BH_MAX_HIDDEN];
    for (int o = 0; o < H; o++) {
        float sum = w->conv2_bias[o];
        for (int k = 0; k < K; k++) {
            int frame_idx = k * BH_DILATION_2;
            const float *frame = conv1_at(st, frame_idx);
            const float *wrow = &w->conv2_weight[(o * H) * K + k];
            for (int c = 0; c < H; c++) {
                sum += wrow[c * K] * frame[c];
            }
        }
        conv2_out[o] = relu(sum);
    }

    /* Step 5: linear head + sigmoid.
     *
     * logit = bias + sum_c head_weight[c] * conv2_out[c]
     * p = sigmoid(logit)
     */
    float logit = w->head_bias[0];
    for (int c = 0; c < H; c++) {
        logit += w->head_weight[c] * conv2_out[c];
    }
    return sigmoid(logit);
}
