/* =============================================================================
 * BreathHead C ↔ PyTorch parity test
 * =============================================================================
 *
 * Reads a binary test fixture written by export_test_fixture.py:
 *
 *     [4 bytes]  magic "BHTV"  (Breath Head Test Vector)
 *     [4 bytes]  uint32 version (1)
 *     [4 bytes]  uint32 hidden
 *     [4 bytes]  uint32 n_weight_floats
 *     [4 bytes]  uint32 n_frames
 *     [n_weight_floats * 4 bytes]  float32 weights (in export_breath_head order)
 *     [n_frames * 384 * 4 bytes]   float32 input (concat features per frame)
 *     [n_frames * 4 bytes]         float32 expected output (PyTorch p(breath))
 *
 * Runs C inference frame-by-frame, prints max absolute diff vs PyTorch.
 * Returns 0 on success (max diff < 1e-5), 1 on parity failure, 2 on I/O error.
 *
 *     cc -O2 -o test_breath_head test_breath_head.c breath_head.c -lm
 *     ./test_breath_head fixture.bin
 */

#include "breath_head.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* Read exactly `n` bytes into `dst`; return 0 on success, -1 on short read. */
static int read_exact(FILE *f, void *dst, size_t n) {
    size_t got = fread(dst, 1, n, f);
    return (got == n) ? 0 : -1;
}


int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <fixture.bin>\n", argv[0]);
        return 2;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 2;
    }

    /* ── Header ─────────────────────────────────────────────────── */
    char magic[4];
    uint32_t version, hidden, n_weights, n_frames;
    if (read_exact(f, magic, 4) ||
        read_exact(f, &version, 4) ||
        read_exact(f, &hidden, 4) ||
        read_exact(f, &n_weights, 4) ||
        read_exact(f, &n_frames, 4)) {
        fprintf(stderr, "Failed to read header\n");
        fclose(f);
        return 2;
    }

    if (memcmp(magic, "BHTV", 4) != 0) {
        fprintf(stderr, "Bad magic: expected 'BHTV', got '%.4s'\n", magic);
        fclose(f);
        return 2;
    }
    if (version != 1) {
        fprintf(stderr, "Unsupported version %u\n", version);
        fclose(f);
        return 2;
    }

    printf("Fixture: hidden=%u, weights=%u floats, frames=%u\n",
           hidden, n_weights, n_frames);

    /* ── Weights ────────────────────────────────────────────────── */
    float *weights = (float *)malloc(n_weights * sizeof(float));
    if (!weights || read_exact(f, weights, n_weights * sizeof(float))) {
        fprintf(stderr, "Failed to read weights\n");
        free(weights);
        fclose(f);
        return 2;
    }

    BreathHeadWeights w;
    if (breath_head_bind_weights(&w, weights, n_weights, (int)hidden) != 0) {
        fprintf(stderr, "breath_head_bind_weights failed\n");
        free(weights);
        fclose(f);
        return 2;
    }

    /* ── Inputs ─────────────────────────────────────────────────── */
    float *inputs = (float *)malloc(n_frames * BH_INPUT_SIZE * sizeof(float));
    if (!inputs ||
        read_exact(f, inputs, n_frames * BH_INPUT_SIZE * sizeof(float))) {
        fprintf(stderr, "Failed to read inputs\n");
        free(weights); free(inputs);
        fclose(f);
        return 2;
    }

    /* ── Expected outputs ───────────────────────────────────────── */
    float *expected = (float *)malloc(n_frames * sizeof(float));
    if (!expected ||
        read_exact(f, expected, n_frames * sizeof(float))) {
        fprintf(stderr, "Failed to read expected outputs\n");
        free(weights); free(inputs); free(expected);
        fclose(f);
        return 2;
    }
    fclose(f);

    /* ── Streaming inference ────────────────────────────────────── */
    BreathHeadState st;
    breath_head_reset_state(&st);

    float *actual = (float *)malloc(n_frames * sizeof(float));
    for (uint32_t t = 0; t < n_frames; t++) {
        actual[t] = breath_head_process_frame(
            &w, &st, &inputs[t * BH_INPUT_SIZE]);
    }

    /* ── Compare (skip the warm-up window where C returns 0) ────── */
    float max_diff = 0.0f;
    int max_diff_frame = -1;
    int compared = 0;
    for (uint32_t t = BH_WARMUP_FRAMES + 1; t < n_frames; t++) {
        float d = fabsf(actual[t] - expected[t]);
        if (d > max_diff) {
            max_diff = d;
            max_diff_frame = (int)t;
        }
        compared++;
    }

    printf("Compared %d frames (skipped first %d warm-up frames)\n",
           compared, BH_WARMUP_FRAMES + 1);
    printf("Max |C - PyTorch| = %.3e at frame %d\n",
           max_diff, max_diff_frame);

    if (max_diff_frame >= 0) {
        printf("  C output[%d]:       %.8f\n",
               max_diff_frame, actual[max_diff_frame]);
        printf("  PyTorch output[%d]: %.8f\n",
               max_diff_frame, expected[max_diff_frame]);
    }

    /* Spot-check a few frames */
    printf("\nSample frames (warm-up + 5):\n");
    for (uint32_t t = BH_WARMUP_FRAMES + 1; t < BH_WARMUP_FRAMES + 6 && t < n_frames; t++) {
        printf("  frame %3u: C=%.6f  PyTorch=%.6f  diff=%.2e\n",
               t, actual[t], expected[t], fabsf(actual[t] - expected[t]));
    }

    free(weights); free(inputs); free(expected); free(actual);

    /* Pass: max diff under 1e-5 (acceptable for single-precision float) */
    if (max_diff < 1e-5f) {
        printf("\nPASS: C and PyTorch agree within 1e-5\n");
        return 0;
    } else {
        printf("\nFAIL: max diff %.3e exceeds 1e-5\n", max_diff);
        return 1;
    }
}
