/*
 * src/qec_decoder/qec_decoder.c
 *
 * v0.4 implementation: all decoder kinds fall back to the greedy
 * matching decoder that already lives in src/toric_code.c. The public
 * API is in place so learned decoders (transformer, Mamba) can slot
 * in behind the same interface in v0.5.
 */
#include <stdlib.h>
#include <string.h>
#include "qec_decoder/qec_decoder.h"

qec_decoder_t qec_decoder_create(qec_decoder_kind_t kind) {
    qec_decoder_t d;
    d.kind = kind;
    /* GREEDY and MWPM are both shipped baselines — always available.
     * Learned decoders (TRANSFORMER, MAMBA) require the external NN
     * engine; without it, they transparently fall back to MWPM so the
     * caller still gets optimal-at-small-K matching behaviour. */
    if (kind == QEC_DECODER_GREEDY || kind == QEC_DECODER_MWPM) {
        d.is_available = 1;
    } else {
        d.kind = QEC_DECODER_MWPM;
        d.is_available = 0;
    }
    return d;
}

int qec_decoder_tokenize(const ToricCode *code,
                         qec_syndrome_token_t *out_tokens,
                         int token_capacity) {
    if (!code || !out_tokens || token_capacity <= 0) return -1;
    int Lx = code->size_x, Ly = code->size_y;
    int written = 0;
    for (int x = 0; x < Lx && written < token_capacity; x++) {
        for (int y = 0; y < Ly && written < token_capacity; y++) {
            int idx = x * Ly + y;
            if (code->plaquette_syndrome[idx]) {
                out_tokens[written++] = (qec_syndrome_token_t){
                    .stab_type = 0, .x = x, .y = y, .time_slice = 0};
            }
        }
    }
    for (int x = 0; x < Lx && written < token_capacity; x++) {
        for (int y = 0; y < Ly && written < token_capacity; y++) {
            int idx = x * Ly + y;
            if (code->vertex_syndrome[idx]) {
                out_tokens[written++] = (qec_syndrome_token_t){
                    .stab_type = 1, .x = x, .y = y, .time_slice = 0};
            }
        }
    }
    return written;
}

int qec_decoder_run(const qec_decoder_t *dec, ToricCode *code) {
    if (!dec || !code) return -1;
    switch (dec->kind) {
        case QEC_DECODER_GREEDY:
            return toric_code_decode_greedy(code);
        case QEC_DECODER_MWPM:
        case QEC_DECODER_TRANSFORMER:  /* fallback when no NN engine */
        case QEC_DECODER_MAMBA:
        default:
            return toric_code_decode_mwpm(code);
    }
}

int qec_decoder_logical_error_rate(const qec_decoder_t *dec,
                                   int distance,
                                   double p,
                                   int num_trials,
                                   unsigned rng_seed,
                                   double *out_rate) {
    if (!dec || !out_rate || distance <= 0 || num_trials <= 0) return -1;
    if (p < 0.0 || p > 1.0) return -1;
    srand(rng_seed);
    int num_logical = 0;
    for (int t = 0; t < num_trials; t++) {
        ToricCode *c = initialize_toric_code(distance, distance);
        if (!c) return -1;
        apply_random_errors(c, p);
        qec_decoder_run(dec, c);
        if (toric_code_has_logical_error(c)) num_logical++;
        free_toric_code(c);
    }
    *out_rate = (double)num_logical / (double)num_trials;
    return 0;
}
