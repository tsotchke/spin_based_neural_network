/*
 * include/qec_decoder/qec_decoder.h
 *
 * Learned surface-code decoder scaffold for v0.5 pillar P1.3. Ships
 * with the same interface the transformer / Mamba decoders will
 * satisfy; the default v0.4 implementation falls back to the greedy
 * matching decoder already in src/toric_code.c, which serves as the
 * MWPM-class baseline the learned variants must beat.
 *
 * The syndrome tokenizer prepares inputs for the neural decoder:
 * each flagged stabilizer becomes a token with (type, position_x,
 * position_y, time_slice). A learned decoder consumes the token
 * sequence and emits a correction Pauli string on the data qubits.
 */
#ifndef QEC_DECODER_H
#define QEC_DECODER_H

#include <stddef.h>
#include "toric_code.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QEC_DECODER_GREEDY      = 0, /* greedy nearest-pair matching (fast, suboptimal) */
    QEC_DECODER_MWPM        = 1, /* exact minimum-weight perfect matching for
                                     K ≤ 14 defects; greedy+2-opt for larger. */
    QEC_DECODER_TRANSFORMER = 2, /* learned decoder target; falls back to MWPM */
    QEC_DECODER_MAMBA       = 3  /* learned decoder target; falls back to MWPM */
} qec_decoder_kind_t;

typedef struct {
    int stab_type;   /* 0 = plaquette (X-error syndrome), 1 = vertex (Z) */
    int x, y;         /* lattice coordinates */
    int time_slice;   /* syndrome-history time step (0 for single-shot) */
} qec_syndrome_token_t;

typedef struct {
    qec_decoder_kind_t kind;
    int is_available;       /* 1 if the chosen kind is actually wired in */
} qec_decoder_t;

/* Build a decoder handle. If `kind` requires an external engine that
 * isn't available, the handle falls back to QEC_DECODER_GREEDY and
 * sets is_available accordingly. Always returns a valid handle. */
qec_decoder_t qec_decoder_create(qec_decoder_kind_t kind);

/* Tokenize the current syndromes of a ToricCode into a buffer.
 * Writes up to `token_capacity` tokens and returns the actual count.
 * Returns -1 on argument error. */
int qec_decoder_tokenize(const ToricCode *code,
                         qec_syndrome_token_t *out_tokens,
                         int token_capacity);

/* Run the decoder on the current syndromes of `code`. Modifies the
 * code's data-qubit state to apply corrections. Returns 0 on success.
 *
 * For v0.4 all non-greedy kinds transparently fall back to
 * toric_code_decode_greedy. */
int qec_decoder_run(const qec_decoder_t *dec, ToricCode *code);

/* Monte-Carlo logical-error rate: simulate `num_trials` independent
 * error realisations at physical rate `p`, decode each, and count how
 * many result in a logical error (per `toric_code_has_logical_error`).
 * Writes the fraction into *out_rate. Returns 0 on success. */
int qec_decoder_logical_error_rate(const qec_decoder_t *dec,
                                   int distance,
                                   double p,
                                   int num_trials,
                                   unsigned rng_seed,
                                   double *out_rate);

#ifdef __cplusplus
}
#endif

#endif /* QEC_DECODER_H */
