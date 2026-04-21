/*
 * scripts/compute_nequip_init.c
 *
 * Reference implementation of the NequIP equivariance-preserving
 * weight-initialisation scheme.
 *
 * For each tensor-product path p with input irrep l_in, spherical-
 * harmonic l_sh, and output irrep l_out, the initial weight
 * distribution is
 *
 *     w_p  ~  TruncNormal(0, σ_p²)     with    σ_p = 1 / √n_paths(l_out)
 *
 * where n_paths(l_out) counts the number of (l_in, l_sh) tensor-product
 * paths contributing to output irrep l_out. This matches e3nn's
 * uvw tensor-product convention and NequIP's Table-2 init, keeping
 * output-feature variance ≈ O(1) per output-irrep channel.
 *
 * Usage:
 *   ./compute_nequip_init --input "2x0e+1x1o" --output "1x1o" --sh 2
 *                         --dump-weights
 *
 * Emits the σ_p array (or optionally a sampled weight vector) to
 * stdout as JSON. Deterministic given --seed. Pure ANSI C, no external
 * dependencies — libirrep's bench can vendor this file directly.
 *
 * This is the "realistic init" complement to the dense-placeholder
 * bench weights libirrep's `bench_downstream_shapes.c` currently uses
 * (w[i] = 0.013 · (i + 1)).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Minimal irrep-parsing support ------------------------------------
 * NequIP paths are determined by the angular-momentum coupling rule
 *     |l_in − l_sh| ≤ l_out ≤ l_in + l_sh,
 * (and parity: output_parity = l_in_parity · l_sh_parity). We parse
 * the simple "Nxl[e|o]" multiset format to enumerate input and output
 * irreps, then count paths per output irrep. */

typedef struct {
    int multiplicity;
    int l;
    int parity;    /* +1 for 'e', -1 for 'o' */
} irrep_term_t;

typedef struct {
    int n_terms;
    irrep_term_t *terms;
} irrep_multiset_t;

static int parse_multiset(const char *spec, irrep_multiset_t *out) {
    /* Simple parser: "Nxl[e|o] [+ ...]". Lenient whitespace around '+'. */
    int cap = 16, n = 0;
    irrep_term_t *terms = malloc((size_t)cap * sizeof(*terms));
    const char *p = spec;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        int mult = 0;
        while (*p >= '0' && *p <= '9') { mult = mult * 10 + (*p - '0'); p++; }
        if (mult <= 0) { free(terms); return -1; }
        if (*p != 'x') { free(terms); return -1; }
        p++;
        int lval = 0;
        while (*p >= '0' && *p <= '9') { lval = lval * 10 + (*p - '0'); p++; }
        int parity;
        if (*p == 'e')      parity = +1;
        else if (*p == 'o') parity = -1;
        else { free(terms); return -1; }
        p++;
        if (n >= cap) { cap *= 2; terms = realloc(terms, (size_t)cap * sizeof(*terms)); }
        terms[n].multiplicity = mult;
        terms[n].l            = lval;
        terms[n].parity       = parity;
        n++;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '+') { p++; continue; }
        if (!*p) break;
        free(terms); return -1;
    }
    out->n_terms = n;
    out->terms = terms;
    return 0;
}

static void free_multiset(irrep_multiset_t *m) { free(m->terms); }

/* --- Per-output-irrep path counting -------------------------------- */

/* Count how many TP paths (u, v, w) produce the given output irrep
 * (l_out, parity_out) given hidden_in multiset and l_sh_max. Each
 * input term (mult_u, l_in, parity_in) and each SH l (with parity
 * (-1)^l) contributes 1 path if the triangle inequality holds and
 * the parity product matches the output parity. The multiplicity-u
 * count does NOT enter the σ_p calculation — σ_p depends on n_paths
 * at the (l_in, l_sh, l_out) level, not on the u/w replicas. */
static int count_paths_for_output(const irrep_multiset_t *hidden_in,
                                   int l_sh_max,
                                   int l_out, int parity_out) {
    int count = 0;
    for (int i = 0; i < hidden_in->n_terms; i++) {
        int l_in      = hidden_in->terms[i].l;
        int parity_in = hidden_in->terms[i].parity;
        for (int l_sh = 0; l_sh <= l_sh_max; l_sh++) {
            int parity_sh = (l_sh % 2 == 0) ? +1 : -1;
            if (parity_in * parity_sh != parity_out) continue;
            int lo = abs(l_in - l_sh);
            int hi = l_in + l_sh;
            if (l_out >= lo && l_out <= hi) count++;
        }
    }
    return count;
}

/* --- CLI ------------------------------------------------------------ */

static void print_usage(void) {
    fprintf(stderr,
            "compute_nequip_init: NequIP equivariance-preserving init helper\n"
            "\n"
            "  --input SPEC       hidden_in multiset, e.g. \"2x0e+1x1o\"\n"
            "  --output SPEC      hidden_out multiset\n"
            "  --sh N             l_sh_max (default 2)\n"
            "  --dump-weights     sample w_p ~ TN(0, σ_p^2) and print\n"
            "  --seed N           RNG seed when sampling (default 1)\n"
            "  --help\n");
}

static double sample_trunc_normal(double sigma, unsigned long long *st) {
    /* Box-Muller, truncate at ±2σ (NequIP's default cutoff). Retry if
     * outside. */
    for (int attempt = 0; attempt < 100; attempt++) {
        *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17;
        double u1 = (double)(*st >> 11) / 9007199254740992.0;
        *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17;
        double u2 = (double)(*st >> 11) / 9007199254740992.0;
        if (u1 < 1e-12) u1 = 1e-12;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
        if (z >= -2.0 && z <= 2.0) return z * sigma;
    }
    return 0.0;
}

int main(int argc, char **argv) {
    const char *input_spec = NULL;
    const char *output_spec = NULL;
    int l_sh_max = 2;
    int dump_weights = 0;
    unsigned long long seed = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_spec = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_spec = argv[++i];
        else if (strcmp(argv[i], "--sh") == 0 && i + 1 < argc) l_sh_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-weights") == 0) dump_weights = 1;
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (unsigned long long)strtoll(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "--help") == 0) { print_usage(); return 0; }
        else { print_usage(); return 1; }
    }
    if (!input_spec || !output_spec) { print_usage(); return 1; }

    irrep_multiset_t hidden_in, hidden_out;
    if (parse_multiset(input_spec, &hidden_in) != 0) {
        fprintf(stderr, "parse error: --input %s\n", input_spec); return 1;
    }
    if (parse_multiset(output_spec, &hidden_out) != 0) {
        fprintf(stderr, "parse error: --output %s\n", output_spec);
        free_multiset(&hidden_in); return 1;
    }

    printf("{\n");
    printf("  \"input_spec\":  \"%s\",\n", input_spec);
    printf("  \"output_spec\": \"%s\",\n", output_spec);
    printf("  \"sh\":          %d,\n", l_sh_max);
    printf("  \"init_rule\":   \"TruncNormal(0, 1/sqrt(n_paths(l_out)))\",\n");
    printf("  \"output_irreps\": [\n");

    unsigned long long rng = seed;
    for (int j = 0; j < hidden_out.n_terms; j++) {
        int l_out      = hidden_out.terms[j].l;
        int parity_out = hidden_out.terms[j].parity;
        int n_paths = count_paths_for_output(&hidden_in, l_sh_max, l_out, parity_out);
        double sigma = n_paths > 0 ? 1.0 / sqrt((double)n_paths) : 0.0;

        printf("    {\n");
        printf("      \"l\":       %d,\n", l_out);
        printf("      \"parity\":  \"%s\",\n", (parity_out == +1) ? "e" : "o");
        printf("      \"n_paths\": %d,\n", n_paths);
        printf("      \"sigma\":   %.17g", sigma);

        if (dump_weights && n_paths > 0) {
            int mult = hidden_out.terms[j].multiplicity;
            int n_weights = n_paths * mult;
            printf(",\n      \"weights\": [");
            for (int k = 0; k < n_weights; k++) {
                if (k) printf((k % 8 == 0) ? ",\n        " : ", ");
                printf("%.17g", sample_trunc_normal(sigma, &rng));
            }
            printf("]");
        }
        printf("\n    }%s\n", (j < hidden_out.n_terms - 1) ? "," : "");
    }
    printf("  ]\n");
    printf("}\n");

    free_multiset(&hidden_in);
    free_multiset(&hidden_out);
    return 0;
}
