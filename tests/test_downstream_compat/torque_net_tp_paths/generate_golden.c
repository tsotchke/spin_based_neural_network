/*
 * tests/test_downstream_compat/torque_net_tp_paths/generate_golden.c
 *
 * Generates the golden-vector JSON files for the torque-net ↔ libirrep
 * NequIP convergence suite. Runs the current hand-rolled torque_net
 * forward pass on 5 fixed configurations and emits:
 *
 *   configs.json           — input graph + node features + weights
 *   expected_outputs.json  — reference τ outputs + CG-scalar prefactors
 *
 * Build / run:
 *   gcc -Wall -std=c11 -Iinclude -O2 \
 *       tests/test_downstream_compat/torque_net_tp_paths/generate_golden.c \
 *       src/equivariant_gnn/torque_net.c -lm -o /tmp/gen_golden
 *   /tmp/gen_golden tests/test_downstream_compat/torque_net_tp_paths/
 *
 * Outputs to the supplied directory. JSON kept deliberately minimal
 * (no indentation framework) so the files are diffable and vendorable
 * into both trees with no dependency on a JSON library on either side.
 *
 * Pinning convention: once libirrep 1.2 lands with NequIP layers, the
 * migrated torque_net implementation is expected to produce the *same*
 * expected_outputs.json to bit-equal precision (given the
 * documented {−1/√2, 1/√3} prefactor book-keeping). Any deviation is
 * a convention-drift signal that fires on both test trees.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "equivariant_gnn/torque_net.h"
typedef struct {
    const char *name;
    const char *description;
    int        Lx, Ly, periodic;
    /* Node features: N × 3 unit vectors, as triples. */
    int        num_nodes;
    const double *node_features;
    /* Parameter block. */
    torque_net_params_t params;
} golden_config_t;
/* --- Five fixed configurations ------------------------------------ */
/* Config 1: minimal — 2×2 open grid, two neighbours aligned, two
 * orthogonal. Smallest non-trivial case. */
static const double cfg1_m[] = {
    1.0, 0.0, 0.0,       /* s0: +x̂ */
    0.0, 1.0, 0.0,       /* s1: +ŷ */
    0.0, 0.0, 1.0,       /* s2: +ẑ */
    1.0/M_SQRT2, 0.0, 1.0/M_SQRT2   /* s3: (+x̂+ẑ)/√2 */
};
/* Config 2: tilted — all four spins rotated ~30° off ẑ toward various
 * horizontal directions. Exercises multi-axis cross products. */
static const double cfg2_m[] = {
    0.5, 0.0, 0.8660254037844387,    /* 30° tilt */
    0.0, 0.5, 0.8660254037844387,
   -0.5, 0.0, 0.8660254037844387,
    0.0,-0.5, 0.8660254037844387
};
/* Config 3: antiferromagnetic Néel on 2×2 open — alternating ±ẑ.
 * Tests that antisymmetric cross-products produce the expected
 * staggered torques. */
static const double cfg3_m[] = {
    0.0, 0.0,  1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0,  1.0
};
/* Config 4: skyrmion-like radial twist on 3×3 periodic. Tests SO(3)
 * equivariance non-trivially. Uses a Bloch-skyrmion ansatz:
 *     m(r) = (sin θ cos φ, sin θ sin φ, cos θ),   θ ∝ r,  φ ∝ polar. */
static double cfg4_m_storage[9 * 3];
static void build_skyrmion_cfg4(void) {
    int Lx = 3, Ly = 3;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++) {
        double dx = x - 1.0, dy = y - 1.0;
        double r   = sqrt(dx * dx + dy * dy);
        double phi = atan2(dy, dx);
        double theta = M_PI * r / 2.0;   /* 0 at centre, π/2 at edge */
        int idx = x * Ly + y;
        cfg4_m_storage[3 * idx + 0] = sin(theta) * cos(phi);
        cfg4_m_storage[3 * idx + 1] = sin(theta) * sin(phi);
        cfg4_m_storage[3 * idx + 2] = cos(theta);
    }
}
/* Config 5: stress test — 4×4 periodic, pseudo-random but deterministic
 * unit vectors sampled from a splitmix-seeded golden-ratio hash so both
 * trees generate bit-equal node features. */
static double cfg5_m_storage[16 * 3];
static void build_stress_cfg5(void) {
    unsigned long long s = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 16; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u1 = (double)(s >> 11) / 9007199254740992.0;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u2 = (double)(s >> 11) / 9007199254740992.0;
        double theta = 2.0 * M_PI * u2;
        double z = 2.0 * u1 - 1.0;
        double sp = sqrt(1.0 - z * z);
        cfg5_m_storage[3 * i + 0] = sp * cos(theta);
        cfg5_m_storage[3 * i + 1] = sp * sin(theta);
        cfg5_m_storage[3 * i + 2] = z;
    }
}
static void emit_config_section(FILE *f, int idx, const golden_config_t *c) {
    int N = c->num_nodes;
    fprintf(f, "    \"%s\": {\n", c->name);
    fprintf(f, "      \"description\": \"%s\",\n", c->description);
    fprintf(f, "      \"grid\": {\"Lx\": %d, \"Ly\": %d, \"periodic\": %d},\n",
            c->Lx, c->Ly, c->periodic);
    fprintf(f, "      \"num_nodes\": %d,\n", N);
    fprintf(f, "      \"node_features\": [");
    for (int i = 0; i < N; i++) {
        if (i) fprintf(f, ", ");
        fprintf(f, "[%.17g, %.17g, %.17g]",
                c->node_features[3 * i + 0],
                c->node_features[3 * i + 1],
                c->node_features[3 * i + 2]);
    }
    fprintf(f, "],\n");
    fprintf(f, "      \"tp_weights\": {\"w0\": %.17g, \"w1\": %.17g, "
              "\"w2\": %.17g, \"w3\": %.17g, \"w4\": %.17g},\n",
            c->params.w0, c->params.w1, c->params.w2,
            c->params.w3, c->params.w4);
    fprintf(f, "      \"radial\": {\"r_cut\": %.17g, \"order\": %.17g}\n",
            c->params.r_cut, c->params.radial_order);
    fprintf(f, "    }%s\n", (idx < 4) ? "," : "");
}
static void emit_output_section(FILE *f, int idx, const char *name,
                                 int N, const double *tau) {
    fprintf(f, "    \"%s\": {\n", name);
    fprintf(f, "      \"num_nodes\": %d,\n", N);
    fprintf(f, "      \"torque\": [");
    for (int i = 0; i < N; i++) {
        if (i) fprintf(f, ", ");
        fprintf(f, "[%.17g, %.17g, %.17g]",
                tau[3 * i + 0], tau[3 * i + 1], tau[3 * i + 2]);
    }
    fprintf(f, "]\n");
    fprintf(f, "    }%s\n", (idx < 4) ? "," : "");
}
int main(int argc, char **argv) {
    const char *outdir = argc > 1 ? argv[1]
                         : "tests/test_downstream_compat/torque_net_tp_paths";
    char configs_path[512], expected_path[512];
    snprintf(configs_path, sizeof(configs_path), "%s/configs.json", outdir);
    snprintf(expected_path, sizeof(expected_path), "%s/expected_outputs.json", outdir);
    build_skyrmion_cfg4();
    build_stress_cfg5();
    /* Common-sense TP weight sets per config. Chosen to exercise every
     * basis term at a non-trivial magnitude. */
    torque_net_params_t pA = {
.w0 =  0.5,.w1 = -0.7,.w2 =  0.3,.w3 =  0.1,.w4 = -0.2,
.r_cut = 1.5,.radial_order = 6.0
    };
    torque_net_params_t pB = {
.w0 =  0.2,.w1 =  1.0,.w2 = -0.5,.w3 =  0.4,.w4 =  0.0,
.r_cut = 1.5,.radial_order = 6.0
    };
    torque_net_params_t pC = {
.w0 = -1.0,.w1 =  0.0,.w2 =  0.8,.w3 = -0.3,.w4 =  0.5,
.r_cut = 1.5,.radial_order = 6.0
    };
    torque_net_params_t pD = {
.w0 =  0.37,.w1 = -1.15,.w2 =  0.42,.w3 =  0.89,.w4 = -0.50,
.r_cut = 1.5,.radial_order = 6.0
    };
    golden_config_t cfgs[5] = {
        { "config_01_minimal_open_2x2",
          "2x2 open grid, node features fixed unit vectors including one at (1,0,1)/sqrt(2).",
          2, 2, 0, 4, cfg1_m, pA },
        { "config_02_tilted_uniform_2x2",
          "2x2 open grid, all spins 30 degrees tilted off zhat in different horizontal directions.",
          2, 2, 0, 4, cfg2_m, pB },
        { "config_03_neel_ztilde_2x2",
          "2x2 open grid, antiferromagnetic Neel alternation along zhat.",
          2, 2, 0, 4, cfg3_m, pC },
        { "config_04_bloch_skyrmion_3x3_periodic",
          "3x3 periodic grid, Bloch-skyrmion ansatz.",
          3, 3, 1, 9, cfg4_m_storage, pD },
        { "config_05_stress_random_4x4_periodic",
          "4x4 periodic grid, deterministic splitmix-seeded unit vectors.",
          4, 4, 1, 16, cfg5_m_storage, pA }
    };
    FILE *cf = fopen(configs_path, "w");
    FILE *ef = fopen(expected_path, "w");
    if (!cf || !ef) {
        fprintf(stderr, "cannot open output files\n");
        if (cf) fclose(cf); if (ef) fclose(ef);
        return 1;
    }
    fprintf(cf, "{\n");
    fprintf(cf, "  \"_version\": \"1\",\n");
    fprintf(cf, "  \"_note\": \"Golden-vector input configurations. Frozen alongside libirrep 1.2.\",\n");
    fprintf(cf, "  \"configs\": {\n");
    fprintf(ef, "{\n");
    fprintf(ef, "  \"_version\": \"1\",\n");
    fprintf(ef, "  \"_note\": \"Expected TP outputs from the hand-rolled torque_net at this commit. libirrep 1.2 NequIP path must match bit-exactly under documented {-1/sqrt(2), 1/sqrt(3)} prefactor book-keeping. Generated by generate_golden.c on current tree.\",\n");
    fprintf(ef, "  \"prefactor_conventions\": {\n");
    fprintf(ef, "    \"basis_0_mj_dot_rhat_mi\": \"CG scalar path (1o x 1o)_0 contributes -1/sqrt(3).\",\n");
    fprintf(ef, "    \"basis_1_mj_cross_rhat\": \"CG axial path (1o x 1o)_1 contributes 1/sqrt(2).\",\n");
    fprintf(ef, "    \"basis_2_mi_cross_mj\": \"Node-centred antisymmetric (1o x 1o)_1 contributes 1/sqrt(2).\",\n");
    fprintf(ef, "    \"basis_3_mi_dot_mj_mi\": \"(0e x 1o)_1 identity path, prefactor unity.\",\n");
    fprintf(ef, "    \"basis_4_mj_identity\": \"Identity path (1o -> 1o), prefactor unity.\"\n");
    fprintf(ef, "  },\n");
    fprintf(ef, "  \"outputs\": {\n");
    for (int i = 0; i < 5; i++) {
        const golden_config_t *c = &cfgs[i];
        int *src, *dst; double *vec; int E;
        torque_net_build_grid(c->Lx, c->Ly, c->periodic, &src, &dst, &vec, &E);
        torque_net_graph_t g = {.num_nodes = c->num_nodes,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
        double *tau = malloc((size_t)3 * c->num_nodes * sizeof(double));
        torque_net_forward(&g, c->node_features, &c->params, tau);
        emit_config_section(cf, i, c);
        emit_output_section(ef, i, c->name, c->num_nodes, tau);
        free(tau); free(src); free(dst); free(vec);
    }
    fprintf(cf, "  }\n");
    fprintf(cf, "}\n");
    fprintf(ef, "  }\n");
    fprintf(ef, "}\n");
    fclose(cf); fclose(ef);
    fprintf(stderr, "wrote %s\nwrote %s\n", configs_path, expected_path);
    return 0;
}