/*
 * tests/test_downstream_compat/test_torque_net_golden.c
 *
 * Runtime verifier: re-generates the torque_net outputs on the five
 * golden-vector configurations and asserts bit-exact agreement with
 * the committed expected values. Fires the moment *our* side drifts.
 *
 * The mirror of this test will live in libirrep's tree as
 * `tests/test_downstream_compat/` with the same JSON files vendored;
 * it asserts the libirrep NequIP path produces the same outputs.
 *
 * JSON parsing is handled by a minimal embedded parser — we don't
 * pull in a JSON library because both trees have to be able to run
 * this test with zero external deps.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "equivariant_gnn/torque_net.h"
/* The five configs are redefined here to avoid parsing JSON. The
 * configs.json file is the authoritative vendorable artifact; this
 * test mirrors it for runtime checking. If they ever diverge,
 * regenerate via generate_golden.c. */
typedef struct {
    const char *name;
    int Lx, Ly, periodic;
    int num_nodes;
    const double *node_features;
    torque_net_params_t params;
    const double *expected_torque;    /* 3 × num_nodes */
} golden_ref_t;
/* Node features: copies of the cfg*_m arrays in generate_golden.c. */
static const double g_cfg1_m[] = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0/M_SQRT2, 0.0, 1.0/M_SQRT2
};
static const double g_cfg2_m[] = {
    0.5, 0.0, 0.8660254037844387,
    0.0, 0.5, 0.8660254037844387,
   -0.5, 0.0, 0.8660254037844387,
    0.0,-0.5, 0.8660254037844387
};
static const double g_cfg3_m[] = {
    0.0, 0.0,  1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0,  1.0
};
static double g_cfg4_m[9 * 3];
static double g_cfg5_m[16 * 3];
static void build_cfg4(void) {
    for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) {
        double dx = x - 1.0, dy = y - 1.0;
        double r   = sqrt(dx * dx + dy * dy);
        double phi = atan2(dy, dx);
        double theta = M_PI * r / 2.0;
        int idx = x * 3 + y;
        g_cfg4_m[3*idx+0] = sin(theta)*cos(phi);
        g_cfg4_m[3*idx+1] = sin(theta)*sin(phi);
        g_cfg4_m[3*idx+2] = cos(theta);
    }
}
static void build_cfg5(void) {
    unsigned long long s = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 16; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u1 = (double)(s >> 11) / 9007199254740992.0;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u2 = (double)(s >> 11) / 9007199254740992.0;
        double theta = 2.0 * M_PI * u2;
        double z = 2.0 * u1 - 1.0;
        double sp = sqrt(1.0 - z * z);
        g_cfg5_m[3*i+0] = sp * cos(theta);
        g_cfg5_m[3*i+1] = sp * sin(theta);
        g_cfg5_m[3*i+2] = z;
    }
}
/* Expected values are long. For brevity, we embed them as flat arrays
 * and check pointer-element-wise; the readable / vendorable authoritative
 * source is expected_outputs.json. */
static const double expect_1[] = {
    0.26588934613625981, -0.63813443072702358, 0.053177869227251956,
    -0.068753306514862653, -0.07520486387928256, 0.024699317210801378,
    0.081656421243702482, 0.27234090350067974, -0.5667088016533901,
    -0.45846344579606157, -0.21916303427342779, -0.33920459261271785
};
static const double expect_2[] = {
    -0.21245571945632222, 0.8059342492237217, 0.16379459181556988,
    0.11513346417481739, 0.47834506559258205, 0.29673926488369978,
    -0.65543412471496487, -0.8059342492237217, 0.071687820475715952,
    0.80593424922372181, -0.47834506559258205, 0.82851795715621945
};
static const double expect_3[] = {
    0, 0, -0.21271147690900785,
    0, 0,  0.21271147690900785,
    0, 0,  0.21271147690900785,
    0, 0, -0.21271147690900785
};
static const double expect_4[] = {
    0.28278745382144416, -0.49193470654543614, -0.076716855834367395,
   -0.050632185615887643, -0.65876100326777998, 0.056208937088550584,
   -0.49193470654543614, -0.2827874538214441, -0.07671685583436752,
    0.65876100326777987, -0.05063218561588758, 0.056208937088550577,
   -9.3950148681928565e-17, 0, 0.78703246456332898,
   -0.65876100326777998, 0.050632185615887483, 0.056208937088550459,
    0.49193470654543608, 0.28278745382144421, -0.076716855834367312,
    0.050632185615887601, 0.65876100326777987, 0.056208937088550646,
   -0.2827874538214441, 0.49193470654543608, -0.076716855834367312
};
static const double expect_5[] = {
   -0.25773571401565382, -0.45384632678711534, -0.018103366957173392,
   -0.35617383348345694, 0.54659536359912209, 0.34411785270280965,
    0.19476585767633267, -0.36755382979001494, -0.058455065166122554,
    0.73787890599312511, -0.27151752348925251, -0.26862719537797469,
   -0.54985732768456363, 0.34545973133983937, -0.062994499366182122,
   -0.24441757236120801, -0.58118605790704836, -1.0468698363082267,
    0.20545247270759762, 0.25487526109568454, -0.20744818989180633,
    0.39944560904912241, 0.87043500909713167, 0.74397206422724571,
    0.61587914166603852, 0.52416887511033383, -0.17923646014534589,
    0.43430627273471689, -0.68726234283068943, -0.36574222311160076,
   -0.46129384285775443, 0.35666362586953254, 0.37214134173230717,
   -0.62606353748631871, 0.55905662364869479, 0.31996008807188075,
    0.48431463814232734, -0.21317894968777157, 0.40441715173513515,
   -0.28182014640660369, 0.25301553044317188, 0.84160609647215578,
   -0.1788605477012947, 0.059534642717599923, 0.45777432771501891,
    0.25020026144633045, -0.45741414288115967, -0.43840989466757202
};
static void run_golden(const golden_ref_t *g) {
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(g->Lx, g->Ly, g->periodic, &src, &dst, &vec, &E);
    torque_net_graph_t gg = {.num_nodes = g->num_nodes,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    double *tau = malloc((size_t)3 * g->num_nodes * sizeof(double));
    torque_net_forward(&gg, g->node_features, &g->params, tau);
    int N3 = 3 * g->num_nodes;
    double max_err = 0;
    for (int i = 0; i < N3; i++) {
        double d = fabs(tau[i] - g->expected_torque[i]);
        if (d > max_err) max_err = d;
    }
    printf("# %s: max |τ − τ_golden| = %.3e\n", g->name, max_err);
    ASSERT_TRUE(max_err < 1e-14);
    free(tau); free(src); free(dst); free(vec);
}
static void test_golden_config_01(void) {
    torque_net_params_t p = { 0.5, -0.7, 0.3, 0.1, -0.2, 1.5, 6.0 };
    golden_ref_t g = { "config_01_minimal_open_2x2", 2, 2, 0, 4,
                       g_cfg1_m, p, expect_1 };
    run_golden(&g);
}
static void test_golden_config_02(void) {
    torque_net_params_t p = { 0.2, 1.0, -0.5, 0.4, 0.0, 1.5, 6.0 };
    golden_ref_t g = { "config_02_tilted_uniform_2x2", 2, 2, 0, 4,
                       g_cfg2_m, p, expect_2 };
    run_golden(&g);
}
static void test_golden_config_03(void) {
    torque_net_params_t p = { -1.0, 0.0, 0.8, -0.3, 0.5, 1.5, 6.0 };
    golden_ref_t g = { "config_03_neel_ztilde_2x2", 2, 2, 0, 4,
                       g_cfg3_m, p, expect_3 };
    run_golden(&g);
}
static void test_golden_config_04(void) {
    torque_net_params_t p = { 0.37, -1.15, 0.42, 0.89, -0.50, 1.5, 6.0 };
    golden_ref_t g = { "config_04_bloch_skyrmion_3x3_periodic", 3, 3, 1, 9,
                       g_cfg4_m, p, expect_4 };
    run_golden(&g);
}
static void test_golden_config_05(void) {
    torque_net_params_t p = { 0.5, -0.7, 0.3, 0.1, -0.2, 1.5, 6.0 };
    golden_ref_t g = { "config_05_stress_random_4x4_periodic", 4, 4, 1, 16,
                       g_cfg5_m, p, expect_5 };
    run_golden(&g);
}
int main(void) {
    build_cfg4();
    build_cfg5();
    TEST_RUN(test_golden_config_01);
    TEST_RUN(test_golden_config_02);
    TEST_RUN(test_golden_config_03);
    TEST_RUN(test_golden_config_04);
    TEST_RUN(test_golden_config_05);
    TEST_SUMMARY();
}