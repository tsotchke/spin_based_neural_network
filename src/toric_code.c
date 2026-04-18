#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "toric_code.h"

/* =====================================================================
 * Toric code — physical data-qubit implementation (v0.4).
 *
 * Layout:
 *   - L_x * L_y vertices, same grid of plaquettes (each offset by half a
 *     lattice spacing but shares indexing with vertices).
 *   - 2 * L_x * L_y data qubits, one per link. Link indexing:
 *        link(x, y, 0) — horizontal (east) from vertex (x,y) to (x+1, y)
 *        link(x, y, 1) — vertical   (north) from vertex (x,y) to (x, y+1)
 *     link_index = 2 * (x * L_y + y) + dir
 *
 *   Stabilizers (periodic boundary conditions):
 *     A_v (vertex at (vx, vy)): 4 links
 *        east  = link(vx, vy, 0)
 *        west  = link((vx-1) mod L_x, vy, 0)
 *        north = link(vx, vy, 1)
 *        south = link(vx, (vy-1) mod L_y, 1)
 *     B_p (plaquette labeled by (px, py), unit square with lower-left (px,py)):
 *        bottom = link(px, py, 0)
 *        top    = link(px, (py+1) mod L_y, 0)
 *        left   = link(px, py, 1)
 *        right  = link((px+1) mod L_x, py, 1)
 *
 *   Vertex stabilizer flags when Σ z_errors on its 4 links is odd.
 *   Plaquette stabilizer flags when Σ x_errors on its 4 links is odd.
 *
 *   Logical operators:
 *     L_x^(1): row of horizontal links across y = 0
 *     L_x^(2): column of vertical   links across x = 0
 *     L_z^(1): row of vertical   links across x = 0
 *     L_z^(2): column of horizontal links across y = 0
 * ===================================================================== */

static int wrap(int a, int n) { return (a % n + n) % n; }

int toric_code_link_index(const ToricCode *code, int x, int y, int dir) {
    int Lx = code->size_x, Ly = code->size_y;
    return 2 * (wrap(x, Lx) * Ly + wrap(y, Ly)) + (dir & 1);
}

void toric_code_vertex_links(const ToricCode *code, int vx, int vy, int out_links[4]) {
    out_links[0] = toric_code_link_index(code, vx,            vy,          0); /* east  */
    out_links[1] = toric_code_link_index(code, vx - 1,        vy,          0); /* west  */
    out_links[2] = toric_code_link_index(code, vx,            vy,          1); /* north */
    out_links[3] = toric_code_link_index(code, vx,            vy - 1,      1); /* south */
}

void toric_code_plaquette_links(const ToricCode *code, int px, int py, int out_links[4]) {
    out_links[0] = toric_code_link_index(code, px,            py,          0); /* bottom */
    out_links[1] = toric_code_link_index(code, px,            py + 1,      0); /* top    */
    out_links[2] = toric_code_link_index(code, px,            py,          1); /* left   */
    out_links[3] = toric_code_link_index(code, px + 1,        py,          1); /* right  */
}

static int vertex_index(const ToricCode *code, int vx, int vy) {
    return wrap(vx, code->size_x) * code->size_y + wrap(vy, code->size_y);
}
static int plaquette_index(const ToricCode *code, int px, int py) {
    return wrap(px, code->size_x) * code->size_y + wrap(py, code->size_y);
}

static void free_legacy_2d(int **arr, int n) {
    if (!arr) return;
    for (int i = 0; i < n; i++) free(arr[i]);
    free(arr);
}

/* --------------------------- Lifecycle --------------------------------- */

ToricCode* initialize_toric_code(int size_x, int size_y) {
    if (size_x <= 0 || size_y <= 0) {
        fprintf(stderr, "Error: Toric code dimensions must be positive\n");
        return NULL;
    }

    ToricCode *code = calloc(1, sizeof(*code));
    if (!code) return NULL;
    code->size_x = size_x;
    code->size_y = size_y;
    code->num_links = 2 * size_x * size_y;
    int num_stabs = size_x * size_y;

    code->x_errors          = calloc((size_t)code->num_links, sizeof(int));
    code->z_errors          = calloc((size_t)code->num_links, sizeof(int));
    code->vertex_syndrome   = calloc((size_t)num_stabs,       sizeof(int));
    code->plaquette_syndrome= calloc((size_t)num_stabs,       sizeof(int));

    /* Legacy mirrors: 4 ints per stabilizer, each initialized to +1.
     * Maintained for back-compat with pre-v0.4 code paths that read
     * star_operators[i][j] / plaquette_operators[i][j].
     */
    code->star_operators      = calloc((size_t)num_stabs, sizeof(int*));
    code->plaquette_operators = calloc((size_t)num_stabs, sizeof(int*));
    for (int i = 0; i < num_stabs; i++) {
        code->star_operators[i]      = malloc(4 * sizeof(int));
        code->plaquette_operators[i] = malloc(4 * sizeof(int));
        for (int j = 0; j < 4; j++) {
            code->star_operators[i][j]      = 1;
            code->plaquette_operators[i][j] = 1;
        }
    }
    code->logical_operators_x = calloc(2, sizeof(int));
    code->logical_operators_z = calloc(2, sizeof(int));
    code->logical_operators_x[0] = code->logical_operators_x[1] = 1;
    code->logical_operators_z[0] = code->logical_operators_z[1] = 1;

    if (!code->x_errors || !code->z_errors ||
        !code->vertex_syndrome || !code->plaquette_syndrome ||
        !code->star_operators || !code->plaquette_operators ||
        !code->logical_operators_x || !code->logical_operators_z) {
        free_toric_code(code);
        return NULL;
    }
    return code;
}

void free_toric_code(ToricCode *code) {
    if (!code) return;
    free(code->x_errors);
    free(code->z_errors);
    free(code->vertex_syndrome);
    free(code->plaquette_syndrome);
    int num_stabs = code->size_x * code->size_y;
    free_legacy_2d(code->star_operators, num_stabs);
    free_legacy_2d(code->plaquette_operators, num_stabs);
    free(code->logical_operators_x);
    free(code->logical_operators_z);
    free(code);
}

/* --------------------------- Syndrome math ----------------------------- */

void toric_code_refresh_syndromes(ToricCode *code) {
    if (!code) return;
    int Lx = code->size_x, Ly = code->size_y;

    for (int vx = 0; vx < Lx; vx++) {
        for (int vy = 0; vy < Ly; vy++) {
            int links[4];
            toric_code_vertex_links(code, vx, vy, links);
            int parity = 0;
            for (int j = 0; j < 4; j++) parity ^= code->z_errors[links[j]];
            code->vertex_syndrome[vertex_index(code, vx, vy)] = parity;
        }
    }
    for (int px = 0; px < Lx; px++) {
        for (int py = 0; py < Ly; py++) {
            int links[4];
            toric_code_plaquette_links(code, px, py, links);
            int parity = 0;
            for (int j = 0; j < 4; j++) parity ^= code->x_errors[links[j]];
            code->plaquette_syndrome[plaquette_index(code, px, py)] = parity;
        }
    }

    /* Mirror syndromes to the legacy 4-per-stabilizer arrays as ±1 eigenvalues
     * at slot 0 (remaining slots held at +1 for back-compat readers). */
    int num_stabs = Lx * Ly;
    for (int i = 0; i < num_stabs; i++) {
        code->star_operators[i][0]      = code->vertex_syndrome[i]    ? -1 : 1;
        code->plaquette_operators[i][0] = code->plaquette_syndrome[i] ? -1 : 1;
    }
}

/* --------------------------- Error channel ----------------------------- */

void toric_code_apply_x_error(ToricCode *code, int link_index) {
    if (!code || link_index < 0 || link_index >= code->num_links) return;
    code->x_errors[link_index] ^= 1;
    toric_code_refresh_syndromes(code);
}
void toric_code_apply_z_error(ToricCode *code, int link_index) {
    if (!code || link_index < 0 || link_index >= code->num_links) return;
    code->z_errors[link_index] ^= 1;
    toric_code_refresh_syndromes(code);
}
void toric_code_apply_x_correction(ToricCode *code, int link_index) {
    toric_code_apply_x_error(code, link_index);
}
void toric_code_apply_z_correction(ToricCode *code, int link_index) {
    toric_code_apply_z_error(code, link_index);
}

void apply_random_errors(ToricCode *code, double error_rate) {
    if (!code || error_rate < 0.0 || error_rate > 1.0) return;
    for (int k = 0; k < code->num_links; k++) {
        if ((double)rand() / RAND_MAX < error_rate) code->x_errors[k] ^= 1;
        if ((double)rand() / RAND_MAX < error_rate) code->z_errors[k] ^= 1;
    }
    toric_code_refresh_syndromes(code);
}

/* --------------------------- Syndrome extraction ----------------------- */

static ErrorSyndrome* extract_syndrome(const int *syndrome_bits, int num, int type) {
    ErrorSyndrome *s = malloc(sizeof(*s));
    if (!s) return NULL;
    s->error_type = type;
    s->num_errors = 0;
    for (int i = 0; i < num; i++) if (syndrome_bits[i]) s->num_errors++;
    s->error_positions = s->num_errors ? malloc((size_t)s->num_errors * sizeof(int)) : NULL;
    if (s->num_errors && !s->error_positions) { free(s); return NULL; }
    int idx = 0;
    for (int i = 0; i < num; i++) if (syndrome_bits[i]) s->error_positions[idx++] = i;
    return s;
}

ErrorSyndrome* toric_code_measure_x_syndrome(ToricCode *code) {
    if (!code) return NULL;
    toric_code_refresh_syndromes(code);
    return extract_syndrome(code->plaquette_syndrome, code->size_x * code->size_y, 0);
}
ErrorSyndrome* toric_code_measure_z_syndrome(ToricCode *code) {
    if (!code) return NULL;
    toric_code_refresh_syndromes(code);
    return extract_syndrome(code->vertex_syndrome, code->size_x * code->size_y, 1);
}

/* Legacy API: combine both into a single syndrome struct; error_type
 * reflects which channel has more flagged stabilizers (ties → bit-flip). */
ErrorSyndrome* measure_error_syndrome(ToricCode *code) {
    if (!code) return NULL;
    toric_code_refresh_syndromes(code);
    int Ls = code->size_x * code->size_y;
    int n_p = 0, n_v = 0;
    for (int i = 0; i < Ls; i++) { n_p += code->plaquette_syndrome[i]; n_v += code->vertex_syndrome[i]; }
    if (n_v > n_p) return extract_syndrome(code->vertex_syndrome, Ls, 1);
    else           return extract_syndrome(code->plaquette_syndrome, Ls, 0);
}

void free_error_syndrome(ErrorSyndrome *syndrome) {
    if (!syndrome) return;
    free(syndrome->error_positions);
    free(syndrome);
}

/* ----------------------------- Decoding -------------------------------- */

/* Toroidal taxicab distance between two stabilizer indices (row-major L_x × L_y). */
static int toroidal_distance(const ToricCode *code, int a, int b, int *dx_out, int *dy_out) {
    int Lx = code->size_x, Ly = code->size_y;
    int ax = a / Ly, ay = a % Ly;
    int bx = b / Ly, by = b % Ly;
    int dx = abs(ax - bx); if (dx > Lx - dx) dx = Lx - dx;
    int dy = abs(ay - by); if (dy > Ly - dy) dy = Ly - dy;
    if (dx_out) *dx_out = dx;
    if (dy_out) *dy_out = dy;
    return dx + dy;
}

/* Apply corrections along the shortest toroidal path between two syndrome sites.
 *   is_z_correction == 1  -> walking on the PRIMAL lattice (vertex → vertex);
 *                            each step east/north crosses a horizontal/vertical
 *                            primal edge respectively, and we flip z_errors there.
 *   is_z_correction == 0  -> walking on the DUAL lattice (plaquette → plaquette);
 *                            each step east/north crosses a VERTICAL/HORIZONTAL
 *                            primal edge respectively (dual is rotated 90°),
 *                            and we flip x_errors there.
 */
static void apply_path_correction(ToricCode *code, int src, int dst, int is_z_correction) {
    int Lx = code->size_x, Ly = code->size_y;
    int sx = src / Ly, sy = src % Ly;
    int tx = dst / Ly, ty = dst % Ly;

    int cx = sx, cy = sy;
    while (cx != tx) {
        int step_forward  = wrap(tx - cx, Lx);
        int step_backward = wrap(cx - tx, Lx);
        int going_east    = (step_forward <= step_backward);
        int next_cx       = wrap(cx + (going_east ? 1 : -1), Lx);

        int link;
        if (is_z_correction) {
            /* Primal east step (vx,vy)->(vx+1,vy) uses horizontal link at (going_east?cx:next_cx, cy, 0). */
            int lx = going_east ? cx : next_cx;
            link = toric_code_link_index(code, lx, cy, 0);
            code->z_errors[link] ^= 1;
        } else {
            /* Dual east step plaq(px,py)->plaq(px+1,py) uses VERTICAL primal link
             * at ((px+1)%Lx, py, 1) == ((going_east?next_cx:cx), cy, 1). */
            int lx = going_east ? next_cx : cx;
            link = toric_code_link_index(code, lx, cy, 1);
            code->x_errors[link] ^= 1;
        }
        cx = next_cx;
    }
    while (cy != ty) {
        int step_forward  = wrap(ty - cy, Ly);
        int step_backward = wrap(cy - ty, Ly);
        int going_north   = (step_forward <= step_backward);
        int next_cy       = wrap(cy + (going_north ? 1 : -1), Ly);

        int link;
        if (is_z_correction) {
            /* Primal north step (vx,vy)->(vx,vy+1) uses vertical link at (cx, going_north?cy:next_cy, 1). */
            int ly = going_north ? cy : next_cy;
            link = toric_code_link_index(code, cx, ly, 1);
            code->z_errors[link] ^= 1;
        } else {
            /* Dual north step plaq(px,py)->plaq(px,py+1) uses HORIZONTAL primal link
             * at (px, (py+1)%Ly, 0) == (cx, going_north?next_cy:cy, 0). */
            int ly = going_north ? next_cy : cy;
            link = toric_code_link_index(code, cx, ly, 0);
            code->x_errors[link] ^= 1;
        }
        cy = next_cy;
    }
}

/* Greedy matching: repeatedly pair the two closest flagged stabilizers.
 * Not optimal (MWPM would be), but correct for low error rates and O(k^2)
 * where k = number of flagged sites. The learned decoder in v0.5 (P1.3) will
 * supersede this. */
static int greedy_pair_and_correct(ToricCode *code, int *syndrome_bits, int num, int is_z_correction) {
    int *flagged = malloc((size_t)num * sizeof(int));
    if (!flagged) return -1;
    int nf = 0;
    for (int i = 0; i < num; i++) if (syndrome_bits[i]) flagged[nf++] = i;

    while (nf >= 2) {
        int best_i = 0, best_j = 1, best_d = toroidal_distance(code, flagged[0], flagged[1], NULL, NULL);
        for (int i = 0; i < nf; i++) {
            for (int j = i + 1; j < nf; j++) {
                int d = toroidal_distance(code, flagged[i], flagged[j], NULL, NULL);
                if (d < best_d) { best_d = d; best_i = i; best_j = j; }
            }
        }
        apply_path_correction(code, flagged[best_i], flagged[best_j], is_z_correction);

        /* Remove best_i and best_j from the flagged list. */
        int new_nf = 0;
        for (int k = 0; k < nf; k++) if (k != best_i && k != best_j) flagged[new_nf++] = flagged[k];
        nf = new_nf;
    }
    free(flagged);
    return 0;
}

int toric_code_decode_greedy(ToricCode *code) {
    if (!code) return -1;
    toric_code_refresh_syndromes(code);
    int Ls = code->size_x * code->size_y;

    /* Vertex syndromes ← correct Z errors; plaquette syndromes ← correct X errors. */
    if (greedy_pair_and_correct(code, code->vertex_syndrome, Ls, 1) != 0) return -1;
    if (greedy_pair_and_correct(code, code->plaquette_syndrome, Ls, 0) != 0) return -1;

    toric_code_refresh_syndromes(code);
    return 0;
}

int perform_error_correction(ToricCode *code, ErrorSyndrome *syndrome) {
    (void)syndrome; /* syndrome is re-derived from the data-qubit state */
    return toric_code_decode_greedy(code);
}

/* ---------------------------- Queries ---------------------------------- */

/* Detect a logical error by computing the homology class of the error chain.
 *
 *   X-errors are a chain on the DUAL lattice (each X on a primal edge is a
 *   dual edge). Its class in H_1(dual, Z_2) = Z_2^2 is captured by parities
 *   of intersections with a basis of primal 1-cycles:
 *     winding_x_h = Σ x_errors[horizontal links at y=0] (mod 2)
 *     winding_x_v = Σ x_errors[vertical   links at x=0] (mod 2)
 *
 *   Z-errors are a chain on the PRIMAL lattice. Its H_1(primal, Z_2) class
 *   is captured by intersections with a basis of DUAL 1-cycles:
 *     winding_z_h = Σ z_errors[vertical   links at y=0] (mod 2)  (dual horizontal)
 *     winding_z_v = Σ z_errors[horizontal links at x=0] (mod 2)  (dual vertical)
 *
 * Any nonzero winding = a logical operator applied = logical error.
 */
int toric_code_has_logical_error(const ToricCode *code) {
    if (!code) return 0;
    int Lx = code->size_x, Ly = code->size_y;

    int winding_x_h = 0, winding_x_v = 0, winding_z_h = 0, winding_z_v = 0;
    for (int x = 0; x < Lx; x++) winding_x_h ^= code->x_errors[toric_code_link_index(code, x, 0, 0)];
    for (int y = 0; y < Ly; y++) winding_x_v ^= code->x_errors[toric_code_link_index(code, 0, y, 1)];
    for (int x = 0; x < Lx; x++) winding_z_h ^= code->z_errors[toric_code_link_index(code, x, 0, 1)];
    for (int y = 0; y < Ly; y++) winding_z_v ^= code->z_errors[toric_code_link_index(code, 0, y, 0)];
    return (winding_x_h | winding_x_v | winding_z_h | winding_z_v);
}

int is_ground_state(ToricCode *code) {
    if (!code) return 0;
    toric_code_refresh_syndromes(code);
    int Ls = code->size_x * code->size_y;
    for (int i = 0; i < Ls; i++) {
        if (code->vertex_syndrome[i])    return 0;
        if (code->plaquette_syndrome[i]) return 0;
    }
    return toric_code_has_logical_error(code) ? 0 : 1;
}

int calculate_ground_state_degeneracy(ToricCode *code) {
    (void)code;
    return 4; /* 2 logical qubits on a torus */
}

/* ---------------- Kitaev-lattice coupling (legacy paths) --------------- */

void calculate_stabilizers(ToricCode *code, KitaevLattice *lattice) {
    if (!code || !lattice) return;
    if (lattice->size_x < code->size_x || lattice->size_y < code->size_y) {
        fprintf(stderr, "Error: Lattice size is smaller than toric code size\n");
        return;
    }
    /* Seed x_errors from the Kitaev lattice's spin sign pattern. Spin -1 at
     * a site marks an accumulated X error on that site's horizontal link.
     * This keeps pre-v0.4 demos visually equivalent while using the new
     * physical model under the hood. */
    for (int x = 0; x < code->size_x; x++) {
        for (int y = 0; y < code->size_y; y++) {
            int k = toric_code_link_index(code, x, y, 0);
            code->x_errors[k] = (lattice->spins[x][y][0] < 0) ? 1 : 0;
        }
    }
    toric_code_refresh_syndromes(code);
}

void map_toric_code_to_lattice(ToricCode *code, KitaevLattice *lattice) {
    if (!code || !lattice) return;
    if (lattice->size_x < code->size_x || lattice->size_y < code->size_y) return;
    /* Write the current x_errors pattern back to spin values on horizontal links. */
    for (int x = 0; x < code->size_x; x++) {
        for (int y = 0; y < code->size_y; y++) {
            int k = toric_code_link_index(code, x, y, 0);
            lattice->spins[x][y][0] = code->x_errors[k] ? -1 : 1;
        }
    }
}
