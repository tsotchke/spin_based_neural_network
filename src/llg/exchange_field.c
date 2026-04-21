/*
 * src/llg/exchange_field.c
 *
 * Heisenberg nearest-neighbour exchange with periodic boundary
 * conditions. No optimisation tricks — this is the reference
 * implementation the equivariant-GNN torque net will be benchmarked
 * against.
 */
#include "llg/exchange_field.h"

/* 2D LLG effective field. Layout: site (x, y) is at index x*Ly + y;
 * b_eff and m have 3 components per site.
 *
 * Contributions:
 *   Exchange:     B_ex,i = J · Σ_{j ∈ neighbours(i)} m_j
 *   Anisotropy:   B_an,i_z += 2 Kz m_i_z
 *   Zeeman:       + (Bx, By, Bz)
 *   Interfacial DMI (Bloch-skyrmion-stabilising):
 *      For a bond from site i → i+x̂: E = D · ẑ · (m_i × m_{i+x̂} × x̂_perp)
 *      which with x̂_perp = ẑ × x̂ = ŷ gives E = D (m_i × m_{i+x̂}) · ŷ
 *      similarly for ŷ-bonds with D (m_i × m_{i+ŷ}) · (-x̂).
 *
 *      B_DMI,i from ∂/∂m_i of all adjacent-bond DMI terms: on each
 *      bond (i, j), B gets a cross-product contribution of
 *         -D (r̂_ij × ẑ) × m_j      (for interfacial DMI)
 *      summed over all four neighbours with appropriate signs for
 *      the bond orientation. */
void llg_2d_field(const double *m, double *b_eff,
                   long num_sites, void *user_data) {
    llg_2d_config_t *p = (llg_2d_config_t *)user_data;
    int Lx = p->Lx, Ly = p->Ly;
    double J = p->J, Kz = p->Kz, D = p->D_dmi;
    (void)num_sites;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++) {
        long i = x * Ly + y;
        double bx = 0, by = 0, bz = 0;
        /* 4-neighbour exchange (periodic BCs). */
        int nx_xm = ((x + Lx - 1) % Lx) * Ly + y;
        int nx_xp = ((x + 1)      % Lx) * Ly + y;
        int nx_ym = x * Ly + ((y + Ly - 1) % Ly);
        int nx_yp = x * Ly + ((y + 1)      % Ly);
        bx += J * (m[3*nx_xm    ] + m[3*nx_xp    ] + m[3*nx_ym    ] + m[3*nx_yp    ]);
        by += J * (m[3*nx_xm + 1] + m[3*nx_xp + 1] + m[3*nx_ym + 1] + m[3*nx_yp + 1]);
        bz += J * (m[3*nx_xm + 2] + m[3*nx_xp + 2] + m[3*nx_ym + 2] + m[3*nx_yp + 2]);
        /* Uniaxial anisotropy. */
        bz += 2.0 * Kz * m[3*i + 2];
        /* Uniform Zeeman. */
        bx += p->Bx; by += p->By; bz += p->Bz;
        /* Interfacial DMI. For a bond (i → i + x̂), the DMI
         * vector is D_ij = D · ŷ (rotating m in the xz plane). The
         * B-field contribution to site i from this bond:
         *   B_DMI,i += -D_ij × m_{i + x̂}
         * so bx_i += -Dy · m^z_{i+x̂} + Dz · m^y_{i+x̂}
         *    by_i += -Dz · m^x_{i+x̂} + Dx · m^z_{i+x̂}
         *    bz_i += -Dx · m^y_{i+x̂} + Dy · m^x_{i+x̂}
         * For x-bonds, D_ij = (0, D, 0) going in the +x direction;
         * flip sign for the bond going in the -x direction. */
        if (D != 0.0) {
            /* +x neighbour: D_vec = (0, D, 0) */
            bx += -D * m[3*nx_xp + 2];  /* (-Dy · m^z) */
            bz +=  D * m[3*nx_xp    ];  /* (+Dy · m^x) */
            /* -x neighbour: D_vec = (0, -D, 0) */
            bx +=  D * m[3*nx_xm + 2];
            bz += -D * m[3*nx_xm    ];
            /* +y neighbour: D_vec = (-D, 0, 0) */
            by +=  D * m[3*nx_yp + 2];  /* (-Dx · m^z) ... Dx = -D */
            bz += -D * m[3*nx_yp + 1];
            /* -y neighbour: D_vec = (+D, 0, 0) */
            by += -D * m[3*nx_ym + 2];
            bz +=  D * m[3*nx_ym + 1];
        }
        b_eff[3*i    ] = bx;
        b_eff[3*i + 1] = by;
        b_eff[3*i + 2] = bz;
    }
}

void llg_heisenberg_1d_field(const double *m, double *b_eff,
                              long num_sites, void *user_data) {
    llg_heisenberg_1d_t *p = (llg_heisenberg_1d_t *)user_data;
    double J  = p->J;
    double Kz = p->Kz;
    double D  = p->D_dmi;
    for (long i = 0; i < num_sites; i++) {
        long im = (i + num_sites - 1) % num_sites;
        long ip = (i + 1)             % num_sites;
        double bx = J * (m[3*im    ] + m[3*ip    ]) + p->Bx;
        double by = J * (m[3*im + 1] + m[3*ip + 1]) + p->By;
        double bz = J * (m[3*im + 2] + m[3*ip + 2]) + p->Bz;
        /* Uniaxial anisotropy along z. */
        bz += 2.0 * Kz * m[3*i + 2];
        /* 1D DMI along z:
         *   B_eff,i = -∂E_DMI/∂m_i
         *     E = D Σ_j ẑ·(m_j × m_{j+1})
         *     ∂/∂m_i_x = D (m_{i+1}_y - m_{i-1}_y)
         *     ∂/∂m_i_y = -D (m_{i+1}_x - m_{i-1}_x)
         *     ∂/∂m_i_z = 0
         * so B_DMI,i = (-D (m_{i+1}_y - m_{i-1}_y), +D (m_{i+1}_x - m_{i-1}_x), 0). */
        if (D != 0.0) {
            bx += -D * (m[3*ip + 1] - m[3*im + 1]);
            by +=  D * (m[3*ip    ] - m[3*im    ]);
        }
        b_eff[3*i  ] = bx;
        b_eff[3*i+1] = by;
        b_eff[3*i+2] = bz;
    }
}
