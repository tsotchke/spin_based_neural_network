#ifndef PHYSICS_LOSS_H
#define PHYSICS_LOSS_H
#include "ising_model.h"
#include "kitaev_model.h"
#include "spin_models.h"

#define ALPHA 1.0e-7                  // m^2/s (thermal diffusivity)
#define C 299792458.0                 // m/s (speed of light in vacuum)
#define HBAR 1.0545718e-34            // J·s (reduced Planck constant)
#define M 9.10938356e-31              // kg (mass of an electron)
//#define NU 1.0e-6                     // m^2/s (kinematic viscosity)
#define EPSILON0 8.854187817e-12      // F/m (vacuum permittivity)
#define MU0 1.25663706212e-6          // N/A^2 or H/m (vacuum permeability)
//#define RHO 1000.0                    // kg/m^3 (density of water at STP)
//#define G 9.81                        // m/s^2 (acceleration due to gravity at Earth's surface)

double compute_physics_loss(double ising_energy, double kitaev_energy, double spin_energy,
                            IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                            double dt, double dx, const char* loss_type);
double schrodinger_loss(double ising_energy, double kitaev_energy, double spin_energy, 
                        IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                        double dt, double dx);
double maxwell_loss(double ising_energy, double kitaev_energy, double spin_energy,
                    IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                    double dt, double dx);
double navier_stokes_loss(double ising_energy, double kitaev_energy, double spin_energy,
                          IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                          double dt, double dx);
double heat_loss(double ising_energy, double kitaev_energy, double spin_energy,
                 IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                 double dt, double dx);
double wave_loss(double ising_energy, double kitaev_energy, double spin_energy,
                 IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                 double dt, double dx);

/*
 * Variational micromagnetic loss (P2.7).  Returns the discretised
 * Landau–Lifshitz free-energy functional
 *
 *   E[m] = J_ex Σ_<ij> ||m_i − m_j||²
 *        + K_u  Σ_i (1 − (m_i · ẑ)²)
 *        − μ₀ Σ_i B · m_i
 *
 * (exchange + uniaxial anisotropy + Zeeman) on the SpinLattice with the
 * supplied parameters.  This is the energy-minimisation form preferred
 * over residual minimisation for stability — the loss vanishes at the
 * micromagnetic ground state of the chosen parameters.  Demag and DMI
 * terms remain to be added in v0.5 (P1.2).
 */
double micromagnetic_loss(SpinLattice* spin_lattice,
                          double J_ex, double K_u,
                          double Bx, double By, double Bz);

/*
 * Hard-constraint projection: rescale every m_i to unit norm in-place.
 * No-op for zero vectors (left at zero).  Use after each integrator step
 * or training update to enforce |m|=1 to machine precision instead of
 * accumulating drift.  Returns the maximum |m_i| − 1 deviation observed
 * before projection (a useful drift diagnostic).
 */
double project_spin_lattice_to_unit_sphere(SpinLattice* spin_lattice);

/*
 * Fourier-feature embedding for PINN coordinate inputs (P2.7).
 *
 * Maps n_in coordinates x ∈ [0,1)^{n_in} to a 2·n_in·n_freqs vector of
 * (sin, cos) pairs at logarithmically-spaced frequencies 2π·2^k for
 * k = 0..n_freqs-1.  This is the standard NeRF-style embedding that lets
 * a small MLP fit high-frequency content; pair with SIREN activations
 * (neural_network.c) for the full P2.7 PINN stack.
 *
 * out[] must have at least 2 · n_in · n_freqs slots.
 */
void fourier_features(const double* coords, int n_in,
                      int n_freqs, double* out);

// Function prototypes for int*** lattices (Ising and Kitaev)
double divergence(int*** u_x, int*** u_y, Spin*** u_z, int x, int y, int z, int size_x, int size_y, int size_z, double dx);
double laplacian_3d(int*** lattice, int x, int y, int z, int size_x, int size_y, int size_z, double dx);
double gradient_x(int*** field, int x, int y, int z, int size_x, double dx);
double gradient_y(int*** field, int x, int y, int z, int size_y, double dx);
double gradient_z(int*** field, int x, int y, int z, int size_z, double dx);

// Function prototypes for Spin*** lattice
double divergence_spin(Spin*** u_x, Spin*** u_y, Spin*** u_z, int x, int y, int z, int size_x, int size_y, int size_z, double dx);
double laplacian_3d_spin(Spin*** lattice, int x, int y, int z, int size_x, int size_y, int size_z, double dx);
double gradient_x_spin(Spin*** field, int x, int y, int z, int size_x, double dx);
double gradient_y_spin(Spin*** field, int x, int y, int z, int size_y, double dx);
double gradient_z_spin(Spin*** field, int x, int y, int z, int size_z, double dx);
#endif // PHYSICS_LOSS_H