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