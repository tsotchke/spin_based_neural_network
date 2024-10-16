#ifndef PHYSICS_LOSS_H
#define PHYSICS_LOSS_H

#define ALPHA 1.0e-7                  // m^2/s (thermal diffusivity)
#define C 299792458.0                 // m/s (speed of light in vacuum)
#define HBAR 1.0545718e-34            // J·s (reduced Planck constant)
#define M 9.10938356e-31              // kg (mass of an electron)
#define NU 1.0e-6                     // m^2/s (kinematic viscosity)
#define EPSILON0 8.854187817e-12      // F/m (vacuum permittivity)
#define MU0 1.25663706212e-6          // N/A^2 or H/m (vacuum permeability)
#define RHO 1000.0                    // kg/m^3 (density of water at STP)
#define G 9.81                        // m/s^2 (acceleration due to gravity at Earth's surface)

double compute_physics_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx, const char* loss_type);
double schrodinger_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx);
double maxwell_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx);
double navier_stokes_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx);
double heat_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx);
double wave_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx);

#endif // PHYSICS_LOSS_H