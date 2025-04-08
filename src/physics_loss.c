#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include "neural_network.h"
#include "physics_loss.h"
#include "energy_utils.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

double compute_physics_loss(double ising_energy, double kitaev_energy, double spin_energy,
                            IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                            double dt, double dx, const char* loss_type) {
    if (strcmp(loss_type, "maxwell") == 0) {
        return maxwell_loss(ising_energy, kitaev_energy, spin_energy, ising_lattice, kitaev_lattice, spin_lattice, dt, dx);
    } else if (strcmp(loss_type, "heat") == 0) {
        return heat_loss(ising_energy, kitaev_energy, spin_energy, ising_lattice, kitaev_lattice, spin_lattice, dt, dx);
    } else if (strcmp(loss_type, "schrodinger") == 0) {
        return schrodinger_loss(ising_energy, kitaev_energy, spin_energy, ising_lattice, kitaev_lattice, spin_lattice, dt, dx);
    } else if (strcmp(loss_type, "navier_stokes") == 0) {
        return navier_stokes_loss(ising_energy, kitaev_energy, spin_energy, ising_lattice, kitaev_lattice, spin_lattice, dt, dx);
    } else if (strcmp(loss_type, "wave") == 0) {
        return wave_loss(ising_energy, kitaev_energy, spin_energy, ising_lattice, kitaev_lattice, spin_lattice, dt, dx);
    } else {
        // Default to Heat loss if an unknown type is specified
        return heat_loss(ising_energy, kitaev_energy, spin_energy, ising_lattice, kitaev_lattice, spin_lattice, dt, dx);
    }
}

// Calculus functions
double laplacian_3d(int*** lattice, int x, int y, int z, int size_x, int size_y, int size_z, double dx) {
    double laplacian = 0.0;
    laplacian += (lattice[(x+1)%size_x][y][z] + lattice[(x-1+size_x)%size_x][y][z] - 2*lattice[x][y][z]) / (dx*dx);
    laplacian += (lattice[x][(y+1)%size_y][z] + lattice[x][(y-1+size_y)%size_y][z] - 2*lattice[x][y][z]) / (dx*dx);
    laplacian += (lattice[x][y][(z+1)%size_z] + lattice[x][y][(z-1+size_z)%size_z] - 2*lattice[x][y][z]) / (dx*dx);
    return laplacian;
}

double discrete_curl_ising(IsingLattice* lattice, int x, int y, int z) {
    double curl = 0.0;
    int size_x = lattice->size_x;
    int size_y = lattice->size_y;
    int size_z = lattice->size_z;
    
    // x-component
    curl += lattice->spins[(x+1) % size_x][y][z] - lattice->spins[(x-1+size_x) % size_x][y][z];
    // y-component
    curl += lattice->spins[x][(y+1) % size_y][z] - lattice->spins[x][(y-1+size_y) % size_y][z];
    // z-component
    curl += lattice->spins[x][y][(z+1) % size_z] - lattice->spins[x][y][(z-1+size_z) % size_z];
    
    return curl;
}

double discrete_curl_kitaev(KitaevLattice* lattice, int x, int y, int z) {
    double curl = 0.0;
    int size_x = lattice->size_x;
    int size_y = lattice->size_y;
    int size_z = lattice->size_z;
    
    // x-component
    curl += lattice->spins[(x+1) % size_x][y][z] - lattice->spins[(x-1+size_x) % size_x][y][z];
    // y-component
    curl += lattice->spins[x][(y+1) % size_y][z] - lattice->spins[x][(y-1+size_y) % size_y][z];
    // z-component
    curl += lattice->spins[x][y][(z+1) % size_z] - lattice->spins[x][y][(z-1+size_z) % size_z];
    
    return curl;
}

double divergence(int*** u_x, int*** u_y, Spin*** u_z, int x, int y, int z, int size_x, int size_y, int size_z, double dx) {
    double div = 0.0;
    div += (u_x[(x+1)%size_x][y][z] - u_x[(x-1+size_x)%size_x][y][z]) / (2*dx);
    div += (u_y[x][(y+1)%size_y][z] - u_y[x][(y-1+size_y)%size_y][z]) / (2*dx);
    div += (u_z[x][y][(z+1)%size_z].sx - u_z[x][y][(z-1+size_z)%size_z].sx) / (2*dx);
    return div;
}

double gradient_x(int*** field, int x, int y, int z, int size_x, double dx) {
    return (field[(x+1)%size_x][y][z] - field[(x-1+size_x)%size_x][y][z]) / (2*dx);
}

double gradient_y(int*** field, int x, int y, int z, int size_y, double dx) {
    return (field[x][(y+1)%size_y][z] - field[x][(y-1+size_y)%size_y][z]) / (2*dx);
}

double gradient_z(int*** field, int x, int y, int z, int size_z, double dx) {
    return (field[x][y][(z+1)%size_z] - field[x][y][(z-1+size_z)%size_z]) / (2*dx);
}

// Spin calculus functions
double laplacian_3d_spin(Spin*** lattice, int x, int y, int z, int size_x, int size_y, int size_z, double dx) {
    double lap = 0.0;
    lap += (lattice[(x+1)%size_x][y][z].sx + lattice[(x-1+size_x)%size_x][y][z].sx - 2*lattice[x][y][z].sx) / (dx*dx);
    lap += (lattice[x][(y+1)%size_y][z].sy + lattice[x][(y-1+size_y)%size_y][z].sy - 2*lattice[x][y][z].sy) / (dx*dx);
    lap += (lattice[x][y][(z+1)%size_z].sz + lattice[x][y][(z-1+size_z)%size_z].sz - 2*lattice[x][y][z].sz) / (dx*dx);
    return lap;
}

double divergence_spin(Spin*** u_x, Spin*** u_y, Spin*** u_z, int x, int y, int z, int size_x, int size_y, int size_z, double dx) {
    double div = 0.0;
    div += (u_x[(x+1)%size_x][y][z].sx - u_x[(x-1+size_x)%size_x][y][z].sx) / (2*dx);
    div += (u_y[x][(y+1)%size_y][z].sy - u_y[x][(y-1+size_y)%size_y][z].sy) / (2*dx);
    div += (u_z[x][y][(z+1)%size_z].sz - u_z[x][y][(z-1+size_z)%size_z].sz) / (2*dx);
    return div;
}

double gradient_x_spin(Spin*** field, int x, int y, int z, int size_x, double dx) {
    return (field[(x+1)%size_x][y][z].sx - field[(x-1+size_x)%size_x][y][z].sx) / (2*dx);
}

double gradient_y_spin(Spin*** field, int x, int y, int z, int size_y, double dx) {
    return (field[x][(y+1)%size_y][z].sy - field[x][(y-1+size_y)%size_y][z].sy) / (2*dx);
}

double gradient_z_spin(Spin*** field, int x, int y, int z, int size_z, double dx) {
    return (field[x][y][(z+1)%size_z].sz - field[x][y][(z-1+size_z)%size_z].sz) / (2*dx);
}

double schrodinger_loss(double ising_energy, double kitaev_energy, double spin_energy, 
                        IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                        double dt, double dx) {
    double scale_factor = 1e-55;
    int size_x = ising_lattice->size_x;
    int size_y = ising_lattice->size_y;
    int size_z = ising_lattice->size_z;
    
    double total_loss = 0.0;
    
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                // Use Ising lattice for real part and Kitaev lattice for imaginary part of wavefunction
                double psi_real = ising_lattice->spins[x][y][z];
                double psi_imag = kitaev_lattice->spins[x][y][z];
                
                // Use spin lattice for potential
                double V = sqrt(spin_lattice->spins[x][y][z].sx * spin_lattice->spins[x][y][z].sx +
                                spin_lattice->spins[x][y][z].sy * spin_lattice->spins[x][y][z].sy +
                                spin_lattice->spins[x][y][z].sz * spin_lattice->spins[x][y][z].sz);

                // Compute Laplacian for real and imaginary parts
                double d2psi_dx2_real = laplacian_3d(ising_lattice->spins, x, y, z, size_x, size_y, size_z, dx);
                double d2psi_dx2_imag = laplacian_3d(kitaev_lattice->spins, x, y, z, size_x, size_y, size_z, dx);

                // Compute time derivatives (this is a simplification; you might want to store previous state for better accuracy)
                double dpsi_dt_real = (psi_imag - psi_real) / dt;
                double dpsi_dt_imag = -(psi_real - psi_imag) / dt;

                // Compute Schrödinger equation residuals
                double residual_real = dpsi_dt_real + (HBAR / (2.0 * M)) * d2psi_dx2_imag + (V / HBAR) * psi_real;
                double residual_imag = dpsi_dt_imag - (HBAR / (2.0 * M)) * d2psi_dx2_real + (V / HBAR) * psi_imag;

                // Add to total loss
                total_loss += (residual_real * residual_real + residual_imag * residual_imag);
            }
        }
    }
    
    // Scale and normalize the loss
    double scaled_loss = total_loss * scale_factor / (size_x * size_y * size_z);
    return log(1 + scaled_loss);
}

double maxwell_loss(double ising_energy, double kitaev_energy, double spin_energy,
                    IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                    double dt, double dx) {
    int size_x = ising_lattice->size_x;
    int size_y = ising_lattice->size_y;
    int size_z = ising_lattice->size_z;
    
    double volume = size_x * size_y * size_z * dx * dx * dx;
    double total_energy = ising_energy + kitaev_energy + spin_energy;
    
    // Estimate field strengths based on total energy
    double field_strength = sqrt(2 * fabs(total_energy) / (EPSILON0 * volume));
    double B_strength = field_strength / 299792458.0; // c in m/s
    
    double loss = 0.0;
    
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                // Estimate local E and B field based on spin configuration
                double E_local = field_strength * ising_lattice->spins[x][y][z];
                double B_local = B_strength * kitaev_lattice->spins[x][y][z];
                
                // Calculate discrete curl-like quantities
                double curl_E = discrete_curl_ising(ising_lattice, x, y, z);
                double curl_B = discrete_curl_kitaev(kitaev_lattice, x, y, z);
                
                // Estimate time derivatives (this is a rough approximation)
                double dEdt = (E_local - field_strength * ising_lattice->spins[x][y][z]) / dt;
                double dBdt = (B_local - B_strength * kitaev_lattice->spins[x][y][z]) / dt;
                
                // Estimate current density from spin lattice
                double J_magnitude = sqrt(spin_lattice->spins[x][y][z].sx * spin_lattice->spins[x][y][z].sx +
                                          spin_lattice->spins[x][y][z].sy * spin_lattice->spins[x][y][z].sy +
                                          spin_lattice->spins[x][y][z].sz * spin_lattice->spins[x][y][z].sz) / dx;
                
                // Calculate Faraday and Ampère residuals
                double faraday_residual = curl_E + dBdt;
                double ampere_residual = curl_B - MU0 * (J_magnitude + EPSILON0 * dEdt);
                
                // Add to total loss
                loss += faraday_residual * faraday_residual + ampere_residual * ampere_residual;
            }
        }
    }
    
    double scale_factor = 1e-40; // This may need adjustment
    return loss * scale_factor;
}

double navier_stokes_loss(double ising_energy, double kitaev_energy, double spin_energy,
                          IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                          double dt, double dx) {
    // All variable declarations at the beginning of the function
    double scale_factor = 1e-40;
    int size_x = ising_lattice->size_x;
    int size_y = ising_lattice->size_y;
    int size_z = ising_lattice->size_z;
    double RHO_BASE = 1.0;  // Base fluid density
    double NU_BASE = 1e-6;  // Base kinematic viscosity
    double G = 9.81;        // Gravitational acceleration
    double total_energy, RHO, NU, total_loss;
    int x, y, z;
    double u, v, w, p, e;
    double div_u, dudt, dvdt, dwdt;
    double dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
    double dpdx, dpdy, dpdz;
    double lap_u, lap_v, lap_w;
    double continuity_residual, momentum_x_residual, momentum_y_residual, momentum_z_residual, energy_residual;

    // Calculations start here
    total_energy = fabs(ising_energy) + fabs(kitaev_energy) + fabs(spin_energy);
    RHO = RHO_BASE * (1.0 + 0.1 * tanh(total_energy));  // Density increases with total energy
    NU = NU_BASE * exp(-total_energy / 1e6);            // Viscosity decreases with total energy
    
    total_loss = 0.0;
    
    for (x = 0; x < size_x; x++) {
        for (y = 0; y < size_y; y++) {
            for (z = 0; z < size_z; z++) {
                // Velocity components
                u = ising_lattice->spins[x][y][z];
                v = kitaev_lattice->spins[x][y][z];
                w = spin_lattice->spins[x][y][z].sx;
                
                // Pressure
                p = spin_lattice->spins[x][y][z].sy;
                
                // Energy density (simplified representation)
                e = (ising_energy * u*u + kitaev_energy * v*v + spin_energy * w*w) / (size_x * size_y * size_z);
                
                // Compute derivatives
                div_u = divergence(ising_lattice->spins, kitaev_lattice->spins, spin_lattice->spins, x, y, z, size_x, size_y, size_z, dx);
                
                dudt = (u - ising_lattice->spins[x][y][z]) / dt;
                dvdt = (v - kitaev_lattice->spins[x][y][z]) / dt;
                dwdt = (w - spin_lattice->spins[x][y][z].sx) / dt;

                dudx = gradient_x(ising_lattice->spins, x, y, z, size_x, dx);
                dudy = gradient_y(ising_lattice->spins, x, y, z, size_y, dx);
                dudz = gradient_z(ising_lattice->spins, x, y, z, size_z, dx);

                dvdx = gradient_x(kitaev_lattice->spins, x, y, z, size_x, dx);
                dvdy = gradient_y(kitaev_lattice->spins, x, y, z, size_y, dx);
                dvdz = gradient_z(kitaev_lattice->spins, x, y, z, size_z, dx);

                dwdx = gradient_x_spin(spin_lattice->spins, x, y, z, size_x, dx);
                dwdy = gradient_y_spin(spin_lattice->spins, x, y, z, size_y, dx);
                dwdz = gradient_z_spin(spin_lattice->spins, x, y, z, size_z, dx);

                dpdx = gradient_x_spin(spin_lattice->spins, x, y, z, size_x, dx);
                dpdy = gradient_y_spin(spin_lattice->spins, x, y, z, size_y, dx);
                dpdz = gradient_z_spin(spin_lattice->spins, x, y, z, size_z, dx);

                lap_u = laplacian_3d(ising_lattice->spins, x, y, z, size_x, size_y, size_z, dx);
                lap_v = laplacian_3d(kitaev_lattice->spins, x, y, z, size_x, size_y, size_z, dx);
                lap_w = laplacian_3d_spin(spin_lattice->spins, x, y, z, size_x, size_y, size_z, dx);
                
                // Compute Navier-Stokes equation residuals
                continuity_residual = div_u;
                // Include pressure effects in momentum equations
                momentum_x_residual = dudt + u*dudx + v*dudy + w*dudz + (1.0/RHO)*dpdx - NU*lap_u + e*dudx/RHO + p/(RHO*size_x);
                momentum_y_residual = dvdt + u*dvdx + v*dvdy + w*dvdz + (1.0/RHO)*dpdy - NU*lap_v + e*dvdy/RHO + p/(RHO*size_y);
                momentum_z_residual = dwdt + u*dwdx + v*dwdy + w*dwdz + (1.0/RHO)*dpdz - NU*lap_w - G + e*dwdz/RHO + p/(RHO*size_z);
                
                // Energy equation residual (simplified)
                energy_residual = (e - (ising_energy + kitaev_energy + spin_energy) / (size_x * size_y * size_z)) / dt 
                                  + u*dudx + v*dvdy + w*dwdz;
                
                // Add to total loss
                total_loss += continuity_residual*continuity_residual +
                              momentum_x_residual*momentum_x_residual +
                              momentum_y_residual*momentum_y_residual +
                              momentum_z_residual*momentum_z_residual +
                              energy_residual*energy_residual;
            }
        }
    }
    
    // Scale and normalize the loss
    return total_loss * scale_factor / (size_x * size_y * size_z);
}

double heat_loss(double ising_energy, double kitaev_energy, double spin_energy,
                 IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                 double dt, double dx) {
    double scale_factor = 1e-40;
    int size_x = ising_lattice->size_x;
    int size_y = ising_lattice->size_y;
    int size_z = ising_lattice->size_z;
    
    double total_loss = 0.0;
    
    // Ising lattice as temperature field
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                // Current temperature
                double T = ising_lattice->spins[x][y][z];
                
                // Compute Laplacian of temperature
                double d2Tdx2 = laplacian_3d(ising_lattice->spins, x, y, z, size_x, size_y, size_z, dx);
                
                // Compute time derivative (this is a simplification; you might want to store previous state for better accuracy)
                double dTdt = (kitaev_lattice->spins[x][y][z] - T) / dt;
                
                // Compute heat equation residual
                double residual = dTdt - ALPHA * d2Tdx2;
                
                // Add to total loss
                total_loss += residual * residual;
            }
        }
    }
    
    // Scale and normalize the loss
    double scaled_loss = total_loss * scale_factor / (size_x * size_y * size_z);
    return scaled_loss;
}

double wave_loss(double ising_energy, double kitaev_energy, double spin_energy,
                 IsingLattice* ising_lattice, KitaevLattice* kitaev_lattice, SpinLattice* spin_lattice,
                 double dt, double dx) {
    double scale_factor = 1e-40;
    int size_x = ising_lattice->size_x;
    int size_y = ising_lattice->size_y;
    int size_z = ising_lattice->size_z;
    
    double total_loss = 0.0;
    
    // Ising lattice as wave field
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                // Current wave amplitude
                double u = ising_lattice->spins[x][y][z];
                
                // Compute Laplacian of wave field
                double d2udx2 = laplacian_3d(ising_lattice->spins, x, y, z, size_x, size_y, size_z, dx);
                
                // Compute second-order time derivative
                // (this is a simplification; ideally, you'd want to store two previous time steps)
                double u_prev = kitaev_lattice->spins[x][y][z];
                double u_next = spin_lattice->spins[x][y][z].sx; // Using sx component as an example
                double d2udt2 = (u_next - 2.0 * u + u_prev) / (dt * dt);
                
                // Compute wave equation residual
                double residual = d2udt2 - C * C * d2udx2;
                
                // Add to total loss
                total_loss += residual * residual;
            }
        }
    }
    
    // Scale and normalize the loss
    double scaled_loss = total_loss * scale_factor / (size_x * size_y * size_z);
    return scaled_loss;
}
