#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include "neural_network.h"
#include "physics_loss.h"
#include "energy_utils.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

double compute_physics_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx, const char* loss_type) {
    double loss = 0.0;

    if (strcmp(loss_type, "heat") == 0) {
        loss = heat_loss(ising_energy, kitaev_energy, spin_energy, dt, dx);
    } else if (strcmp(loss_type, "wave") == 0) {
        loss = wave_loss(ising_energy, kitaev_energy, spin_energy, dt, dx);
    } else if (strcmp(loss_type, "schrodinger") == 0) {
        loss = schrodinger_loss(ising_energy, kitaev_energy, spin_energy, dt, dx);
    } else if (strcmp(loss_type, "navier-stokes") == 0) {
        loss = navier_stokes_loss(ising_energy, kitaev_energy, spin_energy, dt, dx);
    } else if (strcmp(loss_type, "maxwell") == 0) {
        loss = maxwell_loss(ising_energy, kitaev_energy, spin_energy, dt, dx);
    } else {
        fprintf(stderr, "Unknown loss type: %s\n", loss_type);
        loss = 0.0;
    }

    return loss;
}

double schrodinger_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx) {
    double scale_factor = 1e-60;
    
    double scaled_ising = scale_energy(ising_energy);
    double scaled_kitaev = scale_energy(kitaev_energy);
    double scaled_spin = scale_energy(spin_energy);
    
    double psi_real = scaled_ising;
    double psi_imag = scaled_kitaev;
    double V = scaled_spin;

    double d2psi_dx2_real = (psi_real - 2.0 * psi_real + psi_real) / (dx * dx);
    double d2psi_dx2_imag = (psi_imag - 2.0 * psi_imag + psi_imag) / (dx * dx);

    double dpsi_dt_real = (psi_imag - psi_real) / dt;
    double dpsi_dt_imag = -(psi_real - psi_imag) / dt;

    double residual_real = dpsi_dt_real + (HBAR / (2.0 * M)) * d2psi_dx2_imag + (V / HBAR) * psi_real;
    double residual_imag = dpsi_dt_imag - (HBAR / (2.0 * M)) * d2psi_dx2_real + (V / HBAR) * psi_imag;

    double loss = (residual_real * residual_real + residual_imag * residual_imag) * scale_factor;

    return log(1 + loss);
}

double maxwell_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx) {
    double scale_factor = 1e-40;
    
    double scaled_ising = scale_energy(ising_energy);
    double scaled_kitaev = scale_energy(kitaev_energy);
    double scaled_spin = scale_energy(spin_energy);
    
    double E = scaled_ising;
    double B = scaled_kitaev;
    double J = scaled_spin;

    double dEdt = (E - E) / dt;
    double dBdx = (B - B) / dx;
    double dBdt = (B - B) / dt;
    double dEdx = (E - E) / dx;

    double faraday_loss = pow(dEdt + dBdx, 2);
    double ampere_loss = pow(dBdt - dEdx - MU0 * J, 2);

    return (faraday_loss + ampere_loss) * scale_factor;
}

double navier_stokes_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx) {
    double scale_factor = 1e-40;
    
    double scaled_ising = scale_energy(ising_energy);
    double scaled_kitaev = scale_energy(kitaev_energy);
    double scaled_spin = scale_energy(spin_energy);
    
    double u = scaled_ising;
    double v = scaled_kitaev;
    double p = scaled_spin;

    double dudt = (u - u) / dt;
    double dvdt = (v - v) / dt;
    double dudx = (u - u) / dx;
    double dvdy = (v - v) / dx;
    double d2udx2 = (u - 2.0 * u + u) / (dx * dx);
    double d2vdy2 = (v - 2.0 * v + v) / (dx * dx);
    double dpdx = (p - p) / dx;
    double dpdy = (p - p) / dx;

    double continuity_loss = pow(dudx + dvdy, 2);
    double momentum_x_loss = pow(dudt + u * dudx + v * dvdy + (1.0 / RHO) * dpdx - NU * (d2udx2 + d2vdy2), 2);
    double momentum_y_loss = pow(dvdt + u * dvdy + v * dvdy + (1.0 / RHO) * dpdy - NU * (d2udx2 + d2vdy2) - G, 2);

    return (continuity_loss + momentum_x_loss + momentum_y_loss) * scale_factor;
}

double heat_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx) {
    double scale_factor = 1e-40;
    
    double scaled_ising = scale_energy(ising_energy);
    double scaled_kitaev = scale_energy(kitaev_energy);
    double scaled_spin = scale_energy(spin_energy);
    
    double T1 = scaled_ising;
    double T2 = scaled_kitaev;
    double T3 = scaled_spin;

    double dTdt = (T2 - T1) / dt;
    double d2Tdx2 = (T3 - 2.0 * T2 + T1) / (dx * dx);

    return pow(dTdt - ALPHA * d2Tdx2, 2) * scale_factor;
}

double wave_loss(double ising_energy, double kitaev_energy, double spin_energy, double dt, double dx) {
    double scale_factor = 1e-40;
    
    double scaled_ising = scale_energy(ising_energy);
    double scaled_kitaev = scale_energy(kitaev_energy);
    double scaled_spin = scale_energy(spin_energy);
    
    double u1 = scaled_ising;
    double u2 = scaled_kitaev;
    double u3 = scaled_spin;

    double d2udt2 = (u3 - 2.0 * u2 + u1) / (dt * dt);
    double d2udx2 = (u3 - 2.0 * u2 + u1) / (dx * dx);

    return pow(d2udt2 - C * C * d2udx2, 2) * scale_factor;
}