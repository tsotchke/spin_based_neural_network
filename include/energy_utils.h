#ifndef ENERGY_UTILS_H
#define ENERGY_UTILS_H

#define ENERGY_SCALE 1000.0
#define ENERGY_SCALE_FACTOR 1e-2
#define MIN_ENERGY 1e-10

double scale_energy(double energy);
double unscale_energy(double scaled_energy);

#endif