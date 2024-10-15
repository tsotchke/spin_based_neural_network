#include "energy_utils.h"
#include <math.h>
#include <float.h>

double scale_energy(double energy) {
    if (fabs(energy) < MIN_ENERGY) {
        return energy >= 0 ? MIN_ENERGY : -MIN_ENERGY;
    }
    return 2.0 / (1.0 + exp(-ENERGY_SCALE_FACTOR * energy)) - 1.0;
}

double unscale_energy(double scaled_energy) {
    if (fabs(scaled_energy) <= MIN_ENERGY) {
        return scaled_energy >= 0 ? MIN_ENERGY / ENERGY_SCALE_FACTOR : -MIN_ENERGY / ENERGY_SCALE_FACTOR;
    }
    return -log((2.0 / (scaled_energy + 1.0)) - 1.0) / ENERGY_SCALE_FACTOR;
}