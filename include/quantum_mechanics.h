#ifndef QUANTUM_MECHANICS_H
#define QUANTUM_MECHANICS_H

#include <complex.h>
#include "ising_model.h"
#include "kitaev_model.h"
#include "spin_models.h"

// Remove function prototypes for internal functions
void apply_quantum_effects(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, SpinLattice *spin_lattice, double noise_level);
void simulate_entanglement(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, double entanglement_prob);

#endif // QUANTUM_MECHANICS_H