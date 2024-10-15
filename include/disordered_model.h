#ifndef DISORDERED_MODEL_H
#define DISORDERED_MODEL_H

#include "ising_model.h"
#include "kitaev_model.h"

void add_disorder_to_ising_lattice(IsingLattice *ising_lattice, double disorder_strength); // Prototype for adding disorder
void add_disorder_to_kitaev_lattice(KitaevLattice *kitaev_lattice, double disorder_strength); // Prototype for adding disorder to Kitaev model

#endif // DISORDERED_MODEL_H
