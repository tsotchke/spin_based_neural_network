CC = gcc
CFLAGS = -O2 -Wall -std=c11 -Iinclude

all: spin_based_neural_computation

spin_based_neural_computation: src/main.c src/ising_model.c src/kitaev_model.c src/disordered_model.c src/reinforcement_learning.c src/neural_network.c src/physics_loss.c src/quantum_mechanics.c src/spin_models.c src/energy_utils.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f spin_based_neural_computation
