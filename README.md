# Spin-Based Neural Computation Framework

## Overview
The **Spin-Based Neural Computation Framework** is an advanced simulation library that combines quantum mechanics, reinforcement learning, and energy-based neural networks to study and model physical phenomena. Designed for researchers in AI, scientific computing, and statistical physics, this framework explores spin systems—such as the Ising, Kitaev, and disordered models—through machine learning and physics-informed neural networks. It has wide-ranging applications in quantum computing, materials science, and artificial intelligence, making it a versatile and powerful tool.

## Project Highlights
- **Quantum Physics and Machine Learning**: Leverage spin-based neural networks to simulate and learn from quantum systems.
- **Comprehensive Spin Models**: Implements Ising, Kitaev, and disordered spin models with customizable lattice parameters and interaction strengths.
- **Energy-Driven Neural Networks**: Inspired by Boltzmann machines, these networks learn from physical principles of spin configuration energies.
- **Reinforcement Learning Integration**: Dynamically optimize spin configurations using reinforcement learning, enabling robust energy minimization.
- **Customizable Parameters and Loss Functions**: Tailor simulations with specific lattice dimensions, couplings, activation functions, and physics-based loss functions.
- **Extensive Logging**: Ensure reproducibility with detailed logs, supporting research and analysis.

## Key Components

### 1. Neural Networks
- **Thermodynamic Energy Minimization**: Learn optimal spin states based on energy minimization, leveraging principles from statistical mechanics.
- **Dynamic Structure**: Configurable neural network layers and neuron counts to match simulation complexity requirements.

### 2. Spin Models
- **Ising Model**: A classical model for ferromagnetic interactions, extended to three-dimensional lattices for detailed phase transition studies.
- **Kitaev Model**: A model with applications to topological quantum computing, allowing for exploration of spin liquids and complex quantum states.
- **Spin Model**: A model incorporating classical spin and random disorder, useful for studying real-world material impurities.

### 3. Quantum Mechanics
- **Quantum State Calculations**: Calculate properties of quantum spins, including superposition and entanglement, integrating quantum effects directly with neural networks.
- **Hybrid Quantum-Classical Systems**: Model interactions between quantum and classical elements in spin systems.

### 4. Reinforcement Learning
- **Policy and Value Optimization**: Adapts spin configurations based on calculated rewards, enabling energy-efficient configurations.
- **Energy-Based Learning**: Works in tandem with neural networks to achieve refined spin configurations and energy states.

## Directory Structure
The project is organized for clarity and usability:

```
spin_based_neural_network/
├── src
│   ├── disordered_model.c
│   ├── energy_utils.c
│   ├── ising_model.c
│   ├── kitaev_model.c
│   ├── main.c
│   ├── neural_network.c
│   ├── physics_loss.c
│   ├── quantum_mechanics.c
│   └── reinforcement_learning.c
├── include
│   ├── disordered_model.h
│   ├── energy_utils.h
│   ├── ising_model.h
│   ├── kitaev_model.h
│   ├── neural_network.h
│   ├── physics_loss.h
│   ├── reinforcement_learning.h
│   └── spin_models.h
├── Makefile
└── README.md
```

## Installation
The framework is written in C and has no external dependencies, ensuring compatibility across a variety of systems. Compile with a standard C compiler, such as `gcc`, for easy setup:

```bash
make
```

## Usage
```bash
./spin_based_neural_computation [OPTIONS]
```

### Command Line Arguments
- `-i, --iterations N`: Number of iterations to run (default: 100)
- `-v, --verbose`: Enable detailed output
- `--lattice-size X Y Z`: Set lattice dimensions (default: 10x10x10)
- `--jx JX, --jy JY, --jz JZ`: Set coupling constants for the Kitaev model (default: 1.0, 1.0, -1.0)
- `--initial-state STATE`: Initial state configuration (options: `random`, `all-up`)
- `--dx DX, --dt DT`: Spatial and time step sizes (default: 0.1)
- `--loss-type TYPE`: Specify physics-informed loss (options: `heat`, `schrodinger`, `maxwell`, `navier_stokes`, `wave`)
- `--activation FUNC`: Choose activation function (options: `relu`, `tanh`, `sigmoid`)
- `--log LOG_FILE`: Specify log file for output (default: `simulation.log`)

### Example Usage
1. **Run a basic simulation with default settings:**

    ```bash
    ./spin_based_neural_computation --loss-type heat --activation relu
    ```

2. **Run with 500 iterations and verbose output, using the Schrödinger loss function:**

    ```bash
    ./spin_based_neural_computation -i 500 -v --loss-type schrodinger  --activation sigmoid
    ```

3. **Specify lattice size and coupling constants for a Kitaev model simulation:**

    ```bash
    ./spin_based_neural_computation --lattice-size 15 15 15 --jx 0.5 --jy 1.0 --jz -0.8
    ```

4. **Use the Navier-Stokes loss function with Tanh activation, saving output to a custom log file:**

    ```bash
    ./spin_based_neural_computation --loss-type navier_stokes --activation tanh --log custom_output.log
    ```

5. **Set custom spatial and time steps, with a specific initial state configuration:**

    ```bash
    ./spin_based_neural_computation --dx 0.05 --dt 0.02 --initial-state all-up --activation relu
    ```

6. **Navier-Stokes Simulation with Sigmoid Activation**:
    ```bash
    ./spin_based_neural_computation --iterations 500 --verbose --lattice-size 15 15 15 --jx 1.5 --jy 0.8 --jz -0.5 --initial-state random --dx 0.05 --dt 0.02 --loss-type navier_stokes --log navier_stokes.log --activation sigmoid
    ```

7. **Heat Equation Simulation with ReLU Activation**:
    ```bash
    ./spin_based_neural_computation --lattice-size 30 30 30 --loss-type heat --activation relu --initial-state all-down
    ```

### Available Activation Functions and Loss Types
**Activation Functions:**
- `relu`: Rectified Linear Unit, ideal for general purpose networks.
- `tanh`: Hyperbolic tangent, suitable for capturing non-linear dependencies.
- `sigmoid`: Logistic sigmoid, often used for probabilistic outputs.

**Loss Functions:**
- `heat`: Simulates heat diffusion across the lattice.
- `schrodinger`: Models quantum wave functions.
- `maxwell`: Reflects electromagnetic field dynamics.
- `navier_stokes`: Describes fluid flow behavior.
- `wave`: Simulates wave propagation in physical systems.

## Citation
If you use this project in your research, please cite as follows:

```bibtex
@software{SpinBasedNeuralComputation,
  author = {tsotchke},
  title = {Spin-Based Neural Computation Framework},
  year = {2024},
  url = {https://github.com/tsotchke/spin_based_neural_network}
}
```

## License
This project is licensed under the MIT License.

## Conclusion
The **Spin-Based Neural Computation Framework** provides a versatile platform for studying quantum systems, complex spin interactions, and energy-based learning model approaches to neural networks, making it ideal for researchers exploring quantum computing, spintronics, mathematical physics, complexity, and AI. By integrating reinforcement learning and comprehensive spin models, enabling exploration of quantum-inspired neural networks and physics-informed learning, it encourages insightful exploration into physical systems and computational physics.