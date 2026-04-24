# Topological Quantum Visualization Tool

This document provides a comprehensive explanation of the visualization tool for the Spin-Based Neural Computation Framework, which offers interactive visual representations of key topological quantum phenomena.

## Overview

The visualization tool presents four distinct modes that illustrate fundamental concepts in topological quantum computing, each highlighting a different aspect of quantum topology:

1. **Berry Curvature Visualization** (Press key 1)
2. **Toric Code Error Correction** (Press key 2)
3. **Majorana Zero Modes** (Press key 3)
4. **Topological Entanglement Entropy** (Press key 4)

## Berry Curvature Visualization

### Visual Elements

The Berry curvature visualization represents the topological properties of energy bands in momentum space (k-space):

- **Color-coded grid**: A square lattice representing the Brillouin zone in momentum space
- **Red regions**: Positive Berry curvature (analogous to positive magnetic field)
- **Blue regions**: Negative Berry curvature (analogous to negative magnetic field)
- **Brightness variation**: Intensity corresponds to the magnitude of the Berry curvature
- **Moving particles**: Quantum flux visualized as particles that emerge from regions with strong curvature

### Physical Meaning

The Berry curvature is a fundamental geometric quantity that characterizes how the quantum wavefunction's phase changes as parameters are varied. In this visualization:

- The integrated Berry curvature over the entire Brillouin zone equals the **Chern number**, a topological invariant
- Chern number = 1 indicates a **topologically non-trivial state** with protected edge states
- This directly corresponds to the **quantum Hall effect** and the quantization of Hall conductance
- Similar to how a magnetic field affects charged particles, Berry curvature affects electron motion in crystalline solids
- The mathematical expression for Berry curvature is: F(k) = ∇ × A(k), where A(k) is the Berry connection

### Scientific Relevance

The visualization demonstrates how topological insulators differ from ordinary insulators:

- Topological insulators have **non-zero Chern numbers** and protected edge states
- The bulk-boundary correspondence principle connects the bulk topological invariant (Chern number) to the number of protected edge modes
- This forms the basis for topological band theory, which describes materials like topological insulators and Weyl semimetals
- The Nobel Prize in Physics 2016 was awarded partly for theoretical discoveries of topological phase transitions and topological phases of matter

## Toric Code Error Correction

### Visual Elements

The toric code visualization illustrates Kitaev's model for topological quantum error correction:

- **Square lattice grid**: Represents the physical qubits positioned on edges of the lattice
- **Green squares**: Plaquette operators (Z⊗Z⊗Z⊗Z stabilizers) applied to the four qubits surrounding each face
- **Blue squares**: Vertex operators (X⊗X⊗X⊗X stabilizers) applied to the four qubits meeting at each vertex
- **Red elements**: Error syndromes where stabilizer measurements yield -1 eigenvalues instead of +1
- **Particle effects**: Error propagation shown as moving particles emanating from error sites

### Physical Meaning

The toric code is a foundational model for topological quantum error correction:

- It encodes quantum information in a **topologically protected subspace**
- The ground state of the system satisfies all stabilizer constraints (all green and blue operators yield +1)
- **Qubit errors** (bit flips or phase flips) cause localized violations of these constraints
- The visualization shows how **error syndromes** (red markers) appear at the endpoints of error chains
- The **code distance** corresponds to the minimum number of physical qubit operations needed to create a logical error

### Scientific Relevance

The toric code has profound implications for quantum computing:

- It demonstrates how **quantum information can be protected** from local noise and decoherence
- It represents the simplest example of **topological quantum error correction**
- The code's strength increases with system size, making it **scalable**
- The error threshold is approximately 11%, one of the highest known for quantum error correction codes
- It provides a theoretical foundation for **fault-tolerant quantum computing** - a requirement for practical quantum computers

## Majorana Zero Modes

### Visual Elements

The Majorana zero modes visualization shows a chain of Majorana fermions:

- **Circular arrangement**: Represents a 1D Kitaev chain wrapped into a circle
- **Yellow squares**: γ_A Majorana operators (one half of a regular fermion)
- **Purple squares**: γ_B Majorana operators (the other half of a regular fermion)
- **Connecting lines**: Pairs of Majorana operators that form an ordinary fermion
- **Brightness variations**: Represent occupation values - brighter indicates higher occupation probability
- **Particle effects**: Quantum fluctuations in the Majorana modes

### Physical Meaning

Majorana fermions are exotic quasiparticles with unique properties:

- Each Majorana fermion is its **own antiparticle** (γ = γ†)
- They appear at **zero energy** in topological superconductors
- They obey **non-Abelian statistics** - exchanging them transforms the system's state in a way that depends on the order of operations
- Two Majorana modes can be combined to form a single regular fermion: c = (γ_A + iγ_B)/2
- Unpaired Majorana modes can exist at the **edges** of the system or at **defects**

### Scientific Relevance

Majorana zero modes are a central focus in quantum computing research:

- They form the basis for **topological qubits** that are inherently protected from decoherence
- The **Kitaev chain model** (visualized here) is the simplest system exhibiting these modes
- Microsoft is pursuing a **topological quantum computing platform** based on Majorana zero modes
- The visualization demonstrates how Majorana modes could be manipulated for quantum computation
- Recent experimental evidence suggests Majorana modes exist in certain semiconductor-superconductor heterostructures

## Topological Entanglement Entropy

### Visual Elements

The topological entanglement entropy visualization depicts:

- **Tripartite disk**: A circular region divided into three equal sectors
- **Red region**: Region A of the Kitaev-Preskill construction
- **Green region**: Region B of the Kitaev-Preskill construction
- **Blue region**: Region C of the Kitaev-Preskill construction
- **Colored particles**: Quantum fluctuations representing entanglement between regions
- **Particle density**: Related to the strength of the topological order

### Physical Meaning

Topological entanglement entropy quantifies quantum entanglement unique to topologically ordered states:

- It measures **long-range quantum entanglement** that is independent of the boundary size
- The Kitaev-Preskill construction eliminates boundary contributions to entanglement
- The formula is: S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC, where S represents entropy
- For a Z₂ topological order (like the toric code), γ = +log(2) ≈ +0.693
- The positive TEE indicates long-range entanglement; the area-law term is subtracted by the Kitaev-Preskill combination

### Scientific Relevance

Topological entanglement entropy is a powerful diagnostic tool in quantum many-body physics:

- It serves as an **order parameter** for detecting topological phases
- It distinguishes between trivial insulators and topologically ordered states
- It is directly related to the **quantum dimension** of the anyonic excitations in the system
- It can be connected to the **ground state degeneracy** on surfaces with non-trivial topology
- The concept underpins our understanding of topological quantum computation and fault tolerance

## Implementation Details

The visualization uses a simplified model of each topological phenomenon:

1. **Berry curvature**: Simulated as a vortex-like structure in momentum space with integrated Chern number = 1
2. **Toric code**: An 8×8 lattice with randomly generated error syndromes at about 20% error rate
3. **Majorana modes**: 12 Majorana operators (6 sites) arranged in a circle with randomly assigned occupation values
4. **Topological entropy**: Three-region partition with entropy set to the characteristic value log(2)

## Controls and Usage

- **Key 1**: Switch to Berry curvature visualization
- **Key 2**: Switch to Toric code visualization
- **Key 3**: Switch to Majorana zero modes visualization
- **Key 4**: Switch to Topological entanglement entropy visualization
- **ESC key**: Exit the visualization

## Mathematical Background

The visualizations are based on established mathematical models:

- **Berry curvature**: F(k) = ∇ × A(k), where A(k) = i⟨u_k|∇_k|u_k⟩
- **Toric code stabilizers**: A_v = ∏ᵢ∈star(v) σ_i^x and B_p = ∏ᵢ∈∂p σ_i^z
- **Majorana operators**: γ_2j-1 = c_j + c_j† and γ_2j = i(c_j - c_j†)
- **Topological entanglement entropy**: γ = +log(D), where D is the total quantum dimension (γ > 0 signals topological order)

This visualization tool provides an intuitive window into the abstract mathematical concepts that underlie topological quantum computing, making them accessible for research, education, and exploration.

## API Reference

The viewer is an optional SDL2-backed binary. Build with
`make visualization` (requires `pkg-config` + SDL2 headers). The
`include/visualization.h` API is designed so that non-graphical code
can push data into the viewer without depending on SDL2 symbols
directly.

### Opaque handle

```c
typedef struct VisualizationState VisualizationState;
```

`VisualizationState` is opaque — callers never see its internals. It
carries the SDL2 window / renderer handles, the current mode, and the
four per-mode data buffers.

### Lifecycle

```c
VisualizationState *vis = init_visualization();   /* creates window + renderer; NULL on failure */
/* ... set data, call update_visualization() in a loop ... */
cleanup_visualization(vis);                        /* safe on NULL */
```

`init_visualization` pops a window on the current display. On headless
hosts (no `DISPLAY` on Linux, no graphical session on macOS) the call
returns `NULL`; callers should check and skip the viewer.

### Main loop

```c
while (is_visualization_running(vis)) {
    update_visualization(vis);
}
```

- `update_visualization(vis)` renders one frame in the active mode and
  pumps the SDL event queue (keyboard → mode switch, ESC → quit).
- `is_visualization_running(vis)` returns `1` until the user presses
  ESC or closes the window, then `0`.

The viewer is single-threaded — call both functions from the same
thread that invoked `init_visualization`.

### Data setters

Four setter functions push per-mode data. Each may be called while
the viewer is running; the next `update_visualization` tick picks up
the new data.

```c
void set_berry_curvature_data(VisualizationState *state,
                              double *curvature,
                              int resolution,
                              double invariant);
```

- `curvature`: flat array of length `resolution × resolution`,
  row-major, scalar Berry curvature per k-point (sign and magnitude
  both meaningful).
- `resolution`: integer — viewer draws a square grid of that side.
- `invariant`: total Chern number (shown as a HUD label).

```c
void set_toric_code_data(VisualizationState *state,
                         double *stabilizers,
                         int size_x, int size_y);
```

- `stabilizers`: flat array of length `2 × size_x × size_y` — first
  `size_x × size_y` values are star (vertex) stabilizers, next block
  is plaquette stabilizers. Negative values render as error syndromes.

```c
void set_majorana_data(VisualizationState *state,
                       double *positions,
                       int num_modes);
```

- `positions`: flat array of length `num_modes`, representing
  occupation / amplitude of each Majorana mode around the circle.
  Brightness scales with magnitude.

```c
void set_topological_entropy(VisualizationState *state, double entropy);
```

- `entropy`: single scalar, displayed as a HUD label and used to
  modulate the three-region particle density.

### Typical integration pattern

```c
VisualizationState *vis = init_visualization();
if (!vis) return 0;   /* headless — skip viewer */

while (is_visualization_running(vis)) {
    /* compute fresh data from the simulation */
    set_berry_curvature_data(vis, curvature_buf, 64, chern_number);

    update_visualization(vis);   /* renders + polls input */
}
cleanup_visualization(vis);
```

See `src/visualization_main.c` for the full driver that couples the
viewer to the topological computation kernels.

## Build

The SDL2 build uses `pkg-config` to locate SDL2 headers and libs,
falling back to a hardcoded search path list for macOS (Homebrew at
`/opt/homebrew/include/SDL2`) and typical Linux install locations.
Override with `SDL_CFLAGS` / `SDL_LDFLAGS` on the `make` command line
if needed:

```sh
make visualization \
    SDL_CFLAGS="-I/my/sdl2/include" \
    SDL_LDFLAGS="-L/my/sdl2/lib -lSDL2"
```

The resulting binary is `build/visualization`.
