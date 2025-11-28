# Fugu Development Guide

## Overview
Fugu is a Python library for building computational neural graphs using Spiking Neural Networks (SNNs). The architecture follows a modular "Bricks and Scaffold" pattern where reusable components (Bricks) are assembled into computational circuits (Scaffolds) that can be compiled to various backend simulators.

## Core Architecture

### The Three-Layer System
1. **Bricks** (`fugu/bricks/`) - Atomic computational units that generate neural circuit graphs
2. **Scaffold** (`fugu/scaffold/`) - Organizes bricks and manages their interconnections via Ports
3. **Backends** (`fugu/backends/`) - Compile scaffolds to different execution environments (SNN, Loihi, STACS, etc.)

### Key Abstraction: Bricks are Algorithms, Not Graphs
**Critical**: A Brick is the _algorithm_ for generating a neural circuit, not the graph itself. The actual NetworkX graph is only constructed when `scaffold.lay_bricks()` is called. Bricks define:
- How to create neurons and synapses
- Connection patterns between them
- Port specifications for inputs/outputs

### Port-Based Connection System
Modern Fugu uses **Ports** for brick interconnections (legacy code uses integer indices). Each port has:
- **Channels**: Named groups of neurons serving specific purposes (typically 'data', 'complete', 'begin')
- **Coding**: Data representation format (e.g., 'Raster', 'temporal-L', 'binary-B', 'Population')
- **Shape**: Tensor arrangement of neurons

**Example Port Usage**:
```python
scaffold.add_brick(brick_a)
scaffold.add_brick(brick_b)
scaffold.connect(brick_a, brick_b, from_port='output', to_port='input')
```

**Legacy Pattern** (still common in examples):
```python
scaffold.add_brick(brick_a, 'input')
scaffold.add_brick(brick_b, input_nodes=[-1])  # -1 = previous brick
```

## Development Patterns

### Creating New Bricks
1. Inherit from `fugu.bricks.Brick` (or `InputBrick` for sources)
2. Override `build2()` for port-based bricks or `build()` for legacy style
3. Define `input_ports()` and `output_ports()` class methods returning `dict[str, PortSpec]`
4. Use `self.generate_neuron_name()` to ensure globally unique neuron names

**Example** (see `fugu/bricks/utility_bricks.py:Dot`):
```python
class MyBrick(Brick):
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        # Create neurons: graph.add_node(name, threshold=1.0, decay=0.0, potential=0.0, p=1.0)
        # Create synapses: graph.add_edge(n1, n2, weight=1.0, delay=1)
        # Return: (graph, metadata, control_nodes_out, output_lists, output_codings)
```

### Neuron Model (LIF-based)
Neurons use Leaky Integrate-and-Fire dynamics (see `neuron_model.md`):
- **potential**: Internal state (like membrane potential)
- **threshold**: Spike threshold
- **decay**: Decay constant ∈ [0,1] (0=no decay)
- **p**: Spike probability when threshold exceeded
- **bias**: External current injection

**Synapse properties**: `weight` (float), `delay` (int timesteps, minimum 1)

### Typical Workflow
```python
from fugu import Scaffold
from fugu.bricks import Vector_Input, TemporalAdder
from fugu.backends import snn_Backend

# 1. Build scaffold
scaffold = Scaffold()
scaffold.add_brick(Vector_Input(data, coding='Raster', time_dimension=True), 'input')
scaffold.add_brick(TemporalAdder(2), input_nodes=[-1], output=True)

# 2. Construct graph
scaffold.lay_bricks()

# 3. Compile to backend
backend = snn_Backend()
backend.compile(scaffold, {'record': 'all'})

# 4. Execute
result = backend.run(n_steps=1000)
```

## Environment Setup

**Conda (Recommended)**:
```bash
conda env create -f fugu_conda_environment.yml
conda activate fugu
conda develop $PWD
```

**Note**: Uses Python 3.7, NetworkX 2.4, NumPy <1.24. The `.conda` local environment pattern is used in this repo.

## Testing

Run tests with pytest:
```bash
pytest                          # All tests
pytest tests/unit              # Unit tests only
coverage run -m pytest && coverage report -m
```

Tests organized as `tests/unit/` and `tests/integration/`. Use `BrickTest` base class for brick tests.

## Code Standards

**Formatted paths**: `tests/`, `fugu/utils/validation.py`, `fugu/simulators/`
**Excluded from formatting**: `fugu/backends/`

Before commit:
```bash
isort --check --filter-files tests fugu/utils/validation.py fugu/simulators
black --check tests fugu/utils/validation.py fugu/simulators
```

Pre-commit hooks enforce Black and isort. To skip formatting in a file:
```python
"""
isort:skip_file
"""
# fmt: off
```

## Backends

Each backend implements the `Backend` ABC with:
- `compile(scaffold, compile_args)` - Convert scaffold to backend representation
- `run(n_steps, return_potential)` - Execute simulation
- `reset()` / `cleanup()` - State management

**Available backends**: `snn_Backend` (default PyTorch-based), `loihi_Backend`, `stacs_backend`, `lava_backend`, `gsearch_backend`

## Special Conventions

### Control Signals
Bricks communicate completion/timing via control neurons in `control_nodes` dict:
- `'complete'`: Fires when brick finishes processing
- `'begin'`: Fires when brick starts (for temporal coding)

### Input Coding Types
Common formats (see `fugu/__init__.py`):
- `'Raster'`: Spike time rasters
- `'temporal-B'/'temporal-L'`: Temporal encoding (Big/Little endian)
- `'binary-B'/'binary-L'`: Binary encoding
- `'Population'`: Population coding
- `'current'`: Direct current injection

### Whetstone Integration
Convert Keras/TensorFlow models to Fugu via `fugu.utils.whetstone_conversion.whetstone_2_fugu()`. Handles Conv2D, Dense, MaxPooling2D layers.

## Common Gotchas

- **Neuron delays**: Minimum delay is 1 timestep (enforced by most backends)
- **Graph immutability**: Once `lay_bricks()` is called, don't modify brick connections
- **Naming conflicts**: Always use `generate_neuron_name()` to avoid collisions
- **Legacy vs Port APIs**: Many examples use legacy `input_nodes` lists; prefer `connect()` for new code
- **Conda path**: Activate the local `.conda` environment: `conda activate /path/to/Fugu_dev/.conda`
