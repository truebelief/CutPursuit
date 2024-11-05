# Cut Pursuit with L2 Norm

A Python implementation of the Cut Pursuit algorithm using L2 norm for graph optimization problems. This package provides working tools for graph partitioning using max-flow/min-cut optimization. While inspired by the original C++ version , this implementation is not an exact replica but more focused on speed optimization.

Some parameters such as the cut-off threshold are discarded for simplicity. For the original C++ implementation of the Cut Pursuit algorithm, please refer to the [cut-pursuit repository](https://github.com/loicland/cut-pursuit).

Several max-flow libraries have been evaluated, including PyMaxflow, SciPy's sparse module, NetworkX, and iGraph. Among these, PyMaxflow demonstrated the fastest performance. Notably, NetworkX offers CUDA support via RAPIDS and cuGraph; however, as of now, max-flow integration is lacking, and Windows support is limited.


## Installation

You can install the package via pip:

```bash
pip install cut-pursuit-l2
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/truebelief/CutPursuit.git
```

## Requirements

- Python >=3.7
- NumPy <2.0.0
- SciPy
- PyMaxflow

## Basic Usage

Here's a simple example of how to use the Cut Pursuit algorithm:

```python
import numpy as np
from cut_pursuit import perform_cut_pursuit

# Generate sample point cloud data
points = np.random.rand(1000, 3)

# Set parameters
K = 4  # number of nearest neighbors
lambda_ = 1.0  # regularization strength

# Run Cut Pursuit
components = perform_cut_pursuit(K, lambda_, points)
```


## Advanced Usage

For more control over the algorithm, you can use the `CutPursuit` class directly:

```python
from cut_pursuit import CutPursuit, CPParameter

# Create instance
cp = CutPursuit(n_vertices=1000)

# Set custom parameters
cp.set_parameters(
    flow_steps=4,
    max_ite_main=20,
    stopping_ratio=0.001,
    reg_strenth=1.0
)

# Run optimization
energy_values, computation_times = cp.run()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
