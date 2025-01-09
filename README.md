# Cut Pursuit with L2 Norm

A Python implementation of the Cut Pursuit algorithm using L2 norm for graph optimization problems. This package provides working tools for graph partitioning using max-flow/min-cut optimization. While inspired by the original C++ version , this implementation is not an exact replica but more focused on speed optimization.

Some parameters, like the cut-off threshold, have been discarded for simplicity. For the original C++ version of the Cut Pursuit algorithm, see [cut-pursuit repository](https://github.com/loicland/cut-pursuit).

The main code is contained in **cut_pursuit/cut_pursuit_L2.py**. For those interested in a version closer to the original C++ code, cut_pursuit_L2_cpp_replica.py is also provided.

Various max-flow libraries were tested, including PyMaxflow, SciPy's sparse module, NetworkX, and iGraph. PyMaxflow proved to be the fastest. NetworkX has CUDA support through RAPIDS and cuGraph, but currently lacks max-flow functionality and has limited support on Windows.

## Example

The Cut Pursuit algorithm enables robust point clustering that maintains reasonable shape and boundaries of clusters, as shown in the example below:

<div align="center">
  <img width="340" alt="Cut Pursuit Clustering Example" src="https://github.com/user-attachments/assets/3697909c-2bc4-441a-ac58-4d382bf969e6">
</div>

I used this algorithm as the initial segmentation step for tree clustering in CloudCompare's [treeiso plugin](https://github.com/truebelief/cc-treeiso-plugin).

## Installation

You can install the package via pip - [cut-pursuit-l2](https://pypi.org/project/cut-pursuit-l2/):

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

## References

If you use this repository, please cite the following paper:
@Article{landrieu2017cut,
title={Cut pursuit: Fast algorithms to learn piecewise constant functions on general weighted graphs},
author={Landrieu, Loic and Obozinski, Guillaume},
journal={SIAM Journal on Imaging Sciences},
year={2017},
publisher={SIAM}
}
