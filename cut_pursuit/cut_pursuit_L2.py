"""
Cut Pursuit Algorithm Implementation.

This module implements the Cut Pursuit algorithm for graph optimization problems.
The algorithm performs graph partitioning using max-flow/min-cut optimization.
"""

import time
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import maxflow

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class CPParameter:
    """Parameters for Cut Pursuit algorithm.

    Attributes:
        reg_strenth (float): Regularization strength parameter
        flow_steps (int): Number of flow steps in optimization
        max_ite_main (int): Maximum number of main iterations
        stopping_ratio (float): Stopping criterion threshold
    """
    reg_strenth: float = 0.0
    flow_steps: int = 3
    max_ite_main: int = 6
    stopping_ratio: float = 0.0001


class CutPursuit:
    """Implementation of the Cut Pursuit algorithm (L2 only, no cutoff).

    This class implements graph partitioning using max-flow/min-cut optimization.

    Args:
        n_vertices (int, optional): Number of vertices in the graph. Defaults to 1.
    """

    def __init__(self, n_vertices: int = 1):
        """Initialize the Cut Pursuit algorithm.

        Args:
            n_vertices (int, optional): Number of vertices in graph. Defaults to 1.
        """
        self.n_vertex = n_vertices
        self.dim = 1  # Will be updated in setup_cp
        self.parameter = CPParameter()

        # Graph structure
        self.n_total_vertices = n_vertices + 2  # Including source and sink
        self.source = n_vertices
        self.sink = n_vertices + 1

        self.vertex_weights = np.zeros(self.n_total_vertices)
        self.vertex_observations = None  # Will be initialized in setup_cp
        self.vertex_values = None
        self.vertex_colors = np.full(self.n_total_vertices, -1)
        self.vertex_components = np.zeros(self.n_total_vertices, dtype=int)

        # Edge structure using structured array
        self.edges_dtype = np.dtype([
            ('u', np.int32),
            ('v', np.int32),
            ('weight', np.float32),
            ('capacity', np.float32),
            ('is_active', np.bool_),
            ('real_edge', np.bool_)
        ])
        self.edges = np.zeros(0, dtype=self.edges_dtype)
        self.n_edge = 0

        # Component structure
        self.max_components = n_vertices
        self.component_indices = [[] for _ in range(self.max_components)]
        self.root_vertex = np.zeros(self.max_components, dtype=np.int32)
        self.saturated_vertices_logi = np.zeros(n_vertices, dtype=bool)
        self.n_active_components = 1
        self.saturated_components = np.zeros(1, dtype=bool)

        # Initialize source and sink
        self.vertex_weights[self.source] = 1.0
        self.vertex_weights[self.sink] = 1.0

        # Cache for optimization
        self.flow_graph = maxflow.Graph[int]()

    def set_parameters(
            self,
            flow_steps: int = 4,
            max_ite_main: int = 20,
            stopping_ratio: float = 0.001,
            reg_strenth: float = 0.0
    ) -> None:
        """Set parameters for the Cut Pursuit algorithm.

        Args:
            flow_steps (int, optional): Number of flow steps. Defaults to 4.
            max_ite_main (int, optional): Maximum iterations. Defaults to 20.
            stopping_ratio (float, optional): Stopping threshold. Defaults to 0.001.
            reg_strenth (float, optional): Regularization strength. Defaults to 0.0.
        """
        self.parameter.flow_steps = flow_steps
        self.parameter.max_ite_main = max_ite_main
        self.parameter.stopping_ratio = stopping_ratio
        self.parameter.reg_strenth = reg_strenth

    def initialize(self) -> None:
        """Initialize the graph structure for optimization."""
        self.component_indices[0] = list(range(self.n_vertex))
        self.root_vertex[0] = 0
        self.vertex_components[:self.n_vertex] = 0

        self.compute_value(0)

        n_source_sink_edges = 4 * self.n_vertex
        source_sink_edges = np.zeros(n_source_sink_edges, dtype=self.edges_dtype)

        vertices = np.arange(self.n_vertex)
        idx = np.arange(0, n_source_sink_edges, 4)

        # Set up source and sink connections
        source_sink_edges['u'][idx] = self.source
        source_sink_edges['v'][idx] = vertices
        source_sink_edges['u'][idx + 1] = vertices
        source_sink_edges['v'][idx + 1] = self.source
        source_sink_edges['u'][idx + 2] = vertices
        source_sink_edges['v'][idx + 2] = self.sink
        source_sink_edges['u'][idx + 3] = self.sink
        source_sink_edges['v'][idx + 3] = vertices

        if len(self.edges) > 0:
            self.edges = np.concatenate([self.edges, source_sink_edges])
        else:
            self.edges = source_sink_edges
        self.n_edge = len(self.edges)

    def run(self) -> Tuple[List[float], List[float]]:
        """Run the main optimization loop.

        Returns:
            Tuple[List[float], List[float]]: Lists of energy values and computation times.
        """
        self.initialize()

        print(f"Graph with {self.n_vertex} vertices and {len(self.edges)} edges "
              f"and observation of dimension {self.dim}")

        # Initial energy computation
        fidelity_energy, penalty_energy = self.compute_energy()
        energy_zero = fidelity_energy
        old_energy = fidelity_energy + penalty_energy

        # Pre-allocate arrays for benchmarking
        energy_out = np.zeros(self.parameter.max_ite_main)
        time_out = np.zeros(self.parameter.max_ite_main)
        start_time = time.time()

        # Main optimization loop
        for ite_main in range(self.parameter.max_ite_main):
            # Compute optimal binary partition
            saturation = self.split()

            # Reduce graph
            self.reduce()

            # Compute new energy
            fidelity_energy, penalty_energy = self.compute_energy()
            current_total_energy = fidelity_energy + penalty_energy

            # Store benchmarking data
            energy_out[ite_main] = current_total_energy
            time_out[ite_main] = time.time() - start_time

            print(
                f"Iteration {ite_main + 1:3} - {self.n_active_components:4} components - "
                f"Saturation {100.0 * saturation / self.n_vertex:5.1f}% - "
                f"Quadratic Energy {100 * fidelity_energy / energy_zero:6.3f}% - "
                f"Timer {time_out[ite_main]:.2f}s"
            )

            # Check stopping criteria
            if saturation == self.n_vertex:
                print("All components are saturated")
                break

            if ((old_energy - current_total_energy) / old_energy
                    < self.parameter.stopping_ratio):
                print("Stopping criterion reached")
                break

            old_energy = current_total_energy

        return (
            energy_out[:ite_main + 1].tolist(),
            time_out[:ite_main + 1].tolist()
        )

    def compute_value(self, ind_com: int) -> Tuple[np.ndarray, float]:
        """Compute the optimal value for a component.

        Args:
            ind_com (int): Component index.

        Returns:
            Tuple[np.ndarray, float]: Component value and total weight.
        """
        comp_vertices = self.component_indices[ind_com]
        weights = self.vertex_weights[comp_vertices]
        total_weight = np.sum(weights)

        if total_weight > 0:
            comp_value = np.sum(weights[:, np.newaxis] * self.vertex_observations[comp_vertices], axis=0) / total_weight
        else:
            comp_value = np.zeros(self.dim)

        self.vertex_values[comp_vertices] = comp_value
        self.vertex_components[comp_vertices] = ind_com

        return comp_value, total_weight

    def compute_energy(self) -> Tuple[float, float]:
        """Compute the current energy of the solution.

        Returns:
            Tuple[float, float]: Fidelity energy and penalty energy.
        """
        mask = self.vertex_weights[:self.n_vertex] > 0

        # Compute differences vectorized
        diff = (
                self.vertex_observations[:self.n_vertex][mask] -
                self.vertex_values[:self.n_vertex][mask]
        )
        weights = self.vertex_weights[:self.n_vertex][mask]

        # Compute fidelity energy
        fidelity_energy = 0.5 * np.sum(weights * np.sum(diff * diff, axis=1))

        # Compute regularization energy (penalty term)
        active_edges = self.edges[self.edges['is_active'] & self.edges['real_edge']]
        penalty_energy = 0.5 * self.parameter.reg_strenth * np.sum(active_edges['weight'])

        return fidelity_energy, penalty_energy

    def split(self) -> int:
        """Perform the split step of the algorithm.

        Returns:
            int: Number of saturated vertices.
        """
        binary_label = np.zeros(self.n_vertex, dtype=bool)
        self.init_labels(binary_label)

        # Pre-allocate centers array
        centers = np.zeros((self.n_active_components, 2, self.dim))

        self.edge_mask = ~self.edges['is_active'] & self.edges['real_edge']
        self.real_edges = self.edges[self.edge_mask]

        # Flow approximation loop
        for _ in range(self.parameter.flow_steps):
            self.compute_centers(centers, binary_label)
            self.set_capacities(centers)
            binary_label.fill(False)
            source_idx = self.compute_max_flow()
            binary_label[source_idx] = True

        self.vertex_colors.fill(False)
        self.vertex_colors[source_idx] = True
        return self.activate_edges()

    def init_labels(self, binary_label: np.ndarray) -> None:
        """Initialize binary labels using Quickshift.

        Args:
            binary_label (np.ndarray): Array to store binary labels.
        """
        active_comps = np.where(~self.saturated_components)[0]

        for ind_com in active_comps:
            comp_vertices = self.component_indices[ind_com]
            observations = self.vertex_observations[comp_vertices]
            variances = np.var(observations, axis=0, ddof=0)
            var_dim = np.argmax(variances)
            median_value = np.median(observations[:, var_dim])
            binary_label[comp_vertices] = observations[:, var_dim] > median_value

    def compute_centers(self, centers: np.ndarray, binary_label: np.ndarray) -> None:
        """Compute centers for binary partitioning.

        Args:
            centers (np.ndarray): Array to store computed centers.
            binary_label (np.ndarray): Binary labels for vertices.
        """
        active_comps = np.where(~self.saturated_components)[0]

        # Initialize arrays for computation
        total_weights_label0 = np.zeros(len(active_comps))
        total_weights_label1 = np.zeros(len(active_comps))
        sum_obs_label0 = np.zeros((len(active_comps), self.dim))
        sum_obs_label1 = np.zeros((len(active_comps), self.dim))

        for idx, ind_com in enumerate(active_comps):
            comp_vertices = self.component_indices[ind_com]
            weights = self.vertex_weights[comp_vertices]
            observations = self.vertex_observations[comp_vertices]
            labels = binary_label[comp_vertices]

            # Compute sums and weights for both labels
            weights_label0 = weights[~labels]
            weights_label1 = weights[labels]

            if len(weights_label0) == 0 or len(weights_label1) == 0:
                self.saturated_components[ind_com] = True
                centers[ind_com] = self.vertex_values[comp_vertices[0]]
                continue

            obs_label0 = observations[~labels]
            obs_label1 = observations[labels]

            total_weights_label0[idx] = np.sum(weights_label0)
            total_weights_label1[idx] = np.sum(weights_label1)

            sum_obs_label0[idx] = np.sum(
                obs_label0 * weights_label0[:, np.newaxis],
                axis=0
            )
            sum_obs_label1[idx] = np.sum(
                obs_label1 * weights_label1[:, np.newaxis],
                axis=0
            )

        # Compute centers for all components at once
        centers[active_comps, 0] = (
                sum_obs_label0 / total_weights_label0[:, np.newaxis]
        )
        centers[active_comps, 1] = (
                sum_obs_label1 / total_weights_label1[:, np.newaxis]
        )

    def set_capacities(self, centers: np.ndarray) -> None:
        """Set capacities for max-flow computation.

        Args:
            centers (np.ndarray): Centers for binary partitioning.
        """
        SCALE_FACTOR = 1000

        # Initialize graph for this iteration
        self.flow_graph.reset()
        node_ids = self.flow_graph.add_nodes(self.n_vertex)

        # Initialize arrays for terminal capacities
        source_caps = np.zeros(self.n_vertex, dtype=np.int64)
        sink_caps = np.zeros(self.n_vertex, dtype=np.int64)

        # Process each component
        for ind_com in range(self.n_active_components):
            if self.saturated_components[ind_com]:
                continue

            comp_vertices = np.array(
                self.component_indices[ind_com],
                dtype=np.int32
            )
            if len(comp_vertices) == 0:
                continue

            # Compute costs
            obs_diff0 = (
                    self.vertex_observations[comp_vertices] -
                    centers[ind_com, 0]
            )
            obs_diff1 = (
                    self.vertex_observations[comp_vertices] -
                    centers[ind_com, 1]
            )
            vertex_weights = self.vertex_weights[comp_vertices]

            cost_B = 0.5 * np.sum(obs_diff0 ** 2, axis=1) * vertex_weights
            cost_notB = 0.5 * np.sum(obs_diff1 ** 2, axis=1) * vertex_weights

            # Determine capacities for source and sink
            mask_to_sink = cost_B <= cost_notB
            cost_diff = np.abs(cost_B - cost_notB) * SCALE_FACTOR
            cost_diff = cost_diff.astype(np.int64)

            # Assign capacities
            source_caps[comp_vertices] = np.where(~mask_to_sink, cost_diff, 0)
            sink_caps[comp_vertices] = np.where(mask_to_sink, cost_diff, 0)

        # Add terminal edges in batch
        self.flow_graph.add_grid_tedges(node_ids, source_caps, sink_caps)

        # Add regular edges efficiently
        if len(self.real_edges) > 0:
            edge_caps = (
                    self.real_edges['weight'] *
                    self.parameter.reg_strenth *
                    SCALE_FACTOR
            ).astype(np.int64)

            # Add edges with forward and backward capacities
            self.flow_graph.add_edges(
                self.real_edges['u'].astype(np.int32),
                self.real_edges['v'].astype(np.int32),
                edge_caps,
                edge_caps
            )

    def compute_max_flow(self) -> np.ndarray:
        """Compute maximum flow using pymaxflow.

        Returns:
            np.ndarray: Array of reachable vertices.
        """
        self.flow_graph.maxflow()
        reachable = np.where(
            self.flow_graph.get_grid_segments(np.arange(self.n_vertex))
        )[0]
        return reachable

    def activate_edges(self) -> int:
        """Activate edges based on vertex colors.

        Returns:
            int: Number of saturated vertices.
        """
        # Compute saturation directly
        saturation = sum(
            len(self.component_indices[i])
            for i in range(self.n_active_components)
            if self.saturated_components[i]
        )

        # Find crossing edges efficiently
        edges_mask = self.edges['real_edge']
        u_colors = self.vertex_colors[self.edges['u'][edges_mask]]
        v_colors = self.vertex_colors[self.edges['v'][edges_mask]]
        crossing_edges = u_colors != v_colors
        crossing_indices = np.where(edges_mask)[0][crossing_edges]

        # Activate edges
        self.edges['is_active'][crossing_indices] = True
        self.edge_mask = ~self.edges['is_active'] & self.edges['real_edge']
        self.real_edges = self.edges[self.edge_mask]
        return saturation

    def reduce(self) -> None:
        """Compute reduced graph and perform backward step if needed."""
        self.compute_connected_components()
        n_comp = self.n_active_components
        for ind_com in range(n_comp):
            self.compute_value(ind_com)

    def compute_connected_components(self) -> None:
        """Compute connected components of the graph."""
        if len(self.real_edges) == 0:
            vertex_indices = np.arange(self.n_vertex, dtype=np.int32)
            self.n_active_components = self.n_vertex
            self.component_indices = [[i] for i in vertex_indices]
            self.root_vertex = vertex_indices.copy()
            self.vertex_components[:self.n_vertex] = vertex_indices
            self.saturated_components = np.zeros(self.n_vertex, dtype=bool)
            return

        # Create sparse matrix for connected components
        graph = csr_matrix(
            (
                np.ones(len(self.real_edges), dtype=bool),
                (self.real_edges['u'], self.real_edges['v'])
            ),
            shape=(self.n_vertex, self.n_vertex),
            dtype=bool
        )

        # Compute connected components
        n_components, labels = connected_components(
            graph, directed=False, return_labels=True
        )

        self.n_active_components = n_components

        # Group vertices by component label efficiently
        sort_idx = np.argsort(labels)
        sorted_labels = labels[sort_idx]

        # Find boundaries between components
        boundaries = np.nonzero(np.diff(sorted_labels))[0] + 1
        boundaries = np.concatenate([[0], boundaries, [len(labels)]])

        # Create component indices
        self.component_indices = np.split(sort_idx, boundaries[1:-1])

        # Set root vertices
        self.root_vertex = np.array(
            [indices[0] if len(indices) else 0 for indices in self.component_indices],
            dtype=np.int32
        )
        self.vertex_components[:self.n_vertex] = labels
        self.saturated_components = np.zeros(n_components, dtype=bool)


def setup_cp(
        n_nodes: int,
        n_edges: int,
        n_obs: int,
        observation: np.ndarray,
        eu: np.ndarray,
        ev: np.ndarray,
        edge_weight: np.ndarray,
        node_weight: np.ndarray
) -> CutPursuit:
    """Set up Cut Pursuit algorithm with initial data.

    Args:
        n_nodes (int): Number of nodes in the graph.
        n_edges (int): Number of edges in the graph.
        n_obs (int): Number of observations per node.
        observation (np.ndarray): Node observations.
        eu (np.ndarray): Edge source vertices.
        ev (np.ndarray): Edge target vertices.
        edge_weight (np.ndarray): Edge weights.
        node_weight (np.ndarray): Node weights.

    Returns:
        CutPursuit: Initialized Cut Pursuit instance.
    """
    cp = CutPursuit(n_nodes)
    cp.dim = n_obs

    cp.vertex_observations = np.zeros((cp.n_total_vertices, n_obs))
    cp.vertex_values = np.zeros((cp.n_total_vertices, n_obs))
    cp.vertex_weights[:n_nodes] = node_weight
    cp.vertex_observations[:n_nodes] = observation

    edges = np.zeros(2 * n_edges, dtype=cp.edges_dtype)

    # Forward edges
    edges['u'][:n_edges] = eu
    edges['v'][:n_edges] = ev
    edges['weight'][:n_edges] = edge_weight
    edges['capacity'][:n_edges] = edge_weight
    edges['real_edge'][:n_edges] = True

    # Reverse edges
    edges['u'][n_edges:] = ev
    edges['v'][n_edges:] = eu
    edges['weight'][n_edges:] = edge_weight
    edges['capacity'][n_edges:] = edge_weight
    edges['real_edge'][n_edges:] = True

    cp.edges = edges
    cp.n_edge = len(edges)

    return cp


def cut_pursuit(
        n_nodes: int,
        n_edges: int,
        n_obs: int,
        observation: np.ndarray,
        eu: np.ndarray,
        ev: np.ndarray,
        edge_weight: np.ndarray,
        node_weight: np.ndarray,
        lambda_: float
) -> Tuple[np.ndarray, List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
    """Main cut pursuit function with optimized setup and execution.

    Args:
        n_nodes (int): Number of nodes in the graph.
        n_edges (int): Number of edges.
        n_obs (int): Number of observations per node.
        observation (np.ndarray): Node observations.
        eu (np.ndarray): Edge source vertices.
        ev (np.ndarray): Edge target vertices.
        edge_weight (np.ndarray): Edge weights.
        node_weight (np.ndarray): Node weights.
        lambda_ (float): Regularization parameter.

    Returns:
        Tuple containing:
            - vertex_values (np.ndarray): Computed vertex values
            - component_indices (List[List[int]]): Component memberships
            - vertex_components (np.ndarray): Component assignments
            - energy_out (np.ndarray): Energy values per iteration
            - time_out (np.ndarray): Computation times per iteration
    """
    # Set random seed for reproducibility
    np.random.seed(1)

    print("L0-CUT PURSUIT")

    # Setup and run cut pursuit
    cp = setup_cp(
        n_nodes, n_edges, n_obs, observation,
        eu, ev, edge_weight, node_weight
    )

    # Set parameters
    cp.parameter.flow_steps = 4
    cp.parameter.max_ite_main = 20
    cp.parameter.stopping_ratio = 0.001
    cp.parameter.reg_strenth = lambda_

    # Run optimization
    energy_out, time_out = cp.run()

    return (
        cp.vertex_values[:n_nodes].copy(),
        cp.component_indices[:cp.n_active_components],
        cp.vertex_components[:n_nodes].copy(),
        np.array(energy_out),
        np.array(time_out)
    )


def perform_cut_pursuit(K: int, lambda_: float, pc: np.ndarray) -> np.ndarray:
    """Perform cut pursuit on point cloud data.

    Args:
        K (int): Number of nearest neighbors.
        lambda_ (float): Regularization parameter.
        pc (np.ndarray): Point cloud data.

    Returns:
        np.ndarray: Component assignments for each point.
    """
    point_count = len(pc)
    if point_count == 0:
        return False

    # Build KD-tree and find nearest neighbors
    kdtree = cKDTree(pc[:, :3])
    _, nn_idx = kdtree.query(pc, k=K + 1)

    # Remove self-connections and get indices
    indices = nn_idx[:, 1:]

    # Create edge list
    n_nodes = len(pc)
    n_obs = 3
    n_edges = n_nodes * K

    eu = np.repeat(np.arange(n_nodes), K)
    ev = indices.ravel()

    # Center the point cloud
    y = pc[:, :3] - np.mean(pc[:, :3], axis=0)

    # Edge weights following C++ implementation
    edge_weight = np.ones_like(eu)
    node_weight = np.ones(point_count)

    # Run cut pursuit
    _, _, in_component, _, _ = cut_pursuit(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_obs=n_obs,
        observation=y,
        eu=eu,
        ev=ev,
        edge_weight=edge_weight,
        node_weight=node_weight,
        lambda_=lambda_
    )

    return in_component


def decimate_pcd(columns: np.ndarray, min_res: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate point cloud to minimum resolution.

    Args:
        columns (np.ndarray): Point cloud data.
        min_res (float): Minimum resolution.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Decimation indices and inverse indices.
    """
    _, block_idx_uidx, block_inverse_idx = np.unique(
        np.floor(columns[:, :3] / min_res).astype(np.int32),
        axis=0,
        return_index=True,
        return_inverse=True
    )
    return block_idx_uidx, block_inverse_idx


if __name__ == "__main__":
    import cProfile
    import pstats
    import io

    # Algorithm parameters
    K = 4
    reg_strength = 1.0
    min_res = 0.05

    # Load and process point cloud
    path_to_pcd_txt = r"data\TestDemo.txt"
    output_path=r"output\TestDemoRes.txt"

    # Read LAS file
    pcd=np.loadtxt(path_to_pcd_txt)

    # Decimate point cloud
    dec_idx_uidx, dec_inverse_idx = decimate_pcd(pcd[:, :3], min_res)
    pcd_dec = pcd[dec_idx_uidx]

    # Create profiler
    pr = cProfile.Profile()
    pr.enable()

    # Time the algorithm
    t0 = time.time()
    in_component = perform_cut_pursuit(K, reg_strength, pcd_dec)
    main_algo_time = time.time() - t0

    pr.disable()

    # Print profiling statistics
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)

    print("\n=== Overall Timing ===")
    print(f"Main algorithm time: {main_algo_time:.2f} seconds")
    print("\n=== Detailed Function Profiling ===")
    print(s.getvalue())

    np.savetxt(output_path,np.concatenate([pcd[:,:3],in_component[dec_inverse_idx][:,np.newaxis]],axis=-1),fmt="%.3f %.3f %.3f %d")
    print(f"\nResult saved to {output_path}")