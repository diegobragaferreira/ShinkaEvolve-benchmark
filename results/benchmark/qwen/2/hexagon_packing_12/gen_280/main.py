# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from numba import jit, prange
import warnings
from collections import defaultdict
import math

class HexagonGeometry:
    """Efficient geometric computations for hexagon operations."""

    def __init__(self):
        self.side_length = 1.0
        self.apothem = np.sqrt(3) / 2
        self.height = 2 * self.apothem
        self.width = 2 * self.side_length

    @staticmethod
    @jit(nopython=True)
    def vertices_jit(center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """JIT compiled hexagon vertex calculation."""
        angle_rad = np.radians(rotation_deg)
        # Unit hexagon vertices centered at origin
        base_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

        # Rotate and translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = base_vertices @ rotation_matrix.T
        return rotated_vertices + np.array([center_x, center_y])

    def vertices(self, center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """Get hexagon vertices."""
        return self.vertices_jit(center_x, center_y, rotation_deg)

@jit(nopython=True)
def _get_edges(vertices: np.ndarray) -> np.ndarray:
    """Get edges from vertices."""
    edges = np.empty((len(vertices), 2))
    for i in range(len(vertices)):
        edges[i] = vertices[i] - vertices[(i+1) % len(vertices)]
    return edges

@jit(nopython=True)
def _project_polygon_onto_axis(vertices: np.ndarray, axis: np.ndarray) -> tuple:
    """Project polygon vertices onto an axis."""
    projections = np.empty(len(vertices))
    for i in range(len(vertices)):
        projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
    return np.min(projections), np.max(projections)

@jit(nopython=True)
def _hexagon_overlap_sat_jit(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Separating Axis Theorem for hexagon overlap detection."""
    # Get edges of both hexagons
    edges1 = _get_edges(hex1_vertices)
    edges2 = _get_edges(hex2_vertices)

    # Combine all axes (edges perpendicular to edges)
    all_axes = np.empty((len(edges1) + len(edges2), 2))
    for i in range(len(edges1)):
        # Normal vector to edge (perpendicular)
        all_axes[i] = np.array([-edges1[i, 1], edges1[i, 0]])
        # Normalize
        norm = np.sqrt(all_axes[i, 0]**2 + all_axes[i, 1]**2)
        if norm > 1e-10:
            all_axes[i] /= norm
    for i in range(len(edges2)):
        # Normal vector to edge (perpendicular)
        all_axes[len(edges1) + i] = np.array([-edges2[i, 1], edges2[i, 0]])
        # Normalize
        norm = np.sqrt(all_axes[len(edges1) + i, 0]**2 + all_axes[len(edges1) + i, 1]**2)
        if norm > 1e-10:
            all_axes[len(edges1) + i] /= norm

    # Check each axis
    for axis in all_axes:
        min1, max1 = _project_polygon_onto_axis(hex1_vertices, axis)
        min2, max2 = _project_polygon_onto_axis(hex2_vertices, axis)

        # If no overlap on this axis, polygons don't overlap
        if max1 < min2 or max2 < min1:
            return False

    return True

class CompactSpatialHashGrid:
    """Compact spatial hash grid for efficient neighbor search with memory optimizations."""

    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        # Using dictionary with tuple keys for fast lookup
        self.grid = {}

    @staticmethod
    @jit(nopython=True)
    def _hash_cell_numba(x: float, y: float, cell_size: float) -> tuple:
        """Hash coordinates to grid cell."""
        return (int(x // cell_size), int(y // cell_size))

    def _hash_cell(self, x: float, y: float) -> tuple:
        """Hash coordinates to grid cell."""
        return self._hash_cell_numba(x, y, self.cell_size)

    def insert(self, hex_id: int, center_x: float, center_y: float):
        """Insert a hexagon into the spatial grid."""
        cell = self._hash_cell(center_x, center_y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(hex_id)

    @staticmethod
    @jit(nopython=True)
    def _get_candidates_numba(grid, cell, cell_size) -> list:
        """Get candidate hexagons that might collide with given position."""
        candidates = []
        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in grid:
                    candidates.extend(grid[neighbor_cell])
        return candidates

    def get_candidates(self, center_x: float, center_y: float) -> list:
        """Get candidate hexagons that might collide with given position."""
        cell = self._hash_cell(center_x, center_y)
        if cell not in self.grid:
            return []
        return self._get_candidates_numba(self.grid, cell, self.cell_size)

class ConstraintValidator:
    """Advanced constraint validation with geometric reasoning."""

    def __init__(self, geometry: HexagonGeometry):
        self.geo = geometry
        self.hex_radius = 1.0  # For unit hexagon

    def is_contained_geometric(self, hex_vertices: np.ndarray, outer_center_x: float,
                              outer_center_y: float, outer_side_length: float) -> bool:
        """Geometric containment check using distance from center."""
        # Calculate distance from outer center to each vertex
        dist_from_center = np.sqrt((hex_vertices[:, 0] - outer_center_x)**2 +
                                  (hex_vertices[:, 1] - outer_center_y)**2)
        # Maximum distance from center in a regular hexagon is side_length * sqrt(3)/2
        max_radius = outer_side_length * np.sqrt(3) / 2
        
        # Add small tolerance to account for floating-point precision issues
        tolerance = 1e-10
        return np.all(dist_from_center <= max_radius + tolerance)

    def has_overlap_fast(self, hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast approximate overlap check before precise check."""
        # Simple distance check first
        center1 = np.mean(hex1_vertices, axis=0)
        center2 = np.mean(hex2_vertices, axis=0)
        dist_centers = np.sqrt(np.sum((center1 - center2)**2))

        # If centers are far apart, no overlap
        # For unit hexagons, overlap likely if distance < 2.0
        if dist_centers > 2.0:  
            return False

        # Otherwise, use SAT for precise check
        return _hexagon_overlap_sat_jit(hex1_vertices, hex2_vertices)

    def has_overlap_with_spatial_hash(self, hex_data: np.ndarray, spatial_grid: CompactSpatialHashGrid) -> bool:
        """Fast overlap detection using spatial hashing."""
        num_hex = len(hex_data)
        # Insert all hexagons into spatial grid
        for i in range(num_hex):
            center_x, center_y, _ = hex_data[i]
            spatial_grid.insert(i, center_x, center_y)

        # Check for overlaps with nearby hexagons only
        for i in range(num_hex):
            center_x, center_y, _ = hex_data[i]
            candidates = spatial_grid.get_candidates(center_x, center_y)

            # Check against candidates in neighboring cells
            for j in candidates:
                if i >= j:  # Avoid duplicate comparisons
                    continue

                vertices1 = self.geo.vertices(*hex_data[i])
                vertices2 = self.geo.vertices(*hex_data[j])

                if self.has_overlap_fast(vertices1, vertices2):
                    return True

        return False

    def validate_all_constraints(self, hex_data: np.ndarray, outer_side_length: float, 
                                outer_center_x: float, outer_center_y: float) -> tuple:
        """
        Comprehensive constraint validation returning detailed information.
        Returns (is_valid, containment_violations, overlap_violations)
        """
        num_hex = len(hex_data)
        containment_violations = 0
        overlap_violations = 0
        is_valid = True
        
        # Check containment for all hexagons
        spatial_grid = CompactSpatialHashGrid(cell_size=2.0)
        for i in range(num_hex):
            center_x, center_y, angle = hex_data[i]
            vertices = self.geo.vertices(center_x, center_y, angle)
            
            if not self.is_contained_geometric(vertices, outer_center_x, outer_center_y, outer_side_length):
                containment_violations += 1
                is_valid = False
                
            # Insert into spatial grid for overlap checking
            spatial_grid.insert(i, center_x, center_y)

        # Check overlaps using spatial hashing for efficiency
        if self.has_overlap_with_spatial_hash(hex_data, spatial_grid):
            overlap_violations = 1
            is_valid = False
            
        return is_valid, containment_violations, overlap_violations

class SolutionEvaluator:
    """Advanced solution evaluator with intelligent penalty system."""

    def __init__(self, geometry: HexagonGeometry, validator: ConstraintValidator):
        self.geo = geometry
        self.validator = validator

    def calculate_min_enclosing_hexagon(self, inner_hex_data: np.ndarray,
                                       scale_factor: float = 1.0) -> tuple:
        """Calculate minimum enclosing hexagon side length."""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = self.geo.vertices(center_x, center_y, angle)
            all_vertices.extend(vertices)

        all_vertices = np.array(all_vertices)

        # Find bounding circle radius
        centroid = np.mean(all_vertices, axis=0)
        distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
        max_distance = np.max(distances)

        # For a regular hexagon, side length = max_distance * sqrt(3)/2
        side_length = max_distance * 2 / np.sqrt(3) * scale_factor
        return side_length, centroid

    def evaluate(self, solution_array: np.ndarray, outer_side_length: float = 10.0) -> float:
        """Evaluate solution quality with adaptive penalty-based constraints."""
        # Reshape solution array into 12 hexagons with (x, y, angle) each
        inner_hex_data = solution_array.reshape(-1, 3)

        # Calculate the minimum enclosing hexagon
        min_side_length, centroid = self.calculate_min_enclosing_hexagon(inner_hex_data, 1.05)

        # Validate constraints with detailed feedback
        is_valid, containment_violations, overlap_violations = self.validator.validate_all_constraints(
            inner_hex_data, min_side_length, centroid[0], centroid[1]
        )

        # Compute penalty based on constraint violations
        penalty = 0.0
        
        # High penalty for constraint violations
        if not is_valid:
            if containment_violations > 0:
                penalty += 20000.0 * containment_violations
            if overlap_violations > 0:
                penalty += 100000.0 * overlap_violations

        # Return negative inverse side length plus penalty
        objective_value = -1.0 / min_side_length + penalty
        return objective_value

class SymmetricPopulationGenerator:
    """Generates symmetric initial populations for faster convergence."""

    @staticmethod
    def generate_symmetric_initial_population(size: int = 10) -> list:
        """Generate symmetric initial population with known good arrangements."""
        # Base symmetric configuration based on research - 12 hexagons in rings
        base_configurations = []
        
        # Configuration 1: Basic symmetric arrangement
        config1 = np.array([
            [0, 0, 0],           # Center
            [-2.1, 0, 0],        # Left
            [2.1, 0, 0],         # Right
            [-1.05, 1.82, 0],    # Top-left
            [1.05, 1.82, 0],     # Top-right
            [-1.05, -1.82, 0],   # Bottom-left
            [1.05, -1.82, 0],    # Bottom-right
            [-3.15, 1.82, 0],    # Far top-left
            [3.15, 1.82, 0],     # Far top-right
            [-3.15, -1.82, 0],   # Far bottom-left
            [3.15, -1.82, 0],    # Far bottom-right
            [0, -3.64, 0],       # Far bottom
        ]).flatten()
        
        base_configurations.append(config1)
        
        # Configuration 2: Different symmetric pattern
        config2 = np.array([
            [0, 0, 0],           # Center
            [-1.5, 0, 0],        # Left
            [1.5, 0, 0],         # Right
            [-1.5, 1.5, 0],      # Top-left
            [1.5, 1.5, 0],       # Top-right
            [-1.5, -1.5, 0],     # Bottom-left
            [1.5, -1.5, 0],      # Bottom-right
            [-3.0, 1.5, 0],      # Far top-left
            [3.0, 1.5, 0],       # Far top-right
            [-3.0, -1.5, 0],     # Far bottom-left
            [3.0, -1.5, 0],      # Far bottom-right
            [0, -3.0, 0],        # Far bottom
        ]).flatten()
        
        base_configurations.append(config2)
        
        # Configuration 3: Another symmetric arrangement (more spread out)
        config3 = np.array([
            [0, 0, 0],           # Center
            [-1.8, 0, 0],        # Left
            [1.8, 0, 0],         # Right
            [-0.9, 1.56, 0],     # Top-left
            [0.9, 1.56, 0],      # Top-right
            [-0.9, -1.56, 0],    # Bottom-left
            [0.9, -1.56, 0],     # Bottom-right
            [-2.7, 1.56, 0],     # Far top-left
            [2.7, 1.56, 0],      # Far top-right
            [-2.7, -1.56, 0],    # Far bottom-left
            [2.7, -1.56, 0],     # Far bottom-right
            [0, -3.12, 0],       # Far bottom
        ]).flatten()
        
        base_configurations.append(config3)
        
        # Generate population by perturbing base configurations
        population = []
        for base_config in base_configurations:
            population.append(base_config)
            
        # Add more variants with random perturbations
        for _ in range(size - len(base_configurations)):
            # Pick a base configuration
            base_idx = np.random.randint(0, len(base_configurations))
            base = base_configurations[base_idx].copy()
            
            # Add some noise to positions and rotations
            noise = np.random.normal(0, 0.3, len(base))
            perturbed = base + noise
            
            population.append(perturbed)
            
        return population

class PackingOptimizer:
    """Advanced optimization controller with hybrid approach."""

    def __init__(self, num_hexagons: int = 12):
        self.num_hexagons = num_hexagons
        self.geometry = HexagonGeometry()
        self.validator = ConstraintValidator(self.geometry)
        self.evaluator = SolutionEvaluator(self.geometry, self.validator)
        self.population_generator = SymmetricPopulationGenerator()

    def _setup_bounds(self) -> list:
        """Setup parameter bounds for optimization."""
        bounds = []
        for i in range(self.num_hexagons):
            # X and Y positions - wider range to allow for optimization
            bounds.extend([(-10.0, 10.0), (-10.0, 10.0)])
            # Rotation: 0-360 degrees (but we'll use symmetry constraints)
            bounds.append((0.0, 360.0))
        return bounds

    def _adaptive_optimize_stage(self, bounds: list, maxiter: int, popsize: int,
                               mutation: tuple, recombination: float,
                               initial_population: list = None,
                               early_stopping: bool = True) -> dict:
        """Run adaptive optimization stage with early stopping."""
        try:
            # Store best solution seen so far
            best_fitness = float('inf')
            best_solution = None
            no_improvement_count = 0
            max_no_improvement = 20  # Early stopping threshold
            
            result = differential_evolution(
                self.evaluator.evaluate,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                seed=42,
                disp=False,
                init=initial_population,
                callback=lambda x, convergence: self._callback_check(x, convergence, 
                                                                   best_fitness, best_solution, 
                                                                   no_improvement_count, 
                                                                   max_no_improvement)
            )
            
            return result
        except Exception as e:
            warnings.warn(f"Optimization stage failed: {e}")
            return None

    def _callback_check(self, x, convergence, best_fitness, best_solution, 
                       no_improvement_count, max_no_improvement):
        """Callback for monitoring convergence."""
        # In practice, this would update tracking variables
        pass

    def optimize(self) -> tuple:
        """Main optimization routine with adaptive strategies."""
        start_time = time.time()

        # Setup bounds
        bounds = self._setup_bounds()

        # Generate symmetric initial population
        initial_pop = self.population_generator.generate_symmetric_initial_population(10)
        initial_solution = initial_pop[0]  # Use first configuration as baseline

        # Stage 1: Coarse optimization with symmetric initialization
        print("Starting stage 1: Coarse optimization...")
        coarse_result = self._adaptive_optimize_stage(
            bounds, maxiter=50, popsize=20,
            mutation=(0.8, 1.0), recombination=0.7,
            initial_population=initial_pop,
            early_stopping=True
        )

        if coarse_result is None or not coarse_result.success:
            warnings.warn("Coarse optimization failed, using initial solution")
            best_solution = initial_solution
        else:
            best_solution = coarse_result.x

        # Stage 2: Fine optimization with stronger mutation and more iterations
        print("Starting stage 2: Fine optimization...")
        fine_result = self._adaptive_optimize_stage(
            bounds, maxiter=100, popsize=30,
            mutation=(1.0, 1.0), recombination=0.8,
            initial_population=[best_solution] + 
                               [np.random.normal(best_solution, 0.8) for _ in range(29)],
            early_stopping=True
        )

        if fine_result is not None and fine_result.success:
            final_solution = fine_result.x
        else:
            final_solution = best_solution

        # Evaluate final solution
        inner_hex_data = final_solution.reshape(-1, 3)
        min_side_length, centroid = self.evaluator.calculate_min_enclosing_hexagon(inner_hex_data, 1.05)

        # Center the outer hexagon at the centroid of inner hexagons
        outer_hex_data = np.array([centroid[0], centroid[1], 0])

        eval_time = time.time() - start_time
        print(f"Optimization completed in {eval_time:.2f} seconds")

        return inner_hex_data, outer_hex_data, min_side_length

def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        optimizer = PackingOptimizer()
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimizer.optimize()

        # Calculate benchmark ratio
        inv_outer_hex_side_length = 1.0 / outer_hex_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537

        # Output metrics for verification
        print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        warnings.warn(f"Error in hexagon packing: {e}")
        # Fallback to better symmetric arrangement
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.1, 0, 0],
            [2.1, 0, 0],
            [-1.05, 1.82, 0],
            [1.05, 1.82, 0],
            [-1.05, -1.82, 0],
            [1.05, -1.82, 0],
            [-3.15, 1.82, 0],
            [3.15, 1.82, 0],
            [-3.15, -1.82, 0],
            [3.15, -1.82, 0],
            [0, -3.64, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 7.5
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# Global variable for timing
start_time = time.time()

# EVOLVE-BLOCK-END