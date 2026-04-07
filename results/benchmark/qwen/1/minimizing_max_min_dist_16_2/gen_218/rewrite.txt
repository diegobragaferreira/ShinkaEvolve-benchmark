# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import time

class PointConfiguration:
    """Efficient representation of point configurations with lazy evaluation."""
    
    def __init__(self, points):
        self._points = np.asarray(points)
        self._n_points = len(points)
        self._distance_matrix = None
        self._dmin = None
        self._dmax = None
        self._ratio = None
        
    @property
    def points(self):
        return self._points
        
    @property
    def n_points(self):
        return self._n_points
        
    @property
    def distance_matrix(self):
        """Compute and cache distance matrix only when needed."""
        if self._distance_matrix is None:
            if self._n_points < 2:
                self._distance_matrix = np.zeros((0, 0))
            else:
                self._distance_matrix = squareform(pdist(self._points))
                # Set diagonal to infinity to exclude self-distances
                np.fill_diagonal(self._distance_matrix, np.inf)
        return self._distance_matrix
        
    @property
    def dmin(self):
        """Compute and cache minimum distance."""
        if self._dmin is None:
            if self._n_points < 2:
                self._dmin = 0
            else:
                finite_distances = self.distance_matrix[np.isfinite(self.distance_matrix)]
                self._dmin = np.min(finite_distances) if len(finite_distances) > 0 else 0
        return self._dmin
        
    @property
    def dmax(self):
        """Compute and cache maximum distance."""
        if self._dmax is None:
            if self._n_points < 2:
                self._dmax = 0
            else:
                finite_distances = self.distance_matrix[np.isfinite(self.distance_matrix)]
                self._dmax = np.max(finite_distances) if len(finite_distances) > 0 else 0
        return self._dmax
        
    @property
    def min_max_ratio(self):
        """Compute and cache ratio of minimum to maximum distance."""
        if self._ratio is None:
            if self.dmax == 0:
                self._ratio = 0
            else:
                self._ratio = self.dmin / self.dmax
        return self._ratio
        
    def compute_distance_matrix(self):
        """Force recomputation of distance matrix."""
        self._distance_matrix = None
        self._dmin = None
        self._dmax = None
        self._ratio = None
        return self.distance_matrix
        
    def get_clipped_points(self, lower=0, upper=1):
        """Get points clipped to specified bounds."""
        return np.clip(self._points, lower, upper)
        
    def copy(self):
        """Create a copy of this configuration."""
        return PointConfiguration(self._points.copy())


class ObjectiveFunction:
    """Encapsulates the optimization objective function."""
    
    def __init__(self, configuration_class=PointConfiguration):
        self.configuration_class = configuration_class
        
    def __call__(self, x_flat):
        """Evaluate objective function (negative ratio for minimization)."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Ensure points are within bounds [0,1]
        points = np.clip(points, 0, 1)
        
        # Create configuration
        config = self.configuration_class(points)
        
        # Compute ratio
        ratio = config.min_max_ratio
        
        # Return negative because we want to maximize
        return -ratio


class Optimizer:
    """Manages different optimization strategies and multi-start approaches."""
    
    def __init__(self, objective_function, max_evaluations=1000):
        self.objective_function = objective_function
        self.max_evaluations = max_evaluations
        self.bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates
        
    def _differential_evolution_optimization(self, initial_points):
        """Perform differential evolution optimization."""
        x0 = initial_points.flatten()
        try:
            de_result = differential_evolution(
                self.objective_function,
                self.bounds,
                maxiter=self.max_evaluations // 10,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-8,
                strategy='best1bin'
            )
            return de_result.x if de_result.success else x0
        except Exception:
            return x0
            
    def _local_refinement(self, x0):
        """Apply local refinement using L-BFGS-B."""
        try:
            refined_result = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            return refined_result.x if refined_result.success else x0
        except Exception:
            return x0
            
    def optimize_single(self, initial_points):
        """Single optimization run with robust fallbacks."""
        # Differential evolution for global search
        x_opt = self._differential_evolution_optimization(initial_points)
        
        # Local refinement
        x_opt = self._local_refinement(x_opt)
        
        # Final check
        try:
            final_result = minimize(
                self.objective_function,
                x_opt,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            return final_result.x if final_result.success else x_opt
        except Exception:
            return x_opt


class PointGenerator:
    """Generates various initial point configurations."""
    
    @staticmethod
    def generate_hexagonal_grid():
        """Generate a hexagonal grid arrangement."""
        points = []
        rows = 4
        cols = 4

        # Hexagonal packing parameters
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Height of equilateral triangle

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                points.append([x, y])

        return np.array(points)
        
    @staticmethod
    def generate_golden_spiral():
        """Generate a golden spiral arrangement."""
        indices = np.arange(16)
        golden_angle = 2.399963229728653
        angles = golden_angle * indices
        # Use logarithmic distribution for better point spreading
        radii = np.log(indices + 1) / np.log(16)
        golden_spiral = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        return np.clip(golden_spiral, 0, 1)
        
    @staticmethod
    def generate_icosahedral_points():
        """
        Generate points based on icosahedral symmetry.
        """
        # Icosahedron vertices (normalized to unit sphere)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Project to 2D using stereographic projection or simple cylindrical
        points_2d = vertices[:, :2] * 0.4 + 0.5  # Scale and center

        # Take first 16 points by combining different approaches
        additional_points = [
            [0.5, 0.5],  # center
            [0.2, 0.2],  # corner
            [0.8, 0.8],  # opposite corner
            [0.2, 0.8]   # edge point
        ]

        # Combine icosahedral points with additional strategic points
        combined_points = np.vstack([points_2d[:12], additional_points])

        # Ensure we have exactly 16 points
        if len(combined_points) > 16:
            combined_points = combined_points[:16]
        elif len(combined_points) < 16:
            # Fill with random points near the existing ones
            np.random.seed(42)
            extra_points = np.random.rand(16 - len(combined_points), 2)
            combined_points = np.vstack([combined_points, extra_points])

        return np.clip(combined_points, 0, 1)
        
    @staticmethod
    def generate_initial_strategies():
        """Generate multiple initial point configurations."""
        strategies = {}
        
        # Strategy 1: Hexagonal grid
        strategies['hex'] = PointGenerator.generate_hexagonal_grid()
        
        # Strategy 2: Perturbed hexagonal grid
        np.random.seed(42)
        perturbed_hex = strategies['hex'] + np.random.normal(0, 0.02, strategies['hex'].shape)
        strategies['hex_perturbed'] = np.clip(perturbed_hex, 0, 1)
        
        # Strategy 3: Regular grid with jitter
        regular_grid = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                regular_grid.append([x, y])
        regular_grid = np.array(regular_grid)
        jittered_grid = regular_grid + np.random.normal(0, 0.01, regular_grid.shape)
        strategies['grid_jittered'] = np.clip(jittered_grid, 0, 1)
        
        # Strategy 4: Golden spiral
        strategies['spiral'] = PointGenerator.generate_golden_spiral()
        
        # Strategy 5: Icosahedral distribution (mathematical optimality)
        strategies['icosahedral'] = PointGenerator.generate_icosahedral_points()
        
        # Strategy 6: Random points with higher spread
        np.random.seed(123)
        random_points = np.random.rand(16, 2)
        strategies['random'] = np.clip(random_points, 0.05, 0.95)
        
        return strategies


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Setup
    objective_func = ObjectiveFunction()
    optimizer = Optimizer(objective_func, max_evaluations=1000)
    point_generator = PointGenerator()
    
    # Generate initial strategies
    strategies = point_generator.generate_initial_strategies()
    
    # Find the best initial configuration
    best_ratio = 0
    best_points = None
    
    for name, points in strategies.items():
        config = PointConfiguration(points)
        ratio = config.min_max_ratio
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Multi-start optimization with different initial variations
    for restart in range(5):
        # Generate new variation of the initial points
        np.random.seed(restart + 1000)
        perturbed_points = best_points.copy()
        noise_level = 0.03 + restart * 0.005  # Gradually increasing noise
        perturbed_points += np.random.normal(0, noise_level, best_points.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Optimize this variant
        optimized_x = optimizer.optimize_single(perturbed_points)
        optimized_points = optimized_x.reshape(-1, 2)
        
        # Evaluate result
        config = PointConfiguration(optimized_points)
        optimized_ratio = config.min_max_ratio
        
        if optimized_ratio > best_ratio:
            best_ratio = optimized_ratio
            best_points = optimized_points.copy()
    
    # Final optimization on the best configuration found
    final_x = optimizer.optimize_single(best_points)
    final_points = final_x.reshape(-1, 2)
    
    # One final refinement attempt
    np.random.seed(9999)
    last_attempt = final_points + np.random.normal(0, 0.01, final_points.shape)
    last_attempt = np.clip(last_attempt, 0, 1)
    
    # Check if this gives better result
    final_config = PointConfiguration(final_points)
    previous_ratio = final_config.min_max_ratio
    
    refined_x = optimizer.optimize_single(last_attempt)
    refined_points = refined_x.reshape(-1, 2)
    refined_config = PointConfiguration(refined_points)
    refined_ratio = refined_config.min_max_ratio
    
    if refined_ratio > previous_ratio:
        return refined_points
    else:
        return final_points

# EVOLVE-BLOCK-END