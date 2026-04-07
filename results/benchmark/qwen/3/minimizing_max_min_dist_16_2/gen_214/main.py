# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import time
import math

class PointDispersionEvolver:
    """
    Evolves point configurations to maximize the ratio of minimum to maximum distance.
    Implements a multi-phase approach: initialization → gradient optimization → local refinement.
    """
    
    def __init__(self, n_points=16, dimensions=2, seed=42, max_time_seconds=180):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        np.random.seed(seed)
        
    def _compute_distance_matrix(self, points):
        """Vectorized computation of pairwise distances."""
        if len(points.shape) == 1:
            points = points.reshape(-1, self.dimensions)
        return pdist(points)
        
    def _compute_ratio(self, points):
        """Compute the min/max distance ratio for given points."""
        distances = self._compute_distance_matrix(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max
        
    def _initialize_grid_points(self):
        """Generate structured grid-based initial configuration."""
        grid_size = int(np.ceil(np.sqrt(self.n_points)))
        spacing = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < self.n_points:
                    points.append([i * spacing, j * spacing])
        return np.array(points)
        
    def _initialize_perturbed_grid(self):
        """Generate grid with controlled perturbations to break symmetry."""
        base_points = self._initialize_grid_points()
        perturbed_points = []
        for point in base_points:
            if len(perturbed_points) < self.n_points:
                x = max(0, min(1, point[0] + np.random.normal(0, 0.05)))
                y = max(0, min(1, point[1] + np.random.normal(0, 0.05)))
                perturbed_points.append([x, y])
        return np.array(perturbed_points)
        
    def _initialize_hexagonal_pattern(self):
        """Generate hexagonal lattice pattern for better dispersion."""
        points = [[0.5, 0.5]]  # center point
        radius = 0.3
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for angle in angles:
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            if len(points) < self.n_points:
                points.append([x, y])
                
        # Fill remaining points in triangular pattern
        while len(points) < self.n_points:
            points.append([np.random.rand(), np.random.rand()])
            
        return np.array(points[:self.n_points])
        
    def _initialize_fibonacci_spiral(self):
        """Generate points using Fibonacci spiral for even distribution."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            x_mapped = (x + 1) / 2
            y_mapped = (z + 1) / 2
            
            points.append([np.clip(x_mapped, 0, 1), np.clip(y_mapped, 0, 1)])
            
        return np.array(points)
        
    def _initialize_random_points(self):
        """Generate completely random points."""
        return np.random.rand(self.n_points, self.dimensions)
        
    def _generate_initial_configurations(self):
        """Create diverse initial configurations."""
        configs = []
        
        # Different structured patterns to provide good diversity
        configs.append(self._initialize_grid_points())
        configs.append(self._initialize_perturbed_grid())
        configs.append(self._initialize_hexagonal_pattern())
        configs.append(self._initialize_fibonacci_spiral())
        configs.append(self._initialize_random_points())
        
        return configs
        
    def _gradient_optimize(self, initial_points, max_iter=500):
        """Use L-BFGS-B for fast gradient-based optimization."""
        def objective(x):
            points = x.reshape(-1, self.dimensions)
            distances = self._compute_distance_matrix(points)
            if len(distances) == 0:
                return -np.inf
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist <= 1e-12:
                return -np.inf
            return min_dist / max_dist

        flat_initial = initial_points.flatten()
        bounds = [(0, 1) for _ in range(len(flat_initial))]
        
        try:
            result = minimize(
                lambda x: -objective(x),
                flat_initial,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                return optimized_points, objective(result.x)
            else:
                warnings.warn(f"L-BFGS-B optimization failed: {result.message}")
                return initial_points, objective(flat_initial)
                
        except Exception as e:
            warnings.warn(f"L-BFGS-B optimization error: {str(e)}")
            return initial_points, objective(flat_initial)
            
    def _simulated_annealing_refinement(self, points, max_iter=2000, initial_temp=0.1, cooling_rate=0.999):
        """Refine solution with adaptive simulated annealing."""
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        temp = initial_temp

        for iteration in range(max_iter):
            # Create neighbor by perturbing one random point
            neighbor_points = current_points.copy()
            idx = np.random.randint(0, len(neighbor_points))

            # Adaptive step size based on iteration
            step_size = 0.03 if iteration < max_iter//2 else 0.01
            neighbor_points[idx, 0] += np.random.normal(0, step_size)
            neighbor_points[idx, 1] += np.random.normal(0, step_size)

            # Keep within bounds
            neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
            neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)

            # Calculate neighbor ratio
            neighbor_ratio = self._compute_ratio(neighbor_points)

            # Accept or reject the neighbor
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                if delta < 0:
                    acceptance_prob = math.exp(delta / temp)
                    if np.random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio

            # Adaptive cooling schedule
            if iteration % 100 == 0 and iteration > 0:
                temp *= 0.9995
            else:
                temp *= cooling_rate

            # Early stopping
            if temp < 1e-8:
                break

        return best_points, best_ratio
        
    def _single_phase_optimize(self, initial_points):
        """Complete optimization pipeline for one initial configuration."""
        # Phase 1: Gradient optimization
        gradient_points, gradient_ratio = self._gradient_optimize(initial_points)
        
        # Phase 2: Local refinement with simulated annealing
        if gradient_ratio > 0.1:  # Only refine if we have a reasonable solution
            refined_points, refined_ratio = self._simulated_annealing_refinement(gradient_points)
            return refined_points if refined_ratio > gradient_ratio else gradient_points
        else:
            return gradient_points
            
    def evolve(self):
        """Main evolution process with multi-start approach."""
        best_points = None
        best_ratio = -np.inf
        start_time = time.time()

        # Generate initial configurations
        initial_configs = self._generate_initial_configurations()

        # Try each configuration with optimization
        for i, initial_config in enumerate(initial_configs):
            if time.time() - start_time > self.max_time_seconds - 5:
                break
                
            try:
                optimized_points = self._single_phase_optimize(initial_config)
                ratio = self._compute_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Error in optimization round {i}: {str(e)}")
                continue

        # Fallback if nothing worked
        if best_points is None:
            return initial_configs[0] if initial_configs else np.random.rand(self.n_points, self.dimensions)
            
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    evolver = PointDispersionEvolver(n_points=16, dimensions=2, seed=42, max_time_seconds=180)
    points = evolver.evolve()
    return points

# EVOLVE-BLOCK-END