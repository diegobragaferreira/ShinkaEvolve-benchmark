# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time

class PointConfiguration:
    """Manages point configurations and their distance computations."""
    
    def __init__(self, points):
        self.points = np.array(points)
        self._cached_distances = None
        self._cached_min_dist = None
        self._cached_max_dist = None
        
    def get_distances(self):
        """Get cached or compute pairwise distances."""
        if self._cached_distances is None:
            if len(self.points) < 2:
                self._cached_distances = np.array([])
            else:
                self._cached_distances = pdist(self.points)
        return self._cached_distances
        
    def get_min_max_distances(self):
        """Get min and max distances efficiently."""
        distances = self.get_distances()
        if len(distances) == 0:
            return 0.0, 0.0
            
        self._cached_min_dist = np.min(distances)
        self._cached_max_dist = np.max(distances)
        return self._cached_min_dist, self._cached_max_dist
        
    def get_ratio(self):
        """Compute and return the min/max distance ratio."""
        min_d, max_d = self.get_min_max_distances()
        if max_d <= 1e-12:
            return 0.0
        return min_d / max_d
        
    def copy(self):
        """Create a copy of this configuration."""
        return PointConfiguration(self.points.copy())
        
    def update_point(self, idx, new_point):
        """Update a single point."""
        self.points[idx] = new_point
        # Invalidate cache
        self._invalidate_cache()
        
    def update_points(self, indices, new_points):
        """Update multiple points."""
        for i, idx in enumerate(indices):
            self.points[idx] = new_points[i]
        # Invalidate cache
        self._invalidate_cache()
        
    def _invalidate_cache(self):
        """Invalidate cached distance computations."""
        self._cached_distances = None
        self._cached_min_dist = None
        self._cached_max_dist = None

class HexagonalInitialization:
    """Handles hexagonal grid initialization with symmetry breaking."""
    
    @staticmethod
    def create_hexagonal_grid(n_points=16, seed=42):
        """Create optimized hexagonal grid pattern."""
        np.random.seed(seed)
        
        # Mathematical constants for proper hexagonal packing
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2  # Vertical spacing between rows
        col_spacing = 1.0        # Horizontal spacing between columns
        
        points = []
        
        # Create hexagonal lattice with 4 rows and 4 columns
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                # Alternate column offset for proper hexagonal packing
                x_offset = (i % 2) * 0.5
                x = j * col_spacing + x_offset
                y = i * row_spacing
                points.append([x, y])
        
        # Convert to numpy array
        points = np.array(points[:n_points])
        
        # Normalize to fit within [0,1] bounds properly
        if len(points) > 0:
            # Find min/max ranges to normalize properly
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
        
        # Apply systematic perturbations to break symmetry effectively
        # Different noise intensities based on position index and mathematical pattern
        for i in range(n_points):
            # Apply non-uniform noise to break various symmetries
            # Use sine/cosine patterns to create structured yet asymmetric perturbations
            noise_intensity = 0.01 + 0.005 * math.sin(i * 0.5)
            noise_x = np.random.normal(0, noise_intensity, 1)[0]
            noise_y = np.random.normal(0, noise_intensity, 1)[0]
            points[i] += [noise_x, noise_y]
        
        # Clip to ensure all points stay within valid bounds
        points = np.clip(points, 0, 1)
        
        return points

class SimulatedAnnealingOptimizer:
    """Implements adaptive simulated annealing for point optimization."""
    
    def __init__(self, max_iterations=5000, seed=42):
        self.max_iterations = max_iterations
        self.seed = seed
        np.random.seed(seed)
        
    def optimize(self, initial_points):
        """Run adaptive simulated annealing optimization."""
        current_config = PointConfiguration(initial_points)
        best_config = current_config.copy()
        best_ratio = current_config.get_ratio()
        
        # Advanced cooling schedule with multiple phases
        temperature = 1.0
        cooling_rate = 0.9995
        stagnation_counter = 0
        previous_best = best_ratio
        
        # Phase tracking for adaptive cooling
        phase = 0
        phase_thresholds = [1000, 3000]  # Different cooling rates at different stages
        
        for iteration in range(self.max_iterations):
            # Determine perturbation type with adaptive probability
            if np.random.random() < 0.7:  # 70% chance of neighborhood move
                # Choose neighborhood size adaptively based on iteration
                if iteration < 1000:
                    neighborhood_size = 2
                elif iteration < 3000:
                    neighborhood_size = np.random.randint(2, 4)
                else:
                    neighborhood_size = np.random.randint(2, min(5, len(initial_points)))
                
                # Select indices for neighborhood
                indices = np.random.choice(len(initial_points), neighborhood_size, replace=False).tolist()
                neighbor_config = self._perturb_neighborhood(current_config, indices, temperature * 0.05)
            else:
                # Single point perturbation
                point_idx = np.random.randint(len(initial_points))
                neighbor_config = self._perturb_single(current_config, point_idx, temperature * 0.05)
            
            # Evaluate neighbor solution
            neighbor_ratio = neighbor_config.get_ratio()
            
            # Accept/reject based on Metropolis criterion
            if neighbor_ratio > best_ratio:
                current_config = neighbor_config
                best_config = neighbor_config
                best_ratio = neighbor_ratio
                stagnation_counter = 0
            elif np.random.rand() < math.exp((neighbor_ratio - best_ratio) / temperature):
                current_config = neighbor_config
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adaptive cooling schedule with multiple phases
            if iteration in phase_thresholds:
                phase += 1
                
            # Adjust cooling rate based on phase and stagnation
            if stagnation_counter > 50:
                # If stagnating, cool faster to escape local optima
                temperature *= 0.995
                stagnation_counter = 0
            else:
                # Apply phase-dependent cooling rates
                phase_cooling = cooling_rate * (0.95 if phase > 0 else 1.0)
                temperature *= phase_cooling
            
            # Early stopping with progressive convergence checking
            if iteration % 100 == 0 and iteration > 0:
                current_ratio = best_config.get_ratio()
                if abs(previous_best - current_ratio) < 1e-8:
                    break
                previous_best = current_ratio
                
        return best_config.points

    def _perturb_neighborhood(self, config, indices, step_size=0.005):
        """Perturb a group of points together while maintaining local structure."""
        new_config = config.copy()
        
        # Calculate centroid of selected points for coordinated movement
        centroid = np.mean(config.points[indices], axis=0)
        
        # Apply coordinated perturbations to maintain relative structure
        new_points = []
        for idx in indices:
            # Move each point relative to the centroid
            delta = np.random.uniform(-step_size, step_size, 2)
            new_point = config.points[idx] + delta
            # Boundary check
            new_point = np.clip(new_point, 0, 1)
            new_points.append(new_point)
        
        new_config.update_points(indices, new_points)
        return new_config
    
    def _perturb_single(self, config, idx, step_size=0.005):
        """Perturb a single point."""
        new_config = config.copy()
        delta = np.random.uniform(-step_size, step_size, 2)
        new_point = config.points[idx] + delta
        # Boundary check
        new_point = np.clip(new_point, 0, 1)
        new_config.update_point(idx, new_point)
        return new_config

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Create hexagonal grid initialization
    initial_points = HexagonalInitialization.create_hexagonal_grid(n_points=16, seed=42)
    
    # Optimize using simulated annealing
    optimizer = SimulatedAnnealingOptimizer(max_iterations=5000, seed=42)
    optimized_points = optimizer.optimize(initial_points)
    
    return optimized_points

# EVOLVE-BLOCK-END