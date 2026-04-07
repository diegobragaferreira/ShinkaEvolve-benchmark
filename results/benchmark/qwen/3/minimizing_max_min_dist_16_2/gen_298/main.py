# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from numba import jit

@jit(nopython=True)
def fast_compute_ratio_numba(points):
    """Fast computation of min/max distance ratio using numba."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    
    min_dist = 1e10
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    if max_dist == 0:
        return 0.0
    return min_dist / max_dist

class HexagonalSimulatedAnnealingOptimizer:
    """Optimized optimizer using hexagonal initialization with simulated annealing."""
    
    def __init__(self, n_points=16, dimensions=2, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _compute_ratio(self, points):
        """Compute ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def _generate_hexagonal_grid(self):
        """Generate optimized hexagonal grid with strong symmetry breaking."""
        points = []
        
        # Create precise 4x4 triangular lattice with proper spacing
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 0.5)
        spacing_y = spacing_x * np.sqrt(3) / 2

        # Generate points in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    x = j * spacing_x + (i % 2) * spacing_x / 2
                    y = i * spacing_y
                    
                    # Stronger symmetry breaking using prime-based and mathematical perturbations
                    prime_i = (i * 7) % 11
                    prime_j = (j * 13) % 17
                    
                    # Apply complex perturbations for better asymmetry
                    x_pert = 0.01 * (
                        np.sin(prime_i * 0.3) * np.cos(prime_j * 0.7) +
                        0.5 * np.sin(prime_i * 0.1 + prime_j * 0.9) +
                        0.3 * np.cos(prime_i * 0.5 + prime_j * 0.2)
                    )
                    
                    y_pert = 0.01 * (
                        np.cos(prime_i * 0.4) * np.sin(prime_j * 0.6) +
                        0.5 * np.cos(prime_i * 0.2 + prime_j * 0.8) +
                        0.3 * np.sin(prime_i * 0.3 + prime_j * 0.4)
                    )
                    
                    # Add position-dependent unique perturbations
                    unique_pert = 0.005 * np.sin((i + j) * 0.7 + i * j * 0.1)
                    
                    points.append([x + x_pert + unique_pert, y + y_pert + unique_pert])

        points = np.array(points[:self.n_points])
        
        # Normalize carefully to unit square
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0 and y_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range * 0.9 + 0.05
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range * 0.9 + 0.05

        # Add small noise for final symmetry breaking
        points += np.random.normal(0, 0.002, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_fibonacci_spiral(self):
        """Generate Fibonacci spiral points."""
        points = np.zeros((self.n_points, self.dimensions))
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.n_points):
            z = 1 - (i / (self.n_points - 1)) * 2
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)
            
            x = (radius * np.cos(phi) + 1) / 2
            y = (radius * np.sin(phi) + 1) / 2
            
            points[i] = [x, y]
        
        # Add small perturbations
        points += np.random.normal(0, 0.005, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_random_points(self):
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)

    def _neighborhood_move(self, points, neighbor_size=2):
        """Apply coordinated neighborhood moves."""
        new_points = points.copy()
        indices = np.random.choice(len(points), min(neighbor_size, len(points)), replace=False)
        
        # Apply coordinated movement to maintain structure
        for idx in indices:
            # Move in a way that preserves relative positions
            perturbation = np.random.normal(0, 0.005, 2)  # Smaller perturbations
            new_points[idx] += perturbation
            
        new_points = np.clip(new_points, 0, 1)
        return new_points

    def _adaptive_simulated_annealing(self, points, max_iter=2000):
        """Improved adaptive simulated annealing with better cooling schedule."""
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Better cooling schedule with adaptive elements
        temperature = 0.05
        cooling_rate = 0.9995
        min_temperature = 1e-6
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 50
        
        for iteration in range(max_iter):
            # Adapt cooling rate based on progress
            if iteration > 0 and iteration % 100 == 0:
                if len(recent_improvements) >= improvement_window:
                    avg_improvement = np.mean(recent_improvements[-improvement_window:])
                    if avg_improvement < 0.00001:
                        cooling_rate = min(cooling_rate * 0.995, 0.9997)
                    else:
                        cooling_rate = max(cooling_rate * 1.005, 0.9993)
                        
                if len(recent_improvements) > 2 * improvement_window:
                    recent_improvements = recent_improvements[-improvement_window:]

            # Generate neighbor solution
            new_points = self._neighborhood_move(current_points, neighbor_size=3)
            new_ratio = self._compute_ratio(new_points)
            
            # Track improvements
            improvement = new_ratio - current_ratio
            recent_improvements.append(improvement)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temperature):
                current_points = new_points.copy()
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()
            
            # Cool down temperature
            temperature *= cooling_rate
            
            # Early stopping
            if temperature < min_temperature:
                break

        return best_points

    def optimize(self):
        """Main optimization routine with enhanced approach."""
        best_solution = None
        best_ratio = -np.inf
        
        # Multiple diverse starting points
        start_configs = [
            self._generate_hexagonal_grid(),
            self._generate_fibonacci_spiral(),
            self._generate_random_points()
        ]
        
        for i, initial_points in enumerate(start_configs):
            try:
                # Direct optimization from initial point with SA
                optimized_points = self._adaptive_simulated_annealing(
                    initial_points, 
                    max_iter=1500
                )
                
                # Final ratio check
                final_ratio = self._compute_ratio(optimized_points)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_solution = optimized_points.copy()
                    
            except Exception as e:
                continue
                
        # Fallback to best hexagonal grid if nothing worked
        if best_solution is None:
            return self._generate_hexagonal_grid()
            
        return best_solution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = HexagonalSimulatedAnnealingOptimizer(n_points=16, dimensions=2, seed=42)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END