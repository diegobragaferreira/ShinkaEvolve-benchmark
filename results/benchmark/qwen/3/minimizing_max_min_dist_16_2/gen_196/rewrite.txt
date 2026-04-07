# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time
from typing import Tuple, List, Optional
import warnings


class DistanceCalculator:
    """Efficiently computes pairwise distances and ratios."""
    
    @staticmethod
    def calculate_distances(points: np.ndarray) -> Tuple[float, float]:
        """Calculate minimum and maximum pairwise distances."""
        if len(points) < 2:
            return 0.0, 0.0
            
        try:
            distances = pdist(points)
            
            if len(distances) == 0:
                return 0.0, 0.0
                
            return float(np.min(distances)), float(np.max(distances))
        except Exception:
            return 0.0, 0.0
    
    @staticmethod
    def evaluate_ratio(points: np.ndarray) -> float:
        """Evaluate the min/max distance ratio with safety checks."""
        min_d, max_d = DistanceCalculator.calculate_distances(points)
        
        if max_d <= 1e-12:
            return 0.0
            
        return min_d / max_d


class InitializationStrategy:
    """Base class for point initialization strategies."""
    
    def __init__(self, n_points: int, dimension: int, seed: int):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)
    
    def initialize(self) -> np.ndarray:
        raise NotImplementedError


class HexagonalInitialization(InitializationStrategy):
    """Initialize points in a hexagonal grid pattern."""
    
    def initialize(self) -> np.ndarray:
        """Create optimized hexagonal packing with symmetry breaking."""
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2
        col_spacing = 1.0

        points = []

        # Create hexagonal lattice pattern
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                x = j * col_spacing + (i % 2) * 0.5
                y = i * row_spacing
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points[:self.n_points])

        # Normalize to [0,1] bounds
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Apply structured perturbations to break symmetry
        for i in range(self.n_points):
            noise_intensity = 0.01 + 0.005 * math.sin(i * 0.785398)
            noise_x = np.random.normal(0, noise_intensity, 1)[0]
            noise_y = np.random.normal(0, noise_intensity, 1)[0]
            points[i] += [noise_x, noise_y]

        # Clip to ensure bounds
        points = np.clip(points, 0, 1)
        return points


class GridInitialization(InitializationStrategy):
    """Initialize points in a regular grid pattern."""
    
    def initialize(self) -> np.ndarray:
        """Create grid-based initial points."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        return points


class RandomInitialization(InitializationStrategy):
    """Initialize points randomly."""
    
    def initialize(self) -> np.ndarray:
        """Create random initial points."""
        return np.random.rand(self.n_points, self.dimension)


class PerturbationEngine:
    """Handles point perturbations for optimization."""
    
    @staticmethod
    def single_perturb(points: np.ndarray, idx: int, step_size: float = 0.005) -> np.ndarray:
        """Perturb a single point with boundary handling."""
        new_points = points.copy()
        delta = np.random.uniform(-step_size, step_size, points.shape[1])
        new_points[idx] = points[idx] + delta
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points
    
    @staticmethod
    def neighborhood_perturb(points: np.ndarray, indices: List[int], step_size: float = 0.005) -> np.ndarray:
        """Perturb a group of points together while preserving structure."""
        new_points = points.copy()
        centroid = np.mean(points[indices], axis=0)
        
        for idx in indices:
            delta = np.random.uniform(-step_size, step_size, points.shape[1])
            new_points[idx] = points[idx] + delta
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
        return new_points


class AdaptiveSimulatedAnnealing:
    """Enhanced adaptive simulated annealing optimizer."""
    
    def __init__(self, max_iterations: int = 5000, seed: int = 42):
        self.max_iterations = max_iterations
        self.seed = seed
        np.random.seed(seed)
    
    def optimize(self, initial_points: np.ndarray, evaluate_fn) -> np.ndarray:
        """Run adaptive simulated annealing optimization."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = evaluate_fn(current_points)

        temperature = 1.0
        cooling_rate = 0.9995
        stagnation_counter = 0
        previous_best = best_ratio
        phase = 0
        phase_thresholds = [1000, 3000]

        for iteration in range(self.max_iterations):
            # Determine perturbation type
            use_neighborhood = np.random.random() < 0.7
            
            if use_neighborhood:
                # Neighborhood perturbation
                if iteration < 1000:
                    neighborhood_size = 2
                elif iteration < 3000:
                    neighborhood_size = np.random.randint(2, 4)
                else:
                    neighborhood_size = np.random.randint(2, min(5, len(current_points)))
                    
                indices = np.random.choice(len(current_points), neighborhood_size, replace=False).tolist()
                neighbor_points = PerturbationEngine.neighborhood_perturb(
                    current_points, indices, step_size=temperature * 0.05
                )
            else:
                # Single point perturbation
                point_idx = np.random.randint(0, len(current_points))
                neighbor_points = PerturbationEngine.single_perturb(
                    current_points, point_idx, step_size=temperature * 0.05
                )
            
            # Evaluate neighbor solution
            neighbor_ratio = evaluate_fn(neighbor_points)
            
            # Accept/reject based on Metropolis criterion
            if neighbor_ratio > best_ratio:
                current_points = neighbor_points
                best_points = neighbor_points
                best_ratio = neighbor_ratio
                stagnation_counter = 0
            elif np.random.rand() < math.exp((neighbor_ratio - best_ratio) / temperature):
                current_points = neighbor_points
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adaptive cooling logic
            if stagnation_counter > 50:
                temperature *= 0.995
                stagnation_counter = 0
            else:
                phase_cooling = cooling_rate * (0.95 if phase > 0 else 1.0)
                temperature *= phase_cooling
            
            # Phase transitions
            if iteration in phase_thresholds:
                phase += 1
            
            # Early stopping conditions
            if iteration % 100 == 0 and iteration > 0:
                current_ratio = evaluate_fn(best_points)
                if abs(previous_best - current_ratio) < 1e-8:
                    break
                previous_best = current_ratio

            # Prevent infinite loops
            if iteration > 1000 and temperature < 0.001:
                break
                
        return best_points


class PointDispersionOptimizer:
    """Main optimizer class that orchestrates the complete process."""
    
    def __init__(self, n_points: int = 16, dimension: int = 2, seed: int = 42):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        self.initializations = [
            HexagonalInitialization(n_points, dimension, seed),
            GridInitialization(n_points, dimension, seed),
            RandomInitialization(n_points, dimension, seed)
        ]
        self.annealer = AdaptiveSimulatedAnnealing(max_iterations=5000, seed=seed)
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        best_points = None
        best_ratio = -float('inf')
        
        # Try multiple initialization strategies
        for i, initializer in enumerate(self.initializations):
            try:
                initial_points = initializer.initialize()
                optimized_points = self.annealer.optimize(initial_points, DistanceCalculator.evaluate_ratio)
                current_ratio = DistanceCalculator.evaluate_ratio(optimized_points)
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Optimization from start {i} failed: {e}")
                continue
        
        # Return best result or fallback
        return best_points if best_points is not None else self.initializations[0].initialize()


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDispersionOptimizer(n_points=16, dimension=2, seed=42)
    return optimizer.optimize()


# EVOLVE-BLOCK-END