# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from typing import Tuple, List, Optional
import warnings
from dataclasses import dataclass
from enum import Enum

@dataclass
class OptimizationConfig:
    """Configuration parameters for the optimization process."""
    n_points: int = 14
    n_dimensions: int = 3
    max_iterations: int = 100000
    initial_temp: float = 1.0
    final_temp: float = 1e-6
    cooling_rate: float = 0.9995
    perturbation_magnitude: float = 0.1
    log_interval: int = 1000
    min_improvement_threshold: float = 1e-6
    stagnation_limit: int = 5000

class OptimizationState(Enum):
    """Enumeration for optimization process states."""
    INITIAL = "initial"
    OPTIMIZING = "optimizing"
    CONVERGED = "converged"
    FAILED = "failed"

class PointDistributionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_points: Optional[np.ndarray] = None
        self.best_ratio: float = 0.0
        self.current_points: Optional[np.ndarray] = None
        self.current_ratio: float = 0.0
        self.iteration_count: int = 0
        self.start_time: float = 0.0
        self.stagnation_counter: int = 0
        self.recent_improvements: List[int] = []
        self.state: OptimizationState = OptimizationState.INITIAL

    def _fibonacci_sphere(self, n: int) -> np.ndarray:
        """Generate points on a unit sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def _calculate_distances(self, points: np.ndarray) -> Tuple[float, float]:
        """Calculate minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0, 0.0

        try:
            # Use pdist for efficient computation
            distances = pdist(points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            return d_min, d_max
        except Exception:
            return 0.0, 0.0

    def _calculate_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distances."""
        d_min, d_max = self._calculate_distances(points)
        if d_max <= 0:
            return 0.0
        return d_min / d_max

    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms

    def _generate_neighbor(self, current_points: np.ndarray, temp: float) -> np.ndarray:
        """Generate a neighboring solution by perturbing a random point."""
        neighbor_points = current_points.copy()

        # Choose a random point to perturb
        idx = np.random.randint(0, len(current_points))

        # Determine perturbation magnitude based on temperature
        perturbation_mag = temp * 0.1

        # Add Gaussian noise to selected point
        noise = np.random.normal(0, perturbation_mag, self.config.n_dimensions)
        neighbor_points[idx] += noise

        # Project back to sphere
        neighbor_points = self._project_to_sphere(neighbor_points)

        return neighbor_points

    def _adaptive_cooling(self, temp: float, improvement: bool) -> float:
        """Apply adaptive cooling schedule based on recent improvements."""
        if len(self.recent_improvements) < 5:
            # If few improvements, use standard cooling
            return temp * self.config.cooling_rate
        elif len(self.recent_improvements) >= 10:
            # If many recent improvements, use faster cooling
            return temp * (self.config.cooling_rate * 0.9)
        else:
            # Moderate cooling
            return temp * self.config.cooling_rate

    def _initialize_optimization(self) -> None:
        """Initialize the optimization process."""
        # Start with Fibonacci sphere initialization
        self.current_points = self._fibonacci_sphere(self.config.n_points)
        self.current_ratio = self._calculate_ratio(self.current_points)
        self.best_points = self.current_points.copy()
        self.best_ratio = self.current_ratio
        self.iteration_count = 0
        self.start_time = time.time()
        self.stagnation_counter = 0
        self.recent_improvements = []
        self.state = OptimizationState.OPTIMIZING

    def _perform_single_iteration(self, temp: float) -> Tuple[np.ndarray, float, bool]:
        """Perform one optimization iteration."""
        # Generate candidate solution
        candidate_points = self._generate_neighbor(self.current_points, temp)
        candidate_ratio = self._calculate_ratio(candidate_points)

        # Accept or reject based on Metropolis criterion
        delta_ratio = candidate_ratio - self.current_ratio

        # Avoid numerical issues with very small temperature
        if temp < 1e-12:
            accept_prob = 1.0 if delta_ratio > 0 else 0.0
        else:
            accept_prob = min(1.0, np.exp(delta_ratio / temp))

        accepted = False

        if np.random.random() < accept_prob:
            self.current_points = candidate_points
            self.current_ratio = candidate_ratio
            accepted = True

            # Update best solution
            if self.current_ratio > self.best_ratio:
                self.best_points = self.current_points.copy()
                self.best_ratio = self.current_ratio
                self.recent_improvements.append(self.iteration_count)
                if len(self.recent_improvements) > 10:
                    self.recent_improvements.pop(0)
                self.stagnation_counter = 0  # Reset stagnation counter
            else:
                self.stagnation_counter += 1
        else:
            self.stagnation_counter += 1

        return self.current_points, self.current_ratio, accepted

    def _run_optimization_loop(self) -> Tuple[np.ndarray, float]:
        """Run the main optimization loop."""
        temp = self.config.initial_temp

        # Main optimization loop
        while self.iteration_count < self.config.max_iterations and temp > self.config.final_temp:
            # Perform single iteration
            _, _, accepted = self._perform_single_iteration(temp)

            # Apply adaptive cooling
            if accepted or self.stagnation_counter < 500:  # Only cool if there was improvement or not too stale
                temp = self._adaptive_cooling(temp, accepted)

            self.iteration_count += 1

            # Check for stagnation
            if self.stagnation_counter > self.config.stagnation_limit:
                temp = max(temp * 0.95, self.config.final_temp)
                self.stagnation_counter = 0

            # Log progress periodically
            if self.iteration_count % self.config.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                pass  # Could add more sophisticated logging here

        return self.best_points, self.best_ratio

    def optimize(self) -> Tuple[np.ndarray, float]:
        """Execute the optimization process."""
        try:
            self._initialize_optimization()
            best_points, best_ratio = self._run_optimization_loop()
            return best_points, best_ratio
        except Exception as e:
            warnings.warn(f"Optimization failed with error: {str(e)}")
            return self._fibonacci_sphere(self.config.n_points), 0.0

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Uses a spherical simulated annealing approach with adaptive cooling and strategic perturbations.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Create configuration
    config = OptimizationConfig(
        n_points=14,
        n_dimensions=3,
        max_iterations=100000,
        initial_temp=1.0,
        final_temp=1e-6,
        cooling_rate=0.9995,
        perturbation_magnitude=0.1,
        log_interval=1000
    )

    # Initialize optimizer
    optimizer = PointDistributionOptimizer(config)

    # Run optimization
    try:
        optimized_points, best_ratio = optimizer.optimize()

        # Final validation
        final_min, final_max = optimizer._calculate_distances(optimized_points)
        if final_max <= 0:
            # Fallback to Fibonacci initialization if optimization failed
            warnings.warn("Optimization failed, returning Fibonacci sphere initialization")
            optimized_points = optimizer._fibonacci_sphere(14)

        return optimized_points

    except Exception as e:
        # Fallback to basic initialization if anything fails
        warnings.warn(f"Optimization failed with error: {str(e)}, returning Fibonacci sphere initialization")
        return optimizer._fibonacci_sphere(14)

# EVOLVE-BLOCK-END