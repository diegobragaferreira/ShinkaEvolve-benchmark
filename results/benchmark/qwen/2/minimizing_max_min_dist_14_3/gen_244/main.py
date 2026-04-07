# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple, List, Optional, Dict, Any
import warnings
from dataclasses import dataclass, field
from enum import Enum
import math

@dataclass
class OptimizationConfig:
    """Configuration parameters for the optimization process."""
    n_points: int = 14
    n_dimensions: int = 3
    max_iterations: int = 100000
    initial_temp: float = 1.0
    final_temp: float = 1e-6
    cooling_rate: float = 0.9995
    log_interval: int = 1000
    num_seeds: int = 12
    voronoi_threshold: float = 0.1
    local_refinement_iters: int = 500
    early_stopping_patience: int = 10000
    stagnation_limit: int = 5000
    diversity_threshold: float = 0.05
    population_diversity_factor: float = 0.3

class VoronoiOptimizer:
    """Main optimizer class that orchestrates the Voronoi-based simulated annealing."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_points: Optional[np.ndarray] = None
        self.best_ratio: float = 0.0
        self.current_points: Optional[np.ndarray] = None
        self.current_ratio: float = 0.0
        self.iteration_count: int = 0
        self.start_time: float = 0.0
        self.stagnation_counter: int = 0
        self.last_improvement_iter: int = 0
        self.recent_improvements: List[int] = []

    def _fibonacci_sphere(self, n: int, seed_offset: int = 0) -> np.ndarray:
        """Generate points on a unit sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        for i in range(n):
            # Add seed offset to create variations
            y = 1 - ((i + seed_offset) / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * (i + seed_offset)  # golden angle increment with offset

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def _calculate_distances(self, points: np.ndarray) -> Tuple[float, float]:
        """Calculate minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0, 0.0

        try:
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
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms

    def _compute_voronoi_stats(self, points: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute Voronoi-based statistics to guide optimization.
        Returns: (mean_area, std_area, max_deviation_from_mean)
        """
        if len(points) < 3:
            return 0.0, 0.0, 0.0

        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points, radius=1.0)

            # Compute areas of Voronoi cells
            areas = sv.calculate_areas()

            if len(areas) == 0:
                return 0.0, 0.0, 0.0

            mean_area = np.mean(areas)
            std_area = np.std(areas)
            max_deviation = np.max(np.abs(areas - mean_area))

            return mean_area, std_area, max_deviation
        except Exception:
            return 0.0, 0.0, 0.0

    def _hybrid_perturbation(self, current_points: np.ndarray, temp: float,
                           voronoi_threshold: float = 0.1) -> np.ndarray:
        """
        Hybrid perturbation that combines Voronoi guidance with distance characteristics.
        """
        neighbor_points = current_points.copy()

        # Compute Voronoi statistics for current configuration
        mean_area, std_area, max_deviation = self._compute_voronoi_stats(current_points)

        # Calculate pairwise distances for distance-based analysis
        if len(current_points) >= 2:
            distances = pdist(current_points)
            distance_matrix = squareform(distances)
            mean_distances = np.mean(distance_matrix, axis=1)
        else:
            mean_distances = np.zeros(len(current_points))

        # Enhanced point selection using combined criteria
        try:
            sv = SphericalVoronoi(current_points, radius=1.0)
            areas = sv.calculate_areas()

            # Create composite weights combining Voronoi imbalance and distance characteristics
            if len(areas) > 0:
                # Voronoi-based weights (emphasize imbalanced cells)
                voronoi_deviations = np.abs(areas - mean_area)
                voronoi_weights = np.exp(-voronoi_deviations / (std_area + 1e-8))

                # Distance-based weights (points with extreme mean distances)
                median_mean_dist = np.median(mean_distances)
                if median_mean_dist > 0:
                    distance_weights = np.exp(-np.abs(mean_distances - median_mean_dist) / (median_mean_dist + 1e-8))
                else:
                    distance_weights = np.ones(len(mean_distances))

                # Combine weights with adaptive blending
                # Higher weight on Voronoi for early iterations, more on distance later
                blend_factor = min(1.0, self.iteration_count / (self.config.max_iterations * 0.3))
                combined_weights = (1 - blend_factor) * voronoi_weights + blend_factor * distance_weights

                # Normalize weights
                combined_weights = combined_weights / np.sum(combined_weights)
                idx = np.random.choice(len(current_points), p=combined_weights)
            else:
                # Fallback to random selection if Voronoi computation fails
                idx = np.random.randint(0, len(current_points))
        except Exception:
            # Fall back to random selection if any computation fails
            idx = np.random.randint(0, len(current_points))

        # Determine perturbation magnitude with enhanced logic:
        # 1. Temperature-based scaling
        # 2. Adaptive scaling based on optimization stage
        # 3. Context-aware scaling based on Voronoi imbalance and distance characteristics
        base_perturbation = temp * 0.05

        # Stage-dependent adjustment (more aggressive early on)
        stage_factor = min(1.0, self.iteration_count / (self.config.max_iterations * 0.5))
        stage_adjustment = 0.5 + 0.5 * stage_factor  # More aggressive in early stages

        # Quality-dependent adjustment (smaller changes when we're doing well)
        quality_factor = min(1.0, self.current_ratio / 0.3)  # Normalize by expected good range
        quality_adjustment = 0.5 + 0.5 * quality_factor  # Smaller perturbations for good solutions

        # Imbalance-dependent adjustment
        imbalance_factor = min(1.0, max_deviation / (mean_area + 1e-8))
        imbalance_adjustment = 1.0 + 0.5 * imbalance_factor  # More aggressive for imbalanced configs

        # Total adjustment factor
        adjustment_factor = stage_adjustment * quality_adjustment * imbalance_adjustment
        perturbation_mag = base_perturbation * adjustment_factor

        # Further adjust based on cell type and distance characteristics
        if len(areas) > 0:
            area_ratio = areas[idx] / (mean_area + 1e-8)
            mean_dist_ratio = mean_distances[idx] / (np.mean(mean_distances) + 1e-8)

            # If point is in a particularly large or small cell OR far from neighbors, perturb more aggressively
            if area_ratio > 2.0 or area_ratio < 0.5 or mean_dist_ratio > 1.5:
                perturbation_mag *= 2.0
            elif area_ratio > 1.5 or area_ratio < 0.67 or mean_dist_ratio > 1.2:
                perturbation_mag *= 1.5

        # Add Gaussian noise to selected point
        noise = np.random.normal(0, perturbation_mag, 3)
        neighbor_points[idx] += noise

        # Project back to sphere
        neighbor_points = self._project_to_sphere(neighbor_points)

        return neighbor_points

    def _voronoi_guided_perturbation(self, current_points: np.ndarray, temp: float,
                                   voronoi_threshold: float = 0.1) -> np.ndarray:
        """
        Generate neighbor using Voronoi-based guidance to improve point distribution.
        """
        # Use the enhanced hybrid perturbation method
        return self._hybrid_perturbation(current_points, temp, voronoi_threshold)

    def _adaptive_local_refinement(self, starting_points: np.ndarray,
                                 max_iters: int = 500) -> Tuple[np.ndarray, float]:
        """Apply local refinement focused on improving specific regions."""
        current_points = starting_points.copy()
        current_ratio = self._calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Gradient-free local search approach
        for iter_num in range(max_iters):
            # Get Voronoi stats to determine optimization focus
            mean_area, std_area, max_deviation = self._compute_voronoi_stats(current_points)

            # Use Voronoi-guided perturbation with higher focus on problematic regions
            candidate_points = self._voronoi_guided_perturbation(
                current_points,
                temp=0.01,  # Low temperature for local refinement
                voronoi_threshold=0.05
            )

            candidate_ratio = self._calculate_ratio(candidate_points)

            # Always accept improvement (gradient-free local search)
            if candidate_ratio > current_ratio:
                current_points = candidate_points
                current_ratio = candidate_ratio

                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio

        return best_points, best_ratio

    def _adaptive_cooling_schedule(self, temp: float, improvement: bool) -> float:
        """Apply adaptive cooling schedule based on recent improvements."""
        if len(self.recent_improvements) < 3:
            # Use standard cooling if few improvements
            return temp * self.config.cooling_rate
        elif len(self.recent_improvements) >= 8:
            # Faster cooling for frequent improvements
            return temp * (self.config.cooling_rate * 0.95)
        else:
            # Moderate cooling
            return temp * self.config.cooling_rate

    def _initialize_population(self) -> None:
        """Initialize with multiple Fibonacci sphere seeds and diversify the population."""
        # Collect multiple diverse initial configurations
        initial_configurations = []
        best_seed_ratio = 0.0
        best_seed_points = None

        # Generate multiple seeds with different strategies for better diversity
        seed_offsets = []
        for i in range(self.config.num_seeds):
            # Mix of regular Fibonacci offsets and randomized offsets for diversity
            if i < self.config.num_seeds // 2:
                seed_offsets.append(i)
            else:
                # Add some randomness for diversity
                seed_offsets.append(int(np.random.uniform(0, 10000)))

        for seed_offset in seed_offsets:
            # Initialize with Fibonacci sphere configuration
            seed_points = self._fibonacci_sphere(self.config.n_points, seed_offset)
            seed_ratio = self._calculate_ratio(seed_points)

            initial_configurations.append((seed_points.copy(), seed_ratio))

            if seed_ratio > best_seed_ratio:
                best_seed_ratio = seed_ratio
                best_seed_points = seed_points.copy()

        # Keep top configurations based on diversity and quality
        if len(initial_configurations) > 1:
            # Sort by ratio
            initial_configurations.sort(key=lambda x: x[1], reverse=True)

            # Select diverse configurations based on distance metrics
            selected_configs = [initial_configurations[0]]  # Always keep the best

            # Select additional diverse configurations
            for i in range(1, min(len(initial_configurations), max(1, int(self.config.num_seeds * self.config.population_diversity_factor)))):
                current_config = initial_configurations[i][0]

                # Check diversity with already selected configs
                is_diverse = True
                for selected_config in selected_configs:
                    # Compute average distance between points in two configurations
                    avg_distance = np.mean([np.linalg.norm(a-b) for a in current_config for b in selected_config])
                    if avg_distance < self.config.diversity_threshold:
                        is_diverse = False
                        break

                if is_diverse:
                    selected_configs.append(initial_configurations[i])

                # Stop if we have enough diverse configurations
                if len(selected_configs) >= int(self.config.num_seeds * 0.6):
                    break

            # Use the best configuration among selected as primary
            best_selected = max(selected_configs, key=lambda x: x[1])
            self.current_points = best_selected[0]
            self.current_ratio = best_selected[1]
        else:
            self.current_points = best_seed_points
            self.current_ratio = best_seed_ratio

        # Use best seed configuration as starting point
        self.best_points = self.current_points.copy()
        self.best_ratio = self.current_ratio
        self.iteration_count = 0
        self.start_time = time.time()
        self.stagnation_counter = 0
        self.recent_improvements = []
        self.last_improvement_iter = 0

    def _optimize_single_iteration(self, temp: float) -> Tuple[np.ndarray, float, bool]:
        """Perform one optimization iteration."""
        # Generate candidate solution using Voronoi-guided perturbation
        candidate_points = self._voronoi_guided_perturbation(
            self.current_points,
            temp=temp,
            voronoi_threshold=self.config.voronoi_threshold
        )

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
                self.stagnation_counter = 0
                self.last_improvement_iter = self.iteration_count
            else:
                self.stagnation_counter += 1
        else:
            self.stagnation_counter += 1

        return self.current_points, self.current_ratio, accepted

    def _run_optimization_loop(self) -> Tuple[np.ndarray, float]:
        """Run the main optimization loop with early stopping."""
        temp = self.config.initial_temp

        # Main optimization loop
        while self.iteration_count < self.config.max_iterations and temp > self.config.final_temp:
            # Perform single iteration
            _, _, accepted = self._optimize_single_iteration(temp)

            # Apply adaptive cooling
            if accepted or self.stagnation_counter < 1000:  # Only cool if there was improvement or not too stale
                temp = self._adaptive_cooling_schedule(temp, accepted)

            self.iteration_count += 1

            # Check for stagnation and early stopping
            if self.iteration_count - self.last_improvement_iter > self.config.early_stopping_patience:
                break

            # Check for stagnation and apply additional cooling
            if self.stagnation_counter > self.config.stagnation_limit:
                temp = max(temp * 0.95, self.config.final_temp)
                self.stagnation_counter = 0

            # Periodic local refinement to escape local minima
            if self.iteration_count % 2000 == 0 and self.iteration_count > 0:
                refined_points, refined_ratio = self._adaptive_local_refinement(
                    self.best_points, self.config.local_refinement_iters
                )

                if refined_ratio > self.best_ratio:
                    self.best_points = refined_points
                    self.best_ratio = refined_ratio

        return self.best_points, self.best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Uses a spherical Voronoi-based evolution approach that combines geometric insights with
    adaptive optimization strategies to improve upon traditional simulated annealing methods.

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
        log_interval=1000,
        num_seeds=8,
        voronoi_threshold=0.1,
        local_refinement_iters=500,
        early_stopping_patience=10000,
        stagnation_limit=5000
    )

    try:
        # Initialize optimizer
        optimizer = VoronoiOptimizer(config)

        # Initialize population with multiple seeds
        optimizer._initialize_population()

        # Run optimization
        optimized_points, best_ratio = optimizer._run_optimization_loop()

        # Final validation
        if optimized_points is not None:
            final_min, final_max = optimizer._calculate_distances(optimized_points)
            if final_max <= 0:
                warnings.warn("Final validation failed, returning Fibonacci sphere initialization")
                optimized_points = optimizer._fibonacci_sphere(14)
        else:
            # Fallback to Fibonacci initialization if optimization failed
            warnings.warn("Optimization returned None, using Fibonacci sphere initialization")
            optimized_points = optimizer._fibonacci_sphere(14)

        return optimized_points

    except Exception as e:
        # Fallback to basic initialization if anything fails
        warnings.warn(f"Optimization failed with error: {str(e)}, returning Fibonacci sphere initialization")
        return VoronoiOptimizer(config)._fibonacci_sphere(14)

# EVOLVE-BLOCK-END