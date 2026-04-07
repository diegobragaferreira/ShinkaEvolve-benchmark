# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple, List, Optional
import warnings
from dataclasses import dataclass
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
    num_seeds: int = 8  # More diverse seeds for better exploration
    voronoi_threshold: float = 0.1  # Threshold for Voronoi-based perturbation
    local_refinement_iters: int = 500  # Local refinement iterations

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
        local_refinement_iters=500
    )

    def fibonacci_sphere(n: int, seed_offset: int = 0) -> np.ndarray:
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

    def calculate_distances(points: np.ndarray) -> Tuple[float, float]:
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

    def calculate_ratio(points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distances."""
        d_min, d_max = calculate_distances(points)
        if d_max <= 0:
            return 0.0
        return d_min / d_max

    def project_to_sphere(points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms

    def compute_voronoi_stats(points: np.ndarray) -> Tuple[float, float, float]:
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

    def voronoi_guided_perturbation(current_points: np.ndarray, temp: float,
                                  voronoi_threshold: float = 0.1) -> np.ndarray:
        """
        Generate neighbor using Voronoi-based guidance to improve point distribution.
        Uses area-weighted selection and adaptive perturbation magnitudes.
        """
        neighbor_points = current_points.copy()

        # Compute Voronoi statistics for current configuration
        mean_area, std_area, max_deviation = compute_voronoi_stats(current_points)

        # Calculate weighted probabilities based on Voronoi cell sizes
        # Points in larger cells (underpopulated regions) get higher probability
        try:
            sv = SphericalVoronoi(current_points, radius=1.0)
            areas = sv.calculate_areas()

            # Create probability weights (higher weight for larger cells)
            # Use softmax to convert areas to probabilities
            if len(areas) > 0:
                # Normalize areas to be comparable (subtract mean, divide by std)
                normalized_areas = (areas - np.mean(areas)) / (np.std(areas) + 1e-8)
                # Apply softmax to convert to probabilities
                exp_areas = np.exp(normalized_areas * 2)  # Scale up influence
                weights = exp_areas / np.sum(exp_areas)

                # Select point using weighted probability
                idx = np.random.choice(len(current_points), p=weights)
            else:
                # Fall back to random selection if Voronoi computation fails
                idx = np.random.randint(0, len(current_points))

        except Exception:
            # Fall back to random selection if Voronoi computation fails
            idx = np.random.randint(0, len(current_points))

        # Determine perturbation magnitude based on:
        # 1. Temperature
        # 2. Voronoi deviation (larger deviations = more aggressive perturbation)
        # 3. Whether point is in extreme cell (large or small area)
        base_perturbation = temp * 0.05

        # Increase perturbation magnitude for extreme cases
        if max_deviation > voronoi_threshold:
            # More aggressive perturbation when there's significant non-uniformity
            perturbation_mag = base_perturbation * (1.0 + max_deviation / (mean_area + 1e-8))
        else:
            perturbation_mag = base_perturbation

        # Further adjust based on cell size
        if len(areas) > 0:
            area_ratio = areas[idx] / (mean_area + 1e-8)
            # If point is in a particularly large cell, perturb more aggressively
            if area_ratio > 1.5:
                perturbation_mag *= 1.5

        # Add Gaussian noise to selected point
        noise = np.random.normal(0, perturbation_mag, 3)
        neighbor_points[idx] += noise

        # Project back to sphere
        neighbor_points = project_to_sphere(neighbor_points)

        return neighbor_points

    def adaptive_local_refinement(starting_points: np.ndarray,
                                max_iters: int = 500) -> Tuple[np.ndarray, float]:
        """Apply local refinement focused on improving specific regions."""
        current_points = starting_points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Gradient-free local search approach
        for iter_num in range(max_iters):
            # Get Voronoi stats to determine optimization focus
            mean_area, std_area, max_deviation = compute_voronoi_stats(current_points)

            # Use Voronoi-guided perturbation with higher focus on problematic regions
            candidate_points = voronoi_guided_perturbation(
                current_points,
                temp=0.01,  # Low temperature for local refinement
                voronoi_threshold=0.05
            )

            candidate_ratio = calculate_ratio(candidate_points)

            # Always accept improvement (gradient-free local search)
            if candidate_ratio > current_ratio:
                current_points = candidate_points
                current_ratio = candidate_ratio

                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio

        return best_points, best_ratio

    def hybrid_optimization() -> Tuple[np.ndarray, float]:
        """Main hybrid optimization routine combining multiple strategies."""
        # Try multiple Fibonacci seeds for diverse starting points
        best_points_overall = None
        best_ratio_overall = 0.0

        for seed_offset in range(config.num_seeds):
            # Initialize with Fibonacci sphere configuration
            current_points = fibonacci_sphere(config.n_points, seed_offset)
            current_ratio = calculate_ratio(current_points)

            # Store best from this seed
            seed_best_points = current_points.copy()
            seed_best_ratio = current_ratio

            # Main optimization loop with adaptive temperature
            temp = config.initial_temp
            iteration = 0
            stagnant_count = 0
            last_improvement_iter = 0

            while iteration < config.max_iterations and temp > config.final_temp:
                # Voronoi-guided perturbation
                candidate_points = voronoi_guided_perturbation(
                    current_points,
                    temp=temp,
                    voronoi_threshold=config.voronoi_threshold
                )

                candidate_ratio = calculate_ratio(candidate_points)

                # Metropolis acceptance criterion with adaptive temperature
                delta_ratio = candidate_ratio - current_ratio

                # Avoid numerical issues
                if temp < 1e-12:
                    accept_prob = 1.0 if delta_ratio > 0 else 0.0
                else:
                    accept_prob = min(1.0, np.exp(delta_ratio / temp))

                if np.random.random() < accept_prob:
                    current_points = candidate_points
                    current_ratio = candidate_ratio

                    if current_ratio > seed_best_ratio:
                        seed_best_points = current_points.copy()
                        seed_best_ratio = current_ratio
                        last_improvement_iter = iteration

                    stagnant_count = 0
                else:
                    stagnant_count += 1

                # Adaptive cooling based on progress
                if stagnant_count > 500:
                    temp = max(temp * 0.95, config.final_temp)
                    stagnant_count = 0
                else:
                    temp *= config.cooling_rate

                iteration += 1

                # Periodic local refinement to escape local minima
                if iteration % 2000 == 0 and iteration > 0:
                    refined_points, refined_ratio = adaptive_local_refinement(
                        seed_best_points, config.local_refinement_iters
                    )

                    if refined_ratio > seed_best_ratio:
                        seed_best_points = refined_points
                        seed_best_ratio = refined_ratio

            # Update overall best
            if seed_best_ratio > best_ratio_overall:
                best_ratio_overall = seed_best_ratio
                best_points_overall = seed_best_points.copy()

        # Final local refinement on the best configuration
        if best_points_overall is not None:
            final_points, final_ratio = adaptive_local_refinement(
                best_points_overall, config.local_refinement_iters * 2
            )
            return final_points, final_ratio

        return best_points_overall, best_ratio_overall

    try:
        # Run hybrid optimization
        optimized_points, best_ratio = hybrid_optimization()

        # Final validation
        if optimized_points is not None:
            final_min, final_max = calculate_distances(optimized_points)
            if final_max <= 0:
                warnings.warn("Final validation failed, returning Fibonacci sphere initialization")
                optimized_points = fibonacci_sphere(14)
        else:
            # Fallback to Fibonacci initialization if optimization failed
            warnings.warn("Optimization returned None, using Fibonacci sphere initialization")
            optimized_points = fibonacci_sphere(14)

        return optimized_points

    except Exception as e:
        # Fallback to basic initialization if anything fails
        warnings.warn(f"Optimization failed with error: {str(e)}, returning Fibonacci sphere initialization")
        return fibonacci_sphere(14)

# EVOLVE-BLOCK-END