# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import SphericalVoronoi
import time
import math
from typing import Tuple, Optional

class PointInitializer:
    @staticmethod
    def fibonacci_sphere(n: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

class DistanceCalculator:
    @staticmethod
    def compute_min_max_ratio(points: np.ndarray) -> Tuple[float, float]:
        """Compute the ratio of minimum to maximum pairwise distances efficiently"""
        if len(points) < 2:
            return 0.0, 0.0

        # Use only non-zero distances to avoid division by zero issues
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 0:
            return 0.0, 0.0

        return d_min / d_max, d_min

class VoronoiAnalyzer:
    @staticmethod
    def analyze_constraint_regions(points: np.ndarray, threshold_ratio: float = 0.3) -> np.ndarray:
        """Analyze current configuration using spherical Voronoi to identify constraints"""
        try:
            # Create spherical Voronoi diagram with error handling
            sv = SphericalVoronoi(points)

            # Compute Voronoi cell areas and identify small cells (indicating dense regions)
            cell_areas = sv.volume
            mean_area = np.mean(cell_areas)

            # Find points in cells that are below 20% of average area (dense regions)
            dense_point_indices = np.where(cell_areas < 0.2 * mean_area)[0]

            # Also identify points that have many close neighbors using percentiles
            distances = pdist(points)
            if len(distances) == 0:
                return np.ones(len(points)) * 0.5

            distance_matrix = squareform(distances)
            percentile_10 = np.percentile(distances, 10)
            close_neighbor_counts = np.sum(distance_matrix < percentile_10, axis=1)

            # Combine criteria for constraint identification
            constraint_scores = np.zeros(len(points))

            # Score based on close neighbors
            constraint_scores += close_neighbor_counts * 0.5

            # Bonus for being in a dense region
            for idx in dense_point_indices:
                constraint_scores[idx] += 2.0

            return constraint_scores

        except Exception:
            # Fallback if Voronoi computation fails
            return np.ones(len(points)) * 0.5

class AdaptiveSimulatedAnnealing:
    def __init__(self, max_time: int = 360):
        self.max_time = max_time
        self.start_time = None
        self.iter_count = 0
        self.recent_ratios = []
        self.max_recent = 50

    def optimize(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Main optimization loop with adaptive cooling"""
        points = initial_points.copy()
        current_ratio, min_dist = DistanceCalculator.compute_min_max_ratio(points)

        # Parameters for adaptive simulated annealing
        temp = 1.0
        min_temp = 1e-10
        base_cooling_rate = 0.9995
        max_iter = 500000

        # Track best solution
        best_points = points.copy()
        best_ratio = current_ratio
        best_min_dist = min_dist

        self.start_time = time.time()
        self.iter_count = 0

        # Main optimization loop
        while temp > min_temp and self.iter_count < max_iter and time.time() - self.start_time < self.max_time:
            # Select point to perturb using Voronoi analysis
            point_to_move = self._select_point_for_perturbation(points)

            # Create new candidate point with targeted perturbation
            new_points = points.copy()
            perturbation = self._compute_targeted_perturbation(points, point_to_move)
            new_points[point_to_move] += perturbation

            # Project back onto sphere
            norm = np.linalg.norm(new_points[point_to_move])
            if norm > 0:
                new_points[point_to_move] = new_points[point_to_move] / norm

            # Compute new ratio
            new_ratio, new_min_dist = DistanceCalculator.compute_min_max_ratio(new_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                points = new_points
                current_ratio = new_ratio
                min_dist = new_min_dist

                # Update best solution if improved
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
                    best_min_dist = new_min_dist

            # Adaptive cooling based on recent performance
            self._update_adaptive_cooling(temp, base_cooling_rate, current_ratio)

            self.iter_count += 1

            # Periodic refinement
            if self.iter_count % 1000 == 0 and self.iter_count > 0:
                self._perform_refinement(points)
                # Reset temperature to allow for further exploration
                temp = min(temp * 1.1, 1.0)

        return best_points, best_ratio

    def _select_point_for_perturbation(self, points: np.ndarray) -> int:
        """Select point to perturb based on constraint analysis"""
        # Use Voronoi analysis for constraint scoring
        constraint_scores = VoronoiAnalyzer.analyze_constraint_regions(points)

        # Choose point with highest constraint score (but with some randomness)
        if np.random.random() < 0.7:
            # 70% chance to select the most constrained point
            chosen_idx = np.argmax(constraint_scores)
        else:
            # 30% chance to select randomly from top 3 constrained
            top_indices = np.argsort(constraint_scores)[-3:]
            chosen_idx = np.random.choice(top_indices)

        return chosen_idx

    def _compute_targeted_perturbation(self, points: np.ndarray, point_idx: int) -> np.ndarray:
        """Compute a targeted perturbation that should improve the configuration"""
        try:
            # Analyze local geometry around the selected point
            distances = cdist([points[point_idx]], points)[0]
            # Remove self-distance and ensure we have valid distances
            valid_distances = distances[distances > 0]

            if len(valid_distances) == 0:
                # No neighbors, random perturb with small magnitude
                return np.random.normal(0, 0.01, 3)

            avg_distance = np.mean(valid_distances)
            min_distance = np.min(valid_distances)

            # Determine perturbation type based on local density
            if min_distance < avg_distance * 0.4:
                # Point is too close to neighbors - repel it strongly
                repulsion = np.zeros(3)
                for i in range(len(points)):
                    if i != point_idx and distances[i] < avg_distance * 0.7:
                        diff = points[point_idx] - points[i]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            repulsion += diff / dist * (1.0/(dist * dist + 1e-8))

                if np.linalg.norm(repulsion) > 0:
                    repulsion = repulsion / np.linalg.norm(repulsion)
                    # Scale by how close we are to ideal spacing
                    proximity_factor = min_distance / avg_distance
                    magnitude = 0.03 * (1.0 + proximity_factor * 2.0)
                    return repulsion * magnitude
                else:
                    # Fallback to random perturbation
                    return np.random.normal(0, 0.015, 3)
            else:
                # Point is relatively well-spaced - make small adjustment
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                # Magnitude inversely proportional to distance from center
                center_dist = np.linalg.norm(points[point_idx])
                distance_factor = 1.0 - min(abs(center_dist - 0.5), 0.5)
                magnitude = 0.01 * distance_factor
                return direction * magnitude

        except Exception:
            # Fallback to simple random perturbation
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            return direction * 0.015

    def _update_adaptive_cooling(self, temp: float, base_cooling_rate: float, current_ratio: float):
        """Update cooling schedule based on recent performance"""
        self.recent_ratios.append(current_ratio)
        if len(self.recent_ratios) > self.max_recent:
            self.recent_ratios.pop(0)

        # Adjust cooling rate based on recent improvement
        cooling_rate = base_cooling_rate
        if len(self.recent_ratios) >= 10:
            recent_improvement = self.recent_ratios[-1] - self.recent_ratios[0]
            if recent_improvement < 1e-7 and temp > 1e-8:
                # Very slow progress, cool faster
                cooling_rate = base_cooling_rate * 1.15
            elif recent_improvement > 1e-4 and temp > 1e-6:
                # Fast progress, cool slower
                cooling_rate = base_cooling_rate * 0.98

        # Update temp
        return cooling_rate

    def _perform_refinement(self, points: np.ndarray):
        """Perform local refinement on current best solution"""
        # Simple local search - make small adjustments to random points
        for _ in range(50):
            point_idx = np.random.randint(0, len(points))
            # Small perturbation
            new_point = points[point_idx] + np.random.normal(0, 0.001, 3)
            norm = np.linalg.norm(new_point)
            if norm > 0:
                new_point = new_point / norm
            points[point_idx] = new_point

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    # Multi-start optimization with different initializations
    best_points = None
    best_ratio = 0.0

    # Try multiple starting configurations
    initializers = [
        PointInitializer.fibonacci_sphere,
    ]

    # Add multiple Fibonacci sphere seeds with different random perturbations
    for i in range(5):  # 5 additional Fibonacci sphere attempts
        np.random.seed(42 + i * 1000)  # Different seeds for diversity
        points = PointInitializer.fibonacci_sphere(14)
        optimizer = AdaptiveSimulatedAnnealing(max_time=360)
        optimized_points, optimized_ratio = optimizer.optimize(points)

        if optimized_ratio > best_ratio:
            best_ratio = optimized_ratio
            best_points = optimized_points.copy()

    # Run one final optimization from the best found solution to fine-tune
    if best_points is not None:
        np.random.seed(42)  # Reset seed for reproducibility
        optimizer = AdaptiveSimulatedAnnealing(max_time=60)  # Shorter time for final refinement
        final_points, final_ratio = optimizer.optimize(best_points)
        if final_ratio > best_ratio:
            best_points = final_points
            best_ratio = final_ratio

    return best_points

# EVOLVE-BLOCK-END