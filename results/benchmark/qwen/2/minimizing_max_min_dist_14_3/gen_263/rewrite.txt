# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import SphericalVoronoi
from scipy.special import sph_harm
import time
import math
from typing import Tuple, Optional
from sklearn.cluster import KMeans

class SphericalHarmonicAnalyzer:
    """Analyzes point distributions using spherical harmonics for regularity assessment"""
    
    @staticmethod
    def compute_spherical_harmonics(points: np.ndarray, l: int = 2, m: int = 0) -> float:
        """Compute spherical harmonic component for distribution analysis"""
        if len(points) == 0:
            return 0.0
            
        # Convert cartesian to spherical coordinates
        theta = np.arccos(points[:, 2])  # polar angle
        phi = np.arctan2(points[:, 1], points[:, 0])  # azimuthal angle
        
        # Normalize angles to [0, 2π]
        phi = np.where(phi < 0, phi + 2*np.pi, phi)
        
        # Compute spherical harmonic
        try:
            Y_lm = sph_harm(m, l, phi, theta)
            return np.real(np.mean(Y_lm))
        except:
            return 0.0

class DistributedOptimizationEngine:
    """Main optimization engine implementing distributed strategies"""
    
    def __init__(self, max_time: int = 360):
        self.max_time = max_time
        self.start_time = None
        self.iter_count = 0
        self.recent_ratios = []
        self.max_recent = 100
        
    def optimize(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Main optimization loop with distributed strategies"""
        points = initial_points.copy()
        current_ratio, min_dist = self._compute_min_max_ratio(points)

        # Parameters
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

        # Strategy weights for different optimization modes
        strategy_weights = {
            'voronoi': 0.3,
            'neighborhood': 0.4,
            'harmonic': 0.2,
            'random': 0.1
        }

        while temp > min_temp and self.iter_count < max_iter and time.time() - self.start_time < self.max_time:
            # Select optimization strategy based on current progress
            strategy = self._select_strategy(strategy_weights, current_ratio)
            
            # Create new candidate point
            new_points = points.copy()
            
            # Apply strategy-specific perturbation
            if strategy == 'voronoi':
                point_to_move = self._select_voronoi_based_point(points)
                perturbation = self._compute_voronoi_guided_perturbation(points, point_to_move)
            elif strategy == 'neighborhood':
                point_to_move = self._select_neighborhood_based_point(points)
                perturbation = self._compute_neighborhood_guided_perturbation(points, point_to_move)
            elif strategy == 'harmonic':
                point_to_move = self._select_harmonic_based_point(points)
                perturbation = self._compute_harmonic_guided_perturbation(points, point_to_move)
            else:  # random
                point_to_move = np.random.randint(0, len(points))
                perturbation = self._compute_random_perturbation(points, point_to_move)
            
            new_points[point_to_move] += perturbation
            
            # Project back onto sphere
            norm = np.linalg.norm(new_points[point_to_move])
            if norm > 0:
                new_points[point_to_move] = new_points[point_to_move] / norm

            # Compute new ratio
            new_ratio, new_min_dist = self._compute_min_max_ratio(new_points)

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

            # Adaptive cooling with dynamic adjustment based on geometric properties
            cooling_rate = self._update_adaptive_cooling(temp, base_cooling_rate, current_ratio, points)
            temp *= cooling_rate

            self.iter_count += 1

            # Periodic refinement and strategy adjustment
            if self.iter_count % 500 == 0 and self.iter_count > 0:
                self._perform_periodic_refinement(points)
                # Occasionally rebalance strategy weights
                if self.iter_count % 5000 == 0:
                    strategy_weights = self._rebalance_strategy_weights(strategy_weights, best_ratio)

        return best_points, best_ratio

    def _select_strategy(self, weights: dict, current_ratio: float) -> str:
        """Select optimization strategy based on current configuration state"""
        # Bias towards strategies that have been underperforming recently
        if len(self.recent_ratios) > 10:
            recent_avg = np.mean(self.recent_ratios[-10:])
            if current_ratio < recent_avg * 0.95:  # Significant drop in ratio
                # Increase weights for more aggressive strategies
                adjusted_weights = weights.copy()
                adjusted_weights['voronoi'] = min(0.5, weights['voronoi'] + 0.1)
                adjusted_weights['neighborhood'] = min(0.6, weights['neighborhood'] + 0.1)
                return np.random.choice(list(adjusted_weights.keys()), p=list(adjusted_weights.values()))
        
        return np.random.choice(list(weights.keys()), p=list(weights.values()))

    def _select_voronoi_based_point(self, points: np.ndarray) -> int:
        """Select point based on Voronoi analysis"""
        try:
            sv = SphericalVoronoi(points)
            cell_areas = sv.volume
            mean_area = np.mean(cell_areas)
            
            # Points in small cells are likely clustered
            dense_point_indices = np.where(cell_areas < 0.15 * mean_area)[0]
            
            if len(dense_point_indices) > 0 and np.random.random() < 0.7:
                return np.random.choice(dense_point_indices)
                
        except:
            pass
            
        return np.random.randint(0, len(points))

    def _compute_voronoi_guided_perturbation(self, points: np.ndarray, point_idx: int) -> np.ndarray:
        """Compute perturbation guided by Voronoi analysis"""
        distances = cdist([points[point_idx]], points)[0]
        distances = distances[distances > 0]  # Remove self-distance
        
        if len(distances) == 0:
            return np.random.normal(0, 0.01, 3)
            
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        
        # If too close, repel strongly
        if min_dist < avg_dist * 0.35:
            repulsion = np.zeros(3)
            for i in range(len(points)):
                if i != point_idx and distances[i-1] < avg_dist * 0.6:
                    diff = points[point_idx] - points[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        repulsion += diff / dist * (1.0 / (dist * dist + 1e-8))
            
            if np.linalg.norm(repulsion) > 0:
                repulsion = repulsion / np.linalg.norm(repulsion)
                magnitude = 0.03 * (1.0 + (min_dist / avg_dist) * 2.0)
                return repulsion * magnitude
                
        # Otherwise, make subtle adjustment
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        return direction * 0.015

    def _select_neighborhood_based_point(self, points: np.ndarray) -> int:
        """Select point based on neighborhood analysis"""
        distances = pdist(points)
        if len(distances) == 0:
            return np.random.randint(0, len(points))
        
        distance_matrix = squareform(distances)
        avg_dist = np.mean(distances)
        percentile_10 = np.percentile(distances, 10)
        
        # Find points with very close neighbors
        close_neighbor_counts = np.sum(distance_matrix < percentile_10, axis=1)
        close_indices = np.where(close_neighbor_counts > 2)[0]
        
        if len(close_indices) > 0 and np.random.random() < 0.6:
            return np.random.choice(close_indices)
            
        return np.random.randint(0, len(points))

    def _compute_neighborhood_guided_perturbation(self, points: np.ndarray, point_idx: int) -> np.ndarray:
        """Compute perturbation guided by neighborhood analysis"""
        distances = cdist([points[point_idx]], points)[0]
        distances = distances[distances > 0]
        
        if len(distances) == 0:
            return np.random.normal(0, 0.01, 3)
            
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        
        # Adjust based on local density
        if min_dist < avg_dist * 0.4:
            # Repel from close neighbors
            repulsion = np.zeros(3)
            for i in range(len(points)):
                if i != point_idx and distances[i-1] < avg_dist * 0.6:
                    diff = points[point_idx] - points[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        repulsion += diff / dist * (1.0 / (dist * dist + 1e-8))
            
            if np.linalg.norm(repulsion) > 0:
                repulsion = repulsion / np.linalg.norm(repulsion)
                magnitude = 0.025 * (1.0 + (min_dist / avg_dist) * 3.0)
                return repulsion * magnitude
        else:
            # Expand slightly to maximize spread
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            magnitude = 0.02 * (1.0 - min_dist / avg_dist)
            return direction * magnitude

    def _select_harmonic_based_point(self, points: np.ndarray) -> int:
        """Select point based on harmonic analysis"""
        # Sample points where harmonic regularity is low (potential irregularities)
        harmonic_values = []
        for i in range(len(points)):
            h = SphericalHarmonicAnalyzer.compute_spherical_harmonics(points, l=2, m=0)
            harmonic_values.append(abs(h))
            
        harmonic_values = np.array(harmonic_values)
        
        # Select points with low harmonic values (more irregular)
        low_harmonic_indices = np.where(harmonic_values < np.percentile(harmonic_values, 20))[0]
        
        if len(low_harmonic_indices) > 0 and np.random.random() < 0.5:
            return np.random.choice(low_harmonic_indices)
            
        return np.random.randint(0, len(points))

    def _compute_harmonic_guided_perturbation(self, points: np.ndarray, point_idx: int) -> np.ndarray:
        """Compute perturbation guided by harmonic analysis"""
        # Measure current harmonic regularity
        current_h = SphericalHarmonicAnalyzer.compute_spherical_harmonics(points, l=2, m=0)
        
        # Direction based on desired regularization
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Adjust strength based on how far from ideal regularity
        strength = 0.015 * (1.0 - abs(current_h))
        return direction * strength

    def _compute_random_perturbation(self, points: np.ndarray, point_idx: int) -> np.ndarray:
        """Compute standard random perturbation"""
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        return direction * 0.01

    def _compute_min_max_ratio(self, points: np.ndarray) -> Tuple[float, float]:
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0, 0.0

        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 0:
            return 0.0, 0.0

        return d_min / d_max, d_min

    def _update_adaptive_cooling(self, temp: float, base_cooling_rate: float, 
                               current_ratio: float, points: np.ndarray) -> float:
        """Update cooling schedule based on performance and geometric properties"""
        self.recent_ratios.append(current_ratio)
        if len(self.recent_ratios) > self.max_recent:
            self.recent_ratios.pop(0)

        cooling_rate = base_cooling_rate
        
        # Dynamic adjustments based on recent performance
        if len(self.recent_ratios) >= 20:
            recent_improvement = self.recent_ratios[-1] - self.recent_ratios[0]
            improvement_window = len(self.recent_ratios) // 4
            
            if improvement_window > 0:
                recent_improvement_last_quarter = self.recent_ratios[-improvement_window] - self.recent_ratios[-len(self.recent_ratios)]
                if recent_improvement_last_quarter < 1e-8 and temp > 1e-8:
                    # Very slow progress, cool faster
                    cooling_rate = base_cooling_rate * 1.1
                elif recent_improvement_last_quarter > 1e-5 and temp > 1e-6:
                    # Fast progress, cool slower
                    cooling_rate = base_cooling_rate * 0.97
                    
        # Also consider the regularity of the point distribution via harmonic analysis
        regularity = abs(SphericalHarmonicAnalyzer.compute_spherical_harmonics(points, l=2, m=0))
        if regularity < 0.3 and temp > 1e-6:
            # More regular distribution = less need for cooling
            cooling_rate *= 0.99
            
        return cooling_rate

    def _perform_periodic_refinement(self, points: np.ndarray):
        """Perform periodic local refinement"""
        # Cluster points and refine clusters separately
        try:
            # Simple k-means clustering to group nearby points
            kmeans = KMeans(n_clusters=min(4, len(points)//3), random_state=42)
            labels = kmeans.fit_predict(points)
            
            # Refine each cluster
            for i in range(kmeans.n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 1:
                    # Move cluster centroid to optimize inter-cluster spacing
                    cluster_center = np.mean(points[cluster_indices], axis=0)
                    cluster_center = cluster_center / np.linalg.norm(cluster_center)
                    
                    # Apply slight perturbations to cluster members
                    for idx in cluster_indices[:3]:  # Limit to few points per cluster
                        if np.random.random() < 0.3:
                            direction = cluster_center - points[idx]
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                points[idx] += direction * 0.005
                    
        except:
            # Fallback to simple refinement
            for _ in range(20):
                point_idx = np.random.randint(0, len(points))
                new_point = points[point_idx] + np.random.normal(0, 0.001, 3)
                norm = np.linalg.norm(new_point)
                if norm > 0:
                    new_point = new_point / norm
                points[point_idx] = new_point

    def _rebalance_strategy_weights(self, weights: dict, best_ratio: float) -> dict:
        """Rebalance strategy weights based on performance"""
        # If we're doing well, favor more exploration
        if best_ratio > 0.45:
            weights['random'] = min(0.3, weights['random'] + 0.05)
            weights['voronoi'] = max(0.1, weights['voronoi'] - 0.05)
        else:
            # If we're not doing well, favor deterministic approaches
            weights['voronoi'] = min(0.5, weights['voronoi'] + 0.1)
            weights['neighborhood'] = min(0.5, weights['neighborhood'] + 0.1)
            
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

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

    @staticmethod
    def icosahedral_initialization(n: int) -> np.ndarray:
        """Initialize points using icosahedral symmetry for better spread"""
        points = np.zeros((n, 3))

        # Add poles for better symmetry
        points[0] = [0, 0, 1]       # North pole
        points[1] = [0, 0, -1]      # South pole

        # Add equatorial points in two rings
        angle_step = 2 * math.pi / 5
        for i in range(5):
            angle1 = i * angle_step
            angle2 = angle1 + angle_step / 2

            # First ring
            points[2+i] = [math.cos(angle1), math.sin(angle1), 0.0]
            # Second ring offset
            points[7+i] = [math.cos(angle2), math.sin(angle2), 0.0]

        # Add additional points near poles
        points[12] = [0, 0, 0.7]
        points[13] = [0, 0, -0.7]

        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)

        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    # Multi-start optimization with diverse strategies
    best_points = None
    best_ratio = 0.0

    # Try multiple initializations with different strategies
    initializations = [
        ("fibonacci", PointInitializer.fibonacci_sphere),
        ("icosahedral", PointInitializer.icosahedral_initialization)
    ]

    # 10 different Fibonacci sphere attempts with varying seeds
    for i in range(10):
        np.random.seed(42 + i * 1000)  # Different seeds for diversity
        points = PointInitializer.fibonacci_sphere(14)
        optimizer = DistributedOptimizationEngine(max_time=300)
        optimized_points, optimized_ratio = optimizer.optimize(points)

        if optimized_ratio > best_ratio:
            best_ratio = optimized_ratio
            best_points = optimized_points.copy()

    # 5 different icosahedral attempts with varying seeds
    for i in range(5):
        np.random.seed(10000 + i * 100)  # Different seeds for diversity
        points = PointInitializer.icosahedral_initialization(14)
        optimizer = DistributedOptimizationEngine(max_time=300)
        optimized_points, optimized_ratio = optimizer.optimize(points)

        if optimized_ratio > best_ratio:
            best_ratio = optimized_ratio
            best_points = optimized_points.copy()

    # Final refinement with the best configuration
    if best_points is not None:
        np.random.seed(42)  # Reset seed for reproducibility
        optimizer = DistributedOptimizationEngine(max_time=60)  # Shorter time for final refinement
        final_points, final_ratio = optimizer.optimize(best_points)
        if final_ratio > best_ratio:
            best_points = final_points
            best_ratio = final_ratio

    return best_points

# EVOLVE-BLOCK-END