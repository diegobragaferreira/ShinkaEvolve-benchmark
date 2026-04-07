# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import time
from typing import Tuple, List, Optional
import warnings
from dataclasses import dataclass
from collections import defaultdict

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
    num_seeds: int = 8
    voronoi_threshold: float = 0.1
    local_refinement_iters: int = 500
    early_stopping_patience: int = 10000
    stagnation_limit: int = 5000
    refinement_frequency: int = 2000
    clustering_iterations: int = 500
    momentum_factor: float = 0.3

class GeometricClusteringOptimizer:
    """Main optimizer class that uses geometric clustering and momentum-based optimization."""

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
        self.momentum_vectors: Optional[np.ndarray] = None
        self.cluster_assignments: Optional[np.ndarray] = None

    def _fibonacci_sphere(self, n: int, seed_offset: int = 0) -> np.ndarray:
        """Generate points on a unit sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        for i in range(n):
            y = 1 - ((i + seed_offset) / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * (i + seed_offset)  # golden angle increment with offset
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])

        return np.array(points)

    def _calculate_distances_fast(self, points: np.ndarray) -> Tuple[float, float]:
        """Calculate minimum and maximum distances efficiently."""
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
        d_min, d_max = self._calculate_distances_fast(points)
        if d_max <= 0:
            return 0.0
        return d_min / d_max

    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms

    def _compute_cluster_centers(self, points: np.ndarray, n_clusters: int = 4) -> np.ndarray:
        """Compute cluster centers using a modified k-means approach."""
        try:
            # Use sklearn kmeans with spherical distance metric
            # First normalize points to unit sphere (already done in our case)
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, 
                           max_iter=50, tol=1e-4, random_state=42)
            kmeans.fit(points)
            return kmeans.cluster_centers_
        except Exception:
            # Fallback to simple centroid calculation
            return np.array([np.mean(points, axis=0)])

    def _assign_points_to_clusters(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign points to nearest cluster centers with geometric constraint."""
        assignments = []
        for point in points:
            distances = np.linalg.norm(centers - point, axis=1)
            closest_center = np.argmin(distances)
            assignments.append(closest_center)
        return np.array(assignments)

    def _cluster_based_perturbation(self, current_points: np.ndarray, 
                                  cluster_assignments: np.ndarray, 
                                  centers: np.ndarray, temp: float) -> np.ndarray:
        """Generate neighbor using cluster-based perturbation with momentum."""
        neighbor_points = current_points.copy()
        
        # Select a random cluster
        unique_clusters = np.unique(cluster_assignments)
        if len(unique_clusters) > 0:
            selected_cluster = np.random.choice(unique_clusters)
        else:
            selected_cluster = 0
            
        # Get points belonging to selected cluster
        cluster_indices = np.where(cluster_assignments == selected_cluster)[0]
        
        if len(cluster_indices) > 0:
            # Select a random point from the cluster
            selected_point_idx = np.random.choice(cluster_indices)
            
            # Compute cluster-specific momentum vector
            if self.momentum_vectors is not None:
                momentum_vector = self.momentum_vectors[selected_point_idx]
                # Combine with random perturbation
                perturbation = momentum_vector * 0.5 + np.random.normal(0, temp * 0.1, 3)
            else:
                perturbation = np.random.normal(0, temp * 0.1, 3)
            
            # Apply perturbation
            neighbor_points[selected_point_idx] += perturbation
            
            # Ensure it stays on sphere
            neighbor_points[selected_point_idx] = self._project_to_sphere(
                neighbor_points[selected_point_idx].reshape(1, -1)
            ).flatten()
            
        return neighbor_points

    def _geometric_momentum_update(self, current_points: np.ndarray, 
                                 previous_points: np.ndarray, 
                                 temp: float) -> np.ndarray:
        """Update points using geometric momentum from previous iterations."""
        updated_points = current_points.copy()
        
        if self.momentum_vectors is not None:
            # Apply momentum to move points towards better configurations
            momentum_scale = min(1.0, temp * 0.5)
            
            # Create momentum-adjusted update
            for i in range(len(updated_points)):
                # Directional momentum update with smoothing
                if i < len(self.momentum_vectors):
                    momentum_direction = self.momentum_vectors[i]
                    # Blend between momentum and random perturbation
                    combined_update = momentum_direction * momentum_scale + \
                                    np.random.normal(0, temp * 0.05, 3)
                    updated_points[i] += combined_update
                    
        return updated_points

    def _local_geometric_optimization(self, points: np.ndarray, 
                                    max_iters: int = 300) -> Tuple[np.ndarray, float]:
        """Perform local geometric optimization using constrained movements."""
        current_points = points.copy()
        current_ratio = self._calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Local optimization with geometric constraints
        for iter_num in range(max_iters):
            # Generate candidate with geometric constraints
            candidate_points = current_points.copy()
            
            # Select a point with probability inversely proportional to its distance to others
            if len(current_points) > 1:
                # Calculate mean distances to all other points
                distances = pdist(current_points)
                distance_matrix = squareform(distances)
                mean_distances = np.mean(distance_matrix, axis=1)
                
                # Select point with lower mean distance (more crowded points get higher chance)
                probabilities = 1.0 / (mean_distances + 1e-8)
                probabilities /= np.sum(probabilities)
                selected_idx = np.random.choice(len(current_points), p=probabilities)
            else:
                selected_idx = 0
            
            # Apply small perturbation biased towards increasing distance spread
            perturbation = np.random.normal(0, 0.02, 3)
            
            # If point is too close to others, push it outward
            if len(current_points) > 1:
                distances_to_others = np.linalg.norm(
                    current_points[selected_idx] - np.delete(current_points, selected_idx, axis=0), axis=1
                )
                min_distance = np.min(distances_to_others)
                if min_distance < 0.3:  # If too close to someone
                    # Push away from nearest neighbor
                    nearest_idx = np.argmin(distances_to_others)
                    nearest_point = current_points[nearest_idx]
                    direction = current_points[selected_idx] - nearest_point
                    if np.linalg.norm(direction) > 1e-8:
                        direction = direction / np.linalg.norm(direction)
                        perturbation = direction * 0.03 + np.random.normal(0, 0.01, 3)
            
            candidate_points[selected_idx] += perturbation
            candidate_points[selected_idx] = self._project_to_sphere(
                candidate_points[selected_idx].reshape(1, -1)
            ).flatten()
            
            candidate_ratio = self._calculate_ratio(candidate_points)
            
            # Accept improvement or with probability for exploration
            if candidate_ratio > current_ratio or np.random.random() < 0.05:
                current_points = candidate_points
                current_ratio = candidate_ratio
                
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
        
        return best_points, best_ratio

    def _compute_momentum_vectors(self, current_points: np.ndarray, 
                                previous_points: np.ndarray) -> np.ndarray:
        """Compute momentum vectors based on point movement."""
        if previous_points is None:
            return np.zeros_like(current_points)
            
        # Compute displacement vectors
        displacements = current_points - previous_points
        # Normalize displacements to create momentum directions
        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        momentum_vectors = displacements / safe_norms
        
        # Smooth momentum vectors
        return momentum_vectors * self.config.momentum_factor

    def _initialize_with_geometric_clustering(self) -> None:
        """Initialize with geometric clustering approach."""
        best_seed_points = None
        best_seed_ratio = 0.0
        
        # Try multiple Fibonacci seeds
        for seed_offset in range(self.config.num_seeds):
            # Initialize with Fibonacci sphere configuration
            seed_points = self._fibonacci_sphere(self.config.n_points, seed_offset)
            seed_ratio = self._calculate_ratio(seed_points)
            
            if seed_ratio > best_seed_ratio:
                best_seed_ratio = seed_ratio
                best_seed_points = seed_points.copy()
        
        # Enhance using geometric clustering
        if best_seed_points is not None:
            # Apply geometric clustering enhancement
            try:
                # Compute cluster centers
                centers = self._compute_cluster_centers(best_seed_points, n_clusters=4)
                # Assign points to clusters
                assignments = self._assign_points_to_clusters(best_seed_points, centers)
                
                # Refine clustered configuration
                refined_points = best_seed_points.copy()
                for _ in range(10):  # Local refinement passes
                    # Cluster-based perturbations
                    for cluster_id in range(len(centers)):
                        cluster_points = np.where(assignments == cluster_id)[0]
                        if len(cluster_points) > 0:
                            # Move cluster centroids to average positions
                            if len(cluster_points) > 1:
                                new_center = np.mean(refined_points[cluster_points], axis=0)
                                # Adjust points towards cluster center
                                for idx in cluster_points:
                                    direction = new_center - refined_points[idx]
                                    norm_dir = np.linalg.norm(direction)
                                    if norm_dir > 1e-8:
                                        refined_points[idx] += direction * 0.05
                    
                    refined_points = self._project_to_sphere(refined_points)
                    
                self.current_points = refined_points
                self.current_ratio = self._calculate_ratio(refined_points)
                
            except Exception:
                self.current_points = best_seed_points
                self.current_ratio = best_seed_ratio
        
        self.best_points = self.current_points.copy()
        self.best_ratio = self.current_ratio
        self.iteration_count = 0
        self.start_time = time.time()
        self.stagnation_counter = 0
        self.recent_improvements = []
        self.last_improvement_iter = 0
        self.momentum_vectors = None
        self.cluster_assignments = None

    def _optimize_single_iteration(self, temp: float) -> Tuple[np.ndarray, float, bool]:
        """Perform one optimization iteration using geometric clustering approach."""
        previous_points = self.current_points.copy() if self.current_points is not None else None
        
        # Geometric clustering-based perturbation
        if self.cluster_assignments is not None and len(self.cluster_assignments) > 0:
            # Use cluster-based perturbation
            candidate_points = self._cluster_based_perturbation(
                self.current_points,
                self.cluster_assignments,
                self._compute_cluster_centers(self.current_points),
                temp
            )
        else:
            # Standard perturbation
            candidate_points = self.current_points.copy()
            if len(candidate_points) > 0:
                selected_idx = np.random.randint(0, len(candidate_points))
                perturbation = np.random.normal(0, temp * 0.08, 3)
                candidate_points[selected_idx] += perturbation
                candidate_points[selected_idx] = self._project_to_sphere(
                    candidate_points[selected_idx].reshape(1, -1)
                ).flatten()
        
        # Apply geometric momentum update
        if self.momentum_vectors is not None and previous_points is not None:
            momentum_updated = self._geometric_momentum_update(
                candidate_points, previous_points, temp
            )
            candidate_points = momentum_updated
        
        candidate_ratio = self._calculate_ratio(candidate_points)
        
        # Accept or reject based on ratio difference
        delta_ratio = candidate_ratio - self.current_ratio
        
        if delta_ratio > 0:
            accept_prob = 1.0
        else:
            # For negative changes, accept with probability based on temperature
            if temp < 1e-12:
                accept_prob = 0.0
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
        
        # Update momentum vectors
        if self.current_points is not None and previous_points is not None:
            self.momentum_vectors = self._compute_momentum_vectors(
                self.current_points, previous_points
            )
        
        # Update cluster assignments
        if self.current_points is not None and len(self.current_points) > 1:
            try:
                centers = self._compute_cluster_centers(self.current_points, n_clusters=4)
                self.cluster_assignments = self._assign_points_to_clusters(
                    self.current_points, centers
                )
            except Exception:
                self.cluster_assignments = None
        
        return self.current_points, self.current_ratio, accepted

    def _run_optimization_loop(self) -> Tuple[np.ndarray, float]:
        """Run the main optimization loop with geometric clustering."""
        temp = self.config.initial_temp
        previous_points = None
        
        # Main optimization loop
        while self.iteration_count < self.config.max_iterations and temp > self.config.final_temp:
            # Perform single iteration
            _, _, accepted = self._optimize_single_iteration(temp)
            
            # Apply adaptive cooling schedule
            if accepted or self.stagnation_counter < 500:
                temp = max(temp * self.config.cooling_rate, self.config.final_temp)
            
            self.iteration_count += 1
            
            # Periodic local geometric optimization 
            if self.iteration_count % 1000 == 0 and self.iteration_count > 0:
                refined_points, refined_ratio = self._local_geometric_optimization(
                    self.best_points, self.config.local_refinement_iters // 2
                )
                
                if refined_ratio > self.best_ratio:
                    self.best_points = refined_points
                    self.best_ratio = refined_ratio
            
            # Early stopping check
            if self.iteration_count - self.last_improvement_iter > self.config.early_stopping_patience:
                break
                
            # Stagnation handling
            if self.stagnation_counter > self.config.stagnation_limit:
                temp = max(temp * 0.9, self.config.final_temp)
                self.stagnation_counter = 0
        
        return self.best_points, self.best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a geometric clustering evolution approach that combines clustering strategies
    with momentum-based updates and local geometric optimizations to find superior point distributions.
    
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
        stagnation_limit=5000,
        refinement_frequency=2000,
        clustering_iterations=500,
        momentum_factor=0.3
    )
    
    try:
        # Initialize optimizer
        optimizer = GeometricClusteringOptimizer(config)
        
        # Initialize with geometric clustering
        optimizer._initialize_with_geometric_clustering()
        
        # Run optimization
        optimized_points, best_ratio = optimizer._run_optimization_loop()
        
        # Final validation
        if optimized_points is not None:
            final_min, final_max = optimizer._calculate_distances_fast(optimized_points)
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
        return GeometricClusteringOptimizer(config)._fibonacci_sphere(14)

# EVOLVE-BLOCK-END