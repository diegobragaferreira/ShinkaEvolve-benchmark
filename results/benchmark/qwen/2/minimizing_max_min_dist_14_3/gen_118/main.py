# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple, List, Optional
import warnings
from dataclasses import dataclass

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
    local_refinement_iters: int = 500
    early_stopping_patience: int = 10000
    stagnation_limit: int = 5000
    checkpoint_interval: int = 30

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a hybrid approach combining Fibonacci initialization with adaptive simulated annealing
    and checkpoint-restart mechanisms for improved convergence over traditional Voronoi-based methods.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Configuration parameters
    config = OptimizationConfig(
        n_points=14,
        n_dimensions=3,
        max_iterations=100000,
        initial_temp=1.0,
        final_temp=1e-6,
        cooling_rate=0.9995,
        log_interval=1000,
        num_seeds=8,
        local_refinement_iters=500,
        early_stopping_patience=10000,
        stagnation_limit=5000,
        checkpoint_interval=30
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
    
    def distance_weighted_perturbation(current_points: np.ndarray, temp: float) -> np.ndarray:
        """
        Generate neighbor by perturbing points based on their distance characteristics.
        Points that are too close to neighbors get more aggressive perturbations.
        """
        neighbor_points = current_points.copy()
        
        # Calculate pairwise distances
        if len(current_points) < 2:
            return neighbor_points
            
        distances = pdist(current_points)
        distance_matrix = squareform(distances)
        
        # For each point, determine if it's too close to others
        # We'll perturb points that have neighbors below median distance
        mean_distances = np.mean(distance_matrix, axis=1)
        median_distance = np.median(mean_distances)
        
        # Select points that are closer than median to their neighbors
        points_to_perturb = np.where(mean_distances < median_distance)[0]
        
        # If no points are too close, just pick a random one
        if len(points_to_perturb) == 0:
            points_to_perturb = [np.random.randint(0, len(current_points))]
        
        # Pick one point to perturb (weighted by how much it needs fixing)
        if len(points_to_perturb) > 1:
            # Weight by inverse of mean distance (closer points get higher weights)
            weights = 1.0 / (mean_distances[points_to_perturb] + 1e-8)
            weights = weights / np.sum(weights)
            idx = np.random.choice(points_to_perturb, p=weights)
        else:
            idx = points_to_perturb[0]
        
        # Determine perturbation magnitude based on temperature and how close it is
        perturbation_mag = temp * 0.1
        
        # If point is really close to others, perturb more aggressively
        if mean_distances[idx] < median_distance * 0.5:
            perturbation_mag *= 2.0
            
        # Add Gaussian noise to selected point
        noise = np.random.normal(0, perturbation_mag, 3)
        neighbor_points[idx] += noise
        
        # Project back to sphere
        neighbor_points = project_to_sphere(neighbor_points)
        
        return neighbor_points
    
    def adaptive_local_refinement(starting_points: np.ndarray, max_iters: int = 500) -> Tuple[np.ndarray, float]:
        """Apply local refinement to improve convergence."""
        current_points = starting_points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Simple gradient-free local search
        for iter_num in range(max_iters):
            # Use distance-weighted perturbation for local refinement
            candidate_points = distance_weighted_perturbation(current_points, temp=0.01)
            
            candidate_ratio = calculate_ratio(candidate_points)
            
            # Always accept improvement
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
        
        # Track checkpoint information
        best_checkpoint_points = None
        best_checkpoint_ratio = 0.0
        last_checkpoint_iter = 0
        
        for seed_offset in range(config.num_seeds):
            # Initialize with Fibonacci sphere configuration
            current_points = fibonacci_sphere(config.n_points, seed_offset)
            current_ratio = calculate_ratio(current_points)
            
            # Store best from this seed
            seed_best_points = current_points.copy()
            seed_best_ratio = current_ratio
            
            # Main optimization loop
            temp = config.initial_temp
            iteration = 0
            stagnant_count = 0
            last_improvement_iter = 0
            iteration_since_checkpoint = 0
            
            while iteration < config.max_iterations and temp > config.final_temp:
                # Distance-weighted perturbation
                candidate_points = distance_weighted_perturbation(current_points, temp=temp)
                
                candidate_ratio = calculate_ratio(candidate_points)
                
                # Metropolis acceptance criterion
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
                iteration_since_checkpoint += 1
                
                # Checkpoint every N iterations
                if iteration_since_checkpoint >= config.checkpoint_interval:
                    if seed_best_ratio > best_checkpoint_ratio:
                        best_checkpoint_ratio = seed_best_ratio
                        best_checkpoint_points = seed_best_points.copy()
                    last_checkpoint_iter = iteration
                    iteration_since_checkpoint = 0
                
                # Early stopping based on no improvement
                if iteration - last_improvement_iter > config.early_stopping_patience:
                    # Restart from best checkpoint if available
                    if best_checkpoint_points is not None and best_checkpoint_ratio > seed_best_ratio * 0.99:
                        seed_best_points = best_checkpoint_points.copy()
                        seed_best_ratio = best_checkpoint_ratio
                        last_improvement_iter = iteration
                        # Reset checkpoint tracking
                        iteration_since_checkpoint = 0
                    else:
                        break
                
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