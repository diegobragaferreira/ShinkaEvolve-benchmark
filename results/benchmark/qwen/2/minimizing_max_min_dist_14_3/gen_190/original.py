# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import time
from typing import Tuple, List, Optional
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses an innovative hybrid approach combining coarse-to-fine initialization with 
    distance-sensitive perturbation strategies to achieve superior convergence.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    def fibonacci_sphere(n: int, seed_offset: int = 0) -> np.ndarray:
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
    
    def calculate_distances(points: np.ndarray) -> Tuple[float, float]:
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
    
    def find_problematic_pairs(points: np.ndarray, threshold_ratio: float = 0.7) -> List[Tuple[int, int]]:
        """Find pairs of points that are either too close or too far based on current distribution."""
        if len(points) < 2:
            return []
            
        distances = pdist(points)
        distance_matrix = squareform(distances)
        n = len(points)
        
        # Find pairs that are either very close or very far from expected
        # Expected distance for uniform distribution
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        problematic_pairs = []
        
        # Look for pairs that are more than threshold_ratio times the mean distance
        # or less than 1/threshold_ratio of the mean distance
        for i in range(n):
            for j in range(i+1, n):
                dist = distance_matrix[i, j]
                # Mark as problematic if too close or too far
                if dist < mean_dist * (1.0/threshold_ratio) or dist > mean_dist * threshold_ratio:
                    problematic_pairs.append((i, j))
        
        return problematic_pairs
    
    def adaptive_perturbation(points: np.ndarray, temp: float, 
                            problematic_pairs: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """Generate neighbor by intelligently perturbing points based on distance characteristics."""
        neighbor_points = points.copy()
        
        # Get current distances for analysis
        if len(points) < 2:
            return neighbor_points
            
        # Calculate all pairwise distances
        distances = pdist(points)
        distance_matrix = squareform(distances)
        
        # If no problematic pairs, choose based on local density
        if problematic_pairs is None or len(problematic_pairs) == 0:
            # Find points that are part of many very close pairs (high density region)
            # and points that are isolated (low density region)  
            mean_distances = np.mean(distance_matrix, axis=1)
            # Select points with extreme mean distances
            sorted_indices = np.argsort(mean_distances)
            # Take the most extreme ones (both highest and lowest means)
            candidates = sorted_indices[:max(1, len(sorted_indices)//4)]
            candidates = list(candidates) + list(sorted_indices[-max(1, len(sorted_indices)//4):])
            point_idx = np.random.choice(candidates)
        else:
            # Focus on problematic pairs - perturb points involved in those pairs
            involved_points = set()
            for i, j in problematic_pairs:
                involved_points.add(i)
                involved_points.add(j)
            
            # If we have involved points, choose from them
            if involved_points:
                point_idx = np.random.choice(list(involved_points))
            else:
                point_idx = np.random.randint(0, len(points))
        
        # Determine perturbation magnitude based on temperature and context
        base_perturbation = temp * 0.1
        
        # Calculate local density around chosen point
        if len(points) > 1:
            # Compute mean distance to neighbors
            mean_neighbor_dist = np.mean(distance_matrix[point_idx])
            # Also check if this point is unusually close to others
            mean_all_dists = np.mean(distances)
            
            # If point is unusually close to neighbors, make larger perturbations
            if mean_neighbor_dist < mean_all_dists * 0.5:
                base_perturbation *= 2.0
            # If point is unusually far from neighbors, make smaller perturbations
            elif mean_neighbor_dist > mean_all_dists * 1.5:
                base_perturbation *= 0.5
        
        # Add Gaussian noise to selected point
        noise = np.random.normal(0, base_perturbation, 3)
        neighbor_points[point_idx] += noise
        
        # Project back to sphere
        neighbor_points = project_to_sphere(neighbor_points)
        
        return neighbor_points
    
    def coarse_to_fine_optimization() -> Tuple[np.ndarray, float]:
        """Run optimization with coarse-to-fine resolution approach."""
        # Initialize with Fibonacci sphere
        points = fibonacci_sphere(14)
        
        # Coarse optimization phase - start with large steps
        temp = 1.0
        max_iter_coarse = 5000
        prev_ratio = calculate_ratio(points)
        
        for iteration in range(max_iter_coarse):
            # Adaptive perturbation focusing on problem areas
            problematic_pairs = find_problematic_pairs(points, 0.7)
            candidate_points = adaptive_perturbation(points, temp, problematic_pairs)
            
            candidate_ratio = calculate_ratio(candidate_points)
            
            # Metropolis acceptance
            delta_ratio = candidate_ratio - prev_ratio
            accept_prob = min(1.0, np.exp(delta_ratio / temp))
            
            if np.random.random() < accept_prob:
                points = candidate_points
                prev_ratio = candidate_ratio
            
            # Gradually decrease temperature and adjust
            if iteration % 500 == 0:
                temp = max(temp * 0.95, 0.1)
        
        # Fine optimization phase - smaller steps
        temp = 0.1
        max_iter_fine = 15000
        best_points = points.copy()
        best_ratio = prev_ratio
        
        for iteration in range(max_iter_fine):
            # Focus on problematic pairs
            problematic_pairs = find_problematic_pairs(points, 0.8)
            candidate_points = adaptive_perturbation(points, temp, problematic_pairs)
            
            candidate_ratio = calculate_ratio(candidate_points)
            
            # Metropolis acceptance
            delta_ratio = candidate_ratio - prev_ratio
            accept_prob = min(1.0, np.exp(delta_ratio / temp))
            
            if np.random.random() < accept_prob:
                points = candidate_points
                prev_ratio = candidate_ratio
                
                if prev_ratio > best_ratio:
                    best_points = points.copy()
                    best_ratio = prev_ratio
            
            # Gradually decrease temperature
            if iteration % 1000 == 0:
                temp = max(temp * 0.92, 0.01)
        
        return best_points, best_ratio
    
    def local_refinement(points: np.ndarray, max_iters: int = 3000) -> Tuple[np.ndarray, float]:
        """Local refinement to fine-tune the solution."""
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Focus on specific geometric configurations for better local search
        for iter_num in range(max_iters):
            # Use aggressive perturbations to escape local minima
            # Perturb multiple points at once
            num_perturbed = max(1, len(current_points) // 4)
            indices_to_perturb = np.random.choice(len(current_points), num_perturbed, replace=False)
            
            candidate_points = current_points.copy()
            
            for idx in indices_to_perturb:
                # Larger perturbations for local refinement
                perturbation = np.random.normal(0, 0.02, 3)
                candidate_points[idx] += perturbation
            
            # Project back to sphere
            candidate_points = project_to_sphere(candidate_points)
            
            candidate_ratio = calculate_ratio(candidate_points)
            
            if candidate_ratio > current_ratio:
                current_points = candidate_points
                current_ratio = candidate_ratio
                
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
        
        return best_points, best_ratio
    
    try:
        # Run the main optimization
        optimized_points, best_ratio = coarse_to_fine_optimization()
        
        # Apply local refinement
        refined_points, refined_ratio = local_refinement(optimized_points)
        
        # Final validation and return
        final_points = refined_points
        final_ratio = refined_ratio
        
        if final_ratio <= 0:
            warnings.warn("Final validation failed, returning Fibonacci sphere initialization")
            final_points = fibonacci_sphere(14)
        
        return final_points
    
    except Exception as e:
        warnings.warn(f"Optimization failed with error: {str(e)}, returning Fibonacci sphere initialization")
        return fibonacci_sphere(14)

# EVOLVE-BLOCK-END