# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
from scipy.spatial import SphericalVoronoi
import time

def create_geometric_initialization(n_points: int = 14) -> np.ndarray:
    """
    Create initial point configuration based on geometric principles.
    Uses a modified Fibonacci distribution adapted for 3D point placement.
    """
    points = []
    
    # Generate points using modified Fibonacci-like distribution
    # More uniform than basic Fibonacci but still structured
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Generate points along spiral pattern on sphere
    for i in range(n_points):
        # Distribute points more evenly in z-direction
        z = 1 - (i / (n_points - 1)) * 2  # z from 1 to -1
        radius = np.sqrt(1 - z*z)
        
        # Modified angular spacing to avoid clustering
        theta = np.arctan2(z, radius) + (i * 2 * np.pi / n_points) + (i * 0.1) 
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.where(norms == 0, 1, norms)
    
    return points

def calculate_min_max_ratio(points: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate minimum and maximum distances between all point pairs.
    Returns (min_distance, max_distance, ratio).
    """
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0, 0.0, 0.0
        
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist <= 0:
        return 0.0, 0.0, 0.0
    
    ratio = min_dist / max_dist
    return min_dist, max_dist, ratio

def project_to_sphere(points: np.ndarray) -> np.ndarray:
    """Project points to unit sphere maintaining relative directions."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def geometric_relaxation_step(points: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Apply geometric relaxation using force-based repulsion model.
    Each point repels others with inverse-square law, projected back to sphere.
    """
    points = points.copy()
    
    for _ in range(iterations):
        # Calculate pairwise distances
        n = len(points)
        forces = np.zeros_like(points)
        
        # Compute repulsive forces between all pairs
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                
                # Avoid singularity
                if dist_sq > 1e-10:
                    force_magnitude = 1.0 / dist_sq
                    forces[i] += force_magnitude * diff
                    forces[j] -= force_magnitude * diff
        
        # Apply forces and project back to sphere
        points += 0.01 * forces  # Small step size
        points = project_to_sphere(points)
    
    return points

def adaptive_local_optimization(points: np.ndarray, max_time: float = 5.0) -> np.ndarray:
    """
    Perform adaptive local optimization using multiple strategies.
    """
    best_points = points.copy()
    _, _, best_ratio = calculate_min_max_ratio(best_points)
    
    start_time = time.time()
    
    # Strategy 1: Geometric relaxation
    relaxed_points = geometric_relaxation_step(best_points, iterations=20)
    _, _, ratio = calculate_min_max_ratio(relaxed_points)
    if ratio > best_ratio:
        best_ratio = ratio
        best_points = relaxed_points.copy()
    
    # Strategy 2: Differential evolution on subset
    if time.time() - start_time < max_time * 0.7:
        try:
            # Select subset of points for faster optimization
            subset_indices = np.random.choice(len(points), size=min(8, len(points)), replace=False)
            subset_points = points[subset_indices]
            
            # Flatten points for optimization
            x0 = subset_points.flatten()
            bounds = [(-1.0, 1.0) for _ in range(len(x0))]
            
            def objective_subset(x_flat):
                temp_points = points.copy()  # Keep original points
                temp_points[subset_indices] = x_flat.reshape(-1, 3)
                temp_points = project_to_sphere(temp_points)
                distances = pdist(temp_points)
                if len(distances) == 0:
                    return float('inf')
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist <= 0:
                    return float('inf')
                return -min_dist / max_dist
            
            result = differential_evolution(
                objective_subset,
                bounds,
                maxiter=20,
                popsize=10,
                seed=42,
                disp=False,
                tol=1e-6
            )
            
            # Update points with result
            new_subset = result.x.reshape(-1, 3)
            temp_points = points.copy()
            temp_points[subset_indices] = new_subset
            temp_points = project_to_sphere(temp_points)
            
            _, _, ratio = calculate_min_max_ratio(temp_points)
            if ratio > best_ratio:
                best_points = temp_points.copy()
                best_ratio = ratio
                
        except:
            pass
    
    return best_points

def create_symmetric_variants(points: np.ndarray, num_variants: int = 4) -> np.ndarray:
    """
    Create symmetric variants by rotating around different axes.
    """
    variants = [points]
    
    # Create rotations around different axes
    for i in range(num_variants):
        angle = 2 * np.pi * (i + 1) / (num_variants + 1)
        
        # Rotation around z-axis
        rot_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        rotated = points @ rot_z.T
        variants.append(rotated)
        
        # Rotation around x-axis
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        
        rotated_x = points @ rot_x.T
        variants.append(rotated_x)
    
    return np.vstack(variants)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Implements a novel geometric optimization approach combining structured initialization,
    force-based relaxation, and adaptive refinement.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Phase 1: Structured geometric initialization
    initial_points = create_geometric_initialization(14)
    
    # Phase 2: Multi-stage geometric optimization
    optimized_points = initial_points.copy()
    
    # Apply iterative geometric relaxation
    optimized_points = geometric_relaxation_step(optimized_points, iterations=30)
    
    # Apply adaptive local optimization
    optimized_points = adaptive_local_optimization(optimized_points)
    
    # Phase 3: Explore symmetric variants
    variants = create_symmetric_variants(optimized_points, num_variants=3)
    
    # Evaluate all variants and select best
    best_points = optimized_points.copy()
    best_ratio = 0.0
    
    # Check all variants
    for i in range(len(variants) // 14):
        variant_points = variants[i*14:(i+1)*14]
        _, _, ratio = calculate_min_max_ratio(variant_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = variant_points.copy()
    
    # Phase 4: Final global optimization if beneficial
    try:
        # Test if global optimization helps
        test_points = best_points.copy()
        test_points = project_to_sphere(test_points)
        _, _, test_ratio = calculate_min_max_ratio(test_points)
        
        # Only do global optimization if it might improve results
        if test_ratio < 0.2:  # Threshold for when global optimization is worth it
            x0 = test_points.flatten()
            bounds = [(-1.0, 1.0) for _ in range(14*3)]
            
            def objective_global(x_flat):
                points = x_flat.reshape(14, 3)
                points = project_to_sphere(points)
                distances = pdist(points)
                if len(distances) == 0:
                    return float('inf')
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist <= 0:
                    return float('inf')
                return -min_dist / max_dist
            
            # Run differential evolution for final polish
            result = differential_evolution(
                objective_global,
                bounds,
                maxiter=30,
                popsize=15,
                seed=42,
                disp=False,
                tol=1e-6
            )
            
            # Check if this improves results
            final_points = result.x.reshape(14, 3)
            final_points = project_to_sphere(final_points)
            _, _, final_ratio = calculate_min_max_ratio(final_points)
            
            if final_ratio > test_ratio:
                best_points = final_points.copy()
                
    except:
        pass
    
    # Final projection to sphere
    final_points = project_to_sphere(best_points)
    
    return final_points

# EVOLVE-BLOCK-END