# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Voronoi-based geometric optimization approach.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_voronoi_min_distance(points):
        """Calculate minimum distance using Voronoi diagram properties"""
        try:
            vor = Voronoi(points)
            min_dist = float('inf')
            
            # For each point, find minimum distance to any other point
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = np.linalg.norm(points[i] - points[j])
                    min_dist = min(min_dist, dist)
                    
            return min_dist if min_dist != float('inf') else 0.0
        except:
            # Fallback to direct calculation
            if len(points) < 2:
                return 0.0
            distances = pdist(points)
            return np.min(distances) if len(distances) > 0 else 0.0

    def voronoi_objective(points_flat):
        """Objective function based on Voronoi properties"""
        points = points_flat.reshape(-1, 2)
        
        # Apply boundary constraints
        eps = 1e-8
        points = np.clip(points, eps, 1-eps)
        
        # Calculate pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 1e10
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return 1e10
            
        # Ratio optimization with penalty for poor Voronoi structure
        ratio = min_dist / max_dist
        
        # Add penalty for irregular Voronoi cells (indicating poor distribution)
        try:
            vor = Voronoi(points)
            # Calculate variance of Voronoi cell areas (lower variance = more uniform distribution)
            if hasattr(vor, 'areas'):
                area_variance = np.var(vor.areas) if len(vor.areas) > 1 else 0
                # Penalize high variance in cell areas
                penalty = area_variance * 1000
                return -(ratio - penalty)
        except:
            pass
            
        return -ratio

    def generate_voronoi_optimized_initial():
        """Generate initial configuration optimized for good Voronoi structure"""
        # Start with a hexagonal pattern that promotes uniformity
        np.random.seed(42)
        points = []
        
        # Create a hexagonal lattice pattern
        rows = 4
        cols = 4
        
        # Hexagonal spacing constants
        sqrt3 = np.sqrt(3)
        spacing_x = 0.8 / (cols - 1) if cols > 1 else 0.5
        spacing_y = 0.8 / (rows - 1) if rows > 1 else 0.5
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Hexagonal offset pattern
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                
                # Offset odd rows
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                points.append([x, y])
                
        points = np.array(points[:16])
        
        # Add some randomness to break symmetry but maintain structure
        points += np.random.normal(0, 0.01, points.shape)
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        
        return points

    def voronoi_guided_local_search(initial_points, max_iter=500):
        """Perform local search guided by Voronoi structure analysis"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = -1e10
        
        # Track recent improvements to detect stagnation
        recent_improvements = []
        
        for iteration in range(max_iter):
            # Evaluate current configuration
            distances = pdist(current_points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()
                        
                    recent_improvements.append(ratio)
                    if len(recent_improvements) > 10:
                        recent_improvements.pop(0)
            
            # Create candidate by perturbing one point at a time
            candidate_points = current_points.copy()
            
            # Perturb a random point
            idx = random.randint(0, 15)
            # Add small perturbation
            candidate_points[idx] += np.random.normal(0, 0.005, 2)
            # Keep within bounds
            candidate_points = np.clip(candidate_points, 1e-8, 1-1e-8)
            
            # Evaluate candidate
            distances = pdist(candidate_points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    candidate_ratio = min_dist / max_dist
                    # Accept if better or with some probability
                    if candidate_ratio > ratio or random.random() < 0.1:
                        current_points = candidate_points.copy()
            
            # Occasionally restart from best known position
            if iteration % 50 == 0 and len(recent_improvements) > 5:
                avg_improvement = np.mean(recent_improvements[-5:])
                if avg_improvement > 0.95 * best_ratio:
                    # Restart with best configuration plus noise
                    current_points = best_points.copy() + np.random.normal(0, 0.001, best_points.shape)
                    current_points = np.clip(current_points, 1e-8, 1-1e-8)
        
        return best_points, best_ratio

    def smart_differential_evolution(points, maxiter=100):
        """Enhanced differential evolution with smart bounds and constraints"""
        bounds = [(1e-8, 1-1e-8) for _ in range(32)]
        
        def bounded_objective(x):
            points = np.clip(x.reshape(-1, 2), 1e-8, 1-1e-8).flatten()
            return voronoi_objective(points)
        
        try:
            result = differential_evolution(
                bounded_objective,
                bounds,
                maxiter=maxiter,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                return optimized_points, True
        except:
            pass
            
        return points, False

    # Generate initial configurations using Voronoi-aware patterns
    initial_configs = []
    
    # Configuration 1: Hexagonal pattern
    hex_config = generate_voronoi_optimized_initial()
    initial_configs.append(hex_config.copy())
    
    # Configuration 2: Spiral pattern
    spiral_points = []
    for i in range(16):
        if i == 0:
            spiral_points.append([0.5, 0.5])
        else:
            angle = i * 2.0
            radius = min(0.4, i * 0.06)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            spiral_points.append([x, y])
    initial_configs.append(np.array(spiral_points[:16]).clip(0, 1))
    
    # Configuration 3: Grid pattern with noise
    grid_points = []
    for i in range(4):
        for j in range(4):
            if len(grid_points) >= 16:
                break
            base_x = 0.1 + i * 0.225
            base_y = 0.1 + j * 0.225
            jitter_x = np.random.normal(0, 0.015)
            jitter_y = np.random.normal(0, 0.015)
            grid_points.append([base_x + jitter_x, base_y + jitter_y])
    initial_configs.append(np.array(grid_points[:16]).clip(0, 1))
    
    # Configuration 4: Random pattern
    initial_configs.append(np.random.rand(16, 2))
    
    # Try each initial configuration
    best_ratio = -1e10
    best_points = None
    
    for i, initial_config in enumerate(initial_configs):
        try:
            # Step 1: Global optimization with DE
            de_points, de_success = smart_differential_evolution(initial_config.copy(), maxiter=80)
            
            # Step 2: Local refinement with Voronoi-guided search
            refined_points, refined_ratio = voronoi_guided_local_search(de_points, max_iter=200)
            
            # Step 3: Second round of local search with different parameters
            if refined_ratio > best_ratio:
                # Perform additional refinement
                final_points, final_ratio = voronoi_guided_local_search(refined_points, max_iter=150)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()
                    
        except Exception as e:
            continue
    
    # Fallback
    if best_points is None:
        # Use the best initial configuration with focused local search
        fallback_config = generate_voronoi_optimized_initial()
        best_points, _ = voronoi_guided_local_search(fallback_config, max_iter=300)
    
    # Ensure final bounds compliance
    best_points = np.clip(best_points, 1e-8, 1-1e-8)
    
    return best_points

# EVOLVE-BLOCK-END