# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import math
from collections import defaultdict

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def calculate_ratio(points):
        """Calculate min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist
    
    def create_geometric_initial():
        """Create initial configuration based on geometric principles"""
        # Generate points using a combination of geometric patterns
        points = []
        
        # 1. Hexagonal packing pattern (optimal for 2D point distribution)
        rows = 4
        cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Apply hexagonal offset
        for i in range(rows):
            for j in range(cols):
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Ensure bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        # 2. Add some Fibonacci-like spiral points for diversity
        phi = (1 + math.sqrt(5)) / 2
        for i in range(4):
            if len(points) >= 16:
                break
            theta = math.acos(-1 + (2 * i) / 3)
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        # Fill remaining spots with random points near center
        while len(points) < 16:
            x = 0.4 + np.random.uniform(-0.1, 0.1)
            y = 0.4 + np.random.uniform(-0.1, 0.1)
            points.append([x, y])
        
        return np.array(points[:16])
    
    def geometric_local_search(points, max_iterations=50):
        """Specialized local search that respects geometric constraints"""
        current_points = points.copy()
        best_points = current_points.copy()
        best_ratio = calculate_ratio(current_points)
        
        # Create a distance matrix cache for efficiency
        current_distances = pdist(current_points)
        current_min_dist = np.min(current_distances)
        current_max_dist = np.max(current_distances)
        
        if current_max_dist == 0:
            return best_points
            
        # Precompute the indices for distance calculations
        n = len(current_points)
        distance_indices = []
        for i in range(n):
            for j in range(i+1, n):
                distance_indices.append((i, j))
        
        # Geometric constraint preservation approach
        for iteration in range(max_iterations):
            # Store current state
            old_points = current_points.copy()
            old_ratio = best_ratio
            
            # Try small perturbations to each point
            for i in range(n):
                # Create a small perturbation
                perturbation = np.random.normal(0, 0.002, 2)
                
                # Apply perturbation with boundary constraints
                new_point = current_points[i] + perturbation
                new_point[0] = np.clip(new_point[0], 0.001, 0.999)
                new_point[1] = np.clip(new_point[1], 0.001, 0.999)
                
                # Test the new configuration
                test_points = current_points.copy()
                test_points[i] = new_point
                
                # Calculate ratio for this change
                test_distances = pdist(test_points)
                test_min_dist = np.min(test_distances)
                test_max_dist = np.max(test_distances)
                
                if test_max_dist == 0:
                    continue
                    
                test_ratio = test_min_dist / test_max_dist
                
                # Accept improvement or accept with probability
                if test_ratio > best_ratio:
                    current_points = test_points.copy()
                    best_ratio = test_ratio
                    best_points = current_points.copy()
                elif np.random.random() < 0.1:  # 10% chance of accepting worse move
                    current_points = test_points.copy()
        
        # Final verification of the best solution
        final_ratio = calculate_ratio(best_points)
        if final_ratio > best_ratio:
            return best_points
        else:
            return best_points
    
    def constraint_relaxation_optimization(initial_points, max_evaluations=2000):
        """Apply constraint relaxation optimization that works directly with geometric relationships"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = calculate_ratio(current_points)
        
        # Use a specialized optimization that mimics physical spring forces
        # but respects geometric constraints
        n = len(current_points)
        
        # Precompute initial distances
        initial_distances = pdist(current_points)
        initial_min = np.min(initial_distances)
        initial_max = np.max(initial_distances)
        
        if initial_max == 0:
            return best_points
            
        # Simple annealing-like approach with geometric constraints
        temperature = 0.1
        cooling_rate = 0.95
        min_temperature = 1e-6
        
        evaluations = 0
        while evaluations < max_evaluations and temperature > min_temperature:
            # Generate a random perturbation for one point
            idx = np.random.randint(0, n)
            perturbation = np.random.normal(0, temperature * 0.01, 2)
            
            # Apply perturbation with bounds
            new_point = current_points[idx] + perturbation
            new_point[0] = np.clip(new_point[0], 0.001, 0.999)
            new_point[1] = np.clip(new_point[1], 0.001, 0.999)
            
            # Test the change
            test_points = current_points.copy()
            test_points[idx] = new_point
            
            test_ratio = calculate_ratio(test_points)
            
            # Accept based on Metropolis criterion
            if test_ratio > best_ratio:
                current_points = test_points
                best_ratio = test_ratio
                best_points = current_points.copy()
            elif np.random.random() < np.exp((test_ratio - best_ratio) / temperature):
                current_points = test_points
            
            temperature *= cooling_rate
            evaluations += 1
            
            # Occasionally try a global reset if stuck
            if evaluations % 100 == 0 and best_ratio < 0.1:
                current_points = create_geometric_initial()
                best_points = current_points.copy()
                best_ratio = calculate_ratio(current_points)
        
        return best_points
    
    def geometric_clustering_refinement(points):
        """Refine solution using geometric clustering to identify good substructures"""
        # First, compute convex hull to understand overall structure
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # If we have a small convex hull, try to reposition points inside
            if len(hull.vertices) < 8:
                # This might indicate a concentrated cluster; redistribute
                center = np.mean(points, axis=0)
                distances_from_center = np.linalg.norm(points - center, axis=1)
                avg_distance = np.mean(distances_from_center)
                
                # Reposition points further from center if they're clustered
                refined_points = points.copy()
                for i in range(len(points)):
                    if distances_from_center[i] < 0.3 * avg_distance:
                        # Move outward
                        direction = points[i] - center
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            refined_points[i] = center + direction * (avg_distance * 0.8)
                            refined_points[i][0] = np.clip(refined_points[i][0], 0.001, 0.999)
                            refined_points[i][1] = np.clip(refined_points[i][1], 0.001, 0.999)
                
                return refined_points
        except:
            pass
            
        return points
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Geometrically inspired initial configuration
    initial_configs.append(create_geometric_initial())
    
    # 2. Variation with different perturbations
    np.random.seed(42)
    for _ in range(3):
        config = create_geometric_initial() + np.random.normal(0, 0.01, (16, 2))
        config[:, 0] = np.clip(config[:, 0], 0.001, 0.999)
        config[:, 1] = np.clip(config[:, 1], 0.001, 0.999)
        initial_configs.append(config)
    
    # 3. Alternative geometric pattern
    alt_config = np.array([
        [0.1, 0.1], [0.9, 0.1], [0.5, 0.9], [0.5, 0.1],
        [0.1, 0.9], [0.9, 0.9], [0.2, 0.5], [0.8, 0.5],
        [0.5, 0.5], [0.3, 0.3], [0.7, 0.3], [0.3, 0.7],
        [0.7, 0.7], [0.2, 0.8], [0.8, 0.2], [0.6, 0.6]
    ])
    initial_configs.append(alt_config)
    
    # Optimize from each initial configuration
    best_ratio = -np.inf
    best_points = None
    
    for i, initial_config in enumerate(initial_configs):
        # First, apply geometric local search
        local_search_result = geometric_local_search(initial_config.copy(), max_iterations=20)
        local_ratio = calculate_ratio(local_search_result)
        
        # Then, apply constraint relaxation
        refined_result = constraint_relaxation_optimization(local_search_result.copy())
        refined_ratio = calculate_ratio(refined_result)
        
        # Also try geometric clustering refinement
        clustered_result = geometric_clustering_refinement(refined_result.copy())
        clustered_ratio = calculate_ratio(clustered_result)
        
        # Select the best among all refinements
        candidates = [local_search_result, refined_result, clustered_result]
        candidate_ratios = [local_ratio, refined_ratio, clustered_ratio]
        
        for candidate, ratio in zip(candidates, candidate_ratios):
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = candidate.copy()
    
    # If still not good enough, run a final intensive optimization
    if best_points is None or best_ratio < 0.2:
        # Start with the best of our initial attempts and run a more intensive optimization
        final_points = best_points.copy() if best_points is not None else initial_configs[0]
        
        # Run constraint relaxation with higher evaluation count
        final_points = constraint_relaxation_optimization(final_points, max_evaluations=3000)
        
        final_ratio = calculate_ratio(final_points)
        if final_ratio > best_ratio:
            best_points = final_points
    
    # Final geometric refinement if needed
    if best_points is not None:
        final_refinement = geometric_clustering_refinement(best_points)
        final_ratio = calculate_ratio(final_refinement)
        if final_ratio > best_ratio:
            best_points = final_refinement
    
    # Final fallback to a simple hexagonal pattern if nothing worked well
    if best_points is None:
        best_points = create_geometric_initial()
    
    return best_points

# EVOLVE-BLOCK-END