# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import SphericalVoronoi
import time
import math
from numba import jit

@jit(nopython=True)
def compute_min_max_ratio_fast(points):
    """Fast numba-compiled distance ratio computation"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0

    for i in range(n):
        for j in range(i+1, n):
            dist_sq = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2 + (points[i,2]-points[j,2])**2
            dist = np.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return min_dist, max_dist

def fibonacci_sphere(samples=14):
    """Generate points distributed evenly on a sphere using Fibonacci method"""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        points.append([x, y, z])
        
    return np.array(points)

def calculate_ratio(points):
    """Calculate the min/max distance ratio"""
    if len(points) < 2:
        return 0.0
        
    # Use fast compiled version
    min_dist, max_dist = compute_min_max_ratio_fast(points)
    
    # Avoid division by zero
    if max_dist <= 0:
        return 0.0
        
    return min_dist / max_dist

def calculate_voronoi_fitness(points):
    """Calculate fitness based on Voronoi uniformity and distance ratio"""
    try:
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points, radius=1.0)
        areas = sv.calculate_areas()
        
        # Calculate variance of cell areas for uniformity measure
        area_variance = np.var(areas) if len(areas) > 1 else 0.0
        
        # Get distance ratio
        ratio = calculate_ratio(points)
        
        # Combined fitness: prioritize both uniformity and distance ratio
        # Lower area variance + higher distance ratio = better fitness
        fitness = ratio - 0.1 * area_variance
        
        return fitness, ratio
    except:
        # Fallback to simple distance ratio if Voronoi fails
        ratio = calculate_ratio(points)
        return ratio, ratio

def project_to_unit_sphere(points):
    """Project points to unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def adaptive_perturbation(current_points, temperature, ratio):
    """Apply adaptive perturbations targeting under-separated regions"""
    # Create copy of points
    new_points = current_points.copy()
    
    # Analyze current distribution
    distances = cdist(current_points, current_points)
    np.fill_diagonal(distances, np.inf)
    
    # Find minimum distances for each point
    min_distances = np.min(distances, axis=1)
    
    # Identify points that are potentially under-separated or over-separated
    avg_min_dist = np.mean(min_distances)
    std_min_dist = np.std(min_distances)
    
    # Score points based on their distance characteristics
    scores = np.zeros(len(current_points))
    for i in range(len(current_points)):
        # Points with very small minimum distances should be moved apart
        if min_distances[i] < avg_min_dist * 0.5:
            scores[i] = -min_distances[i]  # Stronger penalty for being too close
        elif min_distances[i] > avg_min_dist * 2.0:
            scores[i] = min_distances[i]   # Encourage bringing far points closer
        else:
            scores[i] = 0  # No strong preference
    
    # Choose point to perturb with adaptive probability
    if np.sum(np.abs(scores)) > 0:
        # Use weighted probability based on scores
        probs = np.abs(scores)
        probs = probs / np.sum(probs)
        target_idx = np.random.choice(len(current_points), p=probs)
    else:
        # Fallback to random selection
        target_idx = np.random.randint(len(current_points))
    
    # Adaptive perturbation strength based on temperature and distance characteristics
    base_strength = 0.01 * temperature
    
    # If points are very clustered, increase perturbation
    if np.min(min_distances) < 0.3:
        strength_multiplier = 3.0
    elif np.min(min_distances) < 0.6:
        strength_multiplier = 2.0
    else:
        strength_multiplier = 1.0
    
    # Apply perturbation
    delta = np.random.normal(0, base_strength * strength_multiplier, 3)
    
    # Apply to target point
    new_points[target_idx] += delta
    
    # Project back to unit sphere
    new_points = project_to_unit_sphere(new_points)
    
    return new_points

def multi_stage_optimization(initial_points, max_iterations=15000):
    """Multi-stage optimization with progressive refinement"""
    # Stage 1: Global exploration with high temperature
    points = initial_points.copy()
    current_ratio = calculate_ratio(points)
    best_ratio = current_ratio
    best_points = points.copy()
    
    # Temperature schedule for different phases
    temperature = 1.0
    min_temp = 1e-6
    cooling_rate = 0.999
    
    # Track improvement history for adaptive cooling
    ratio_history = [current_ratio]
    
    # Phase 1: High temperature exploration (first 5000 iterations)
    for i in range(5000):
        # Apply adaptive perturbation
        new_points = adaptive_perturbation(points, temperature, current_ratio)
        
        # Calculate new ratio
        new_ratio = calculate_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio:
            points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else:
            # Accept with probability based on temperature
            if np.random.rand() < np.exp((new_ratio - current_ratio) / temperature):
                points = new_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
        
        # Update temperature
        temperature = max(temperature * cooling_rate, min_temp)
        
        # Track history for adaptive cooling
        ratio_history.append(current_ratio)
        if len(ratio_history) > 100:
            ratio_history = ratio_history[-100:]  # Keep recent history
    
    # Phase 2: Medium temperature exploitation (next 5000 iterations)
    max_temp = 0.5
    temperature = max_temp
    for i in range(5000):
        # Apply adaptive perturbation
        new_points = adaptive_perturbation(points, temperature, current_ratio)
        
        # Calculate new ratio
        new_ratio = calculate_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio:
            points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else:
            # Accept with probability based on temperature
            if np.random.rand() < np.exp((new_ratio - current_ratio) / temperature):
                points = new_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
        
        # Update temperature
        temperature = max(temperature * cooling_rate, min_temp)
        
        # Track history
        ratio_history.append(current_ratio)
        if len(ratio_history) > 100:
            ratio_history = ratio_history[-100:]
    
    # Phase 3: Fine tuning with low temperature (last 5000 iterations)
    temperature = 0.05
    for i in range(5000):
        # Apply adaptive perturbation
        new_points = adaptive_perturbation(points, temperature, current_ratio)
        
        # Calculate new ratio
        new_ratio = calculate_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio:
            points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else:
            # Accept with probability based on temperature
            if np.random.rand() < np.exp((new_ratio - current_ratio) / temperature):
                points = new_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
        
        # Track history
        ratio_history.append(current_ratio)
        if len(ratio_history) > 100:
            ratio_history = ratio_history[-100:]
    
    return best_points, best_ratio

def local_refinement(points, max_iter=1000):
    """Local refinement with multiple restarts"""
    best_points = points.copy()
    best_ratio = calculate_ratio(points)
    
    # Multiple restarts with small random perturbations
    for restart in range(3):
        # Start with current points
        current_points = points.copy()
        
        # Add small random noise
        noise = np.random.normal(0, 0.005, current_points.shape)
        current_points += noise
        current_points = project_to_unit_sphere(current_points)
        
        # Simple local search for a few iterations
        for iter in range(max_iter // 3):
            # Try to improve by perturbing one point at a time
            idx = np.random.randint(0, len(current_points))
            old_point = current_points[idx].copy()
            
            # Small perturbation
            delta = np.random.normal(0, 0.001, 3)
            current_points[idx] += delta
            current_points = project_to_unit_sphere(current_points)
            
            new_ratio = calculate_ratio(current_points)
            
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
            else:
                # Revert if not better
                current_points[idx] = old_point
        
        # Also try direct optimization if available
        try:
            from scipy.optimize import minimize
            
            def objective(x_flat):
                points_test = x_flat.reshape(-1, 3)
                points_test = project_to_unit_sphere(points_test)
                ratio = calculate_ratio(points_test)
                return -ratio  # negative because we want to maximize
            
            def constraint_func(x_flat):
                points_test = x_flat.reshape(-1, 3)
                norms = np.linalg.norm(points_test, axis=1)
                return norms - 1.0
            
            x0 = current_points.flatten()
            cons = {'type': 'eq', 'fun': constraint_func}
            
            result = minimize(objective, x0, method='SLSQP', constraints=cons,
                             options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_points = project_to_unit_sphere(refined_points)
                refined_ratio = calculate_ratio(refined_points)
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
                    
        except:
            pass
    
    return best_points, best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a multi-stage hybrid optimization approach combining evolutionary strategies with Voronoi 
    analysis, adaptive perturbations, and local refinement to achieve superior results.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Multiple initialization strategies
    initial_strategies = []
    
    # Strategy 1: Fibonacci sphere 
    points1 = fibonacci_sphere(14)
    initial_strategies.append(("fibonacci", points1))
    
    # Strategy 2: Random points on sphere
    points2 = np.random.randn(14, 3)
    points2 = project_to_unit_sphere(points2)
    initial_strategies.append(("random_sphere", points2))
    
    # Strategy 3: Perturbed Fibonacci
    points3 = fibonacci_sphere(14)
    noise = np.random.normal(0, 0.02, points3.shape)
    points3 = points3 + noise
    points3 = project_to_unit_sphere(points3)
    initial_strategies.append(("perturbed_fibonacci", points3))
    
    # Strategy 4: Icosahedron-based (approximated)
    try:
        # Simple regular polyhedron approach
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1, 0, phi], [1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
            [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
            [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]
        ])
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add two more points for 14 total
        additional = np.array([[0, 0, 1], [0, 0, -1]])  # North and south poles
        points4 = np.vstack([vertices, additional])
        points4 = project_to_unit_sphere(points4)
        initial_strategies.append(("icosahedron", points4))
    except:
        pass

    # Run optimization on each strategy and select the best
    best_points = None
    best_ratio = 0.0
    
    for strategy_name, initial_points in initial_strategies:
        # Apply multi-stage optimization
        optimized_points, optimized_ratio = multi_stage_optimization(initial_points, max_iterations=15000)
        
        # Apply local refinement
        refined_points, refined_ratio = local_refinement(optimized_points, max_iter=500)
        
        if refined_ratio > best_ratio:
            best_ratio = refined_ratio
            best_points = refined_points.copy()
    
    # Final validation and normalization to unit cube [0,1]^3
    if best_points is None:
        # Fallback to Fibonacci initialization
        best_points = fibonacci_sphere(14)
    
    # Ensure points are on unit sphere (should already be true)
    best_points = project_to_unit_sphere(best_points)
    
    # Project to unit cube
    min_coords = np.min(best_points, axis=0)
    max_coords = np.max(best_points, axis=0)
    
    # Handle case where there's no variation
    ranges = max_coords - min_coords
    if np.any(ranges == 0):
        # If any dimension has no variation, return points centered at 0.5
        points_in_cube = np.full_like(best_points, 0.5)
    else:
        # Scale to [0,1] range
        normalized = (best_points - min_coords) / ranges
        # Ensure they're clipped to [0,1]
        points_in_cube = np.clip(normalized, 0, 1)
    
    return points_in_cube

# EVOLVE-BLOCK-END