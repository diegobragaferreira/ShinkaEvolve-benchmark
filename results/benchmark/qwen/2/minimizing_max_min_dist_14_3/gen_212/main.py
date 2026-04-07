# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses Voronoi Lattice Optimization - a structured approach that combines:
    1. Lattice-based initialization for good starting configurations
    2. Physics-inspired force field optimization targeting min/max distance ratio
    3. Adaptive refinement based on convergence behavior
    """
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def calculate_distance_ratio_gradient(points):
        """
        Calculate gradient that pushes points toward configurations with better 
        min/max distance ratios.
        """
        n = len(points)
        if n < 2:
            return np.zeros_like(points)
            
        # Calculate all pairwise distances
        distances = pdist(points)
        distance_matrix = squareform(distances)
        
        # Avoid division by zero
        distance_matrix = np.maximum(distance_matrix, 1e-12)
        
        # Calculate gradient contributions for each point pair
        gradient = np.zeros_like(points)
        
        # For each point, compute how much it contributes to the gradient
        for i in range(n):
            for j in range(i+1, n):
                # Vector from point i to point j
                diff = points[j] - points[i]
                dist = np.linalg.norm(diff)
                
                if dist > 1e-12:
                    # Unit vector
                    unit_vector = diff / dist
                    
                    # Simple gradient contribution based on distance ratio
                    # We want to increase d_min and decrease d_max
                    # So we push points apart when they're too close, pull them closer when too far
                    ratio_contribution = 1.0 / (dist * dist + 1e-6)
                    gradient[i] += ratio_contribution * unit_vector
                    gradient[j] -= ratio_contribution * unit_vector
        
        return gradient

    def initialize_lattice_points(n, grid_resolution=10):
        """
        Initialize points on a structured 3D lattice that projects to sphere.
        This creates a good structured starting configuration.
        """
        # Create 3D lattice points
        lattice_points = []
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                for k in range(grid_resolution):
                    x = i / (grid_resolution - 1) * 2 - 1
                    y = j / (grid_resolution - 1) * 2 - 1  
                    z = k / (grid_resolution - 1) * 2 - 1
                    # Only include points inside unit cube
                    if abs(x) <= 1 and abs(y) <= 1 and abs(z) <= 1:
                        lattice_points.append([x, y, z])
        
        # Filter to get approximately n points and project to sphere
        points = np.array(lattice_points)[:n]
        
        # Normalize to unit sphere
        for i in range(len(points)):
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] = points[i] / norm
                
        # If we don't have enough points, fill with random ones
        if len(points) < n:
            extra_points = np.random.uniform(-1, 1, (n - len(points), 3))
            for i in range(len(extra_points)):
                norm = np.linalg.norm(extra_points[i])
                if norm > 0:
                    extra_points[i] = extra_points[i] / norm
            points = np.vstack([points, extra_points])
            
        return points

    def initialize_fibonacci_sphere(n):
        """Better fibonacci-based sphere initialization"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def initialize_spherical_code():
        """Initialize points using a known good spherical code for 14 points."""
        # Using the vertices of a snub cuboctahedron which provides relatively good distribution
        points = [
            [0.9255, 0.3773, 0.0000],
            [0.3773, 0.0000, 0.9255],
            [0.0000, 0.9255, 0.3773],
            [-0.3773, 0.0000, 0.9255],
            [-0.9255, 0.3773, 0.0000],
            [-0.3773, 0.9255, 0.0000],
            [0.0000, 0.3773, 0.9255],
            [0.3773, -0.9255, 0.0000],
            [0.9255, -0.3773, 0.0000],
            [0.0000, -0.3773, 0.9255],
            [-0.9255, -0.3773, 0.0000],
            [-0.3773, -0.9255, 0.0000],
            [0.0000, -0.9255, -0.3773],
            [0.3773, 0.0000, -0.9255]
        ]
        return np.array(points)

    def project_to_unit_cube(points):
        """Project points to unit cube [0,1]^3"""
        # Find min/max along each axis
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Handle case where there's no variation
        ranges = max_coords - min_coords
        if np.any(ranges == 0):
            # If any dimension has no variation, return points centered at 0.5
            return np.full_like(points, 0.5)

        # Scale to [0,1] range
        normalized = (points - min_coords) / ranges

        # Ensure they're clipped to [0,1]
        return np.clip(normalized, 0, 1)

    def voronoi_entropy_score(points):
        """
        Calculate entropy-based score of Voronoi cell distribution.
        High entropy indicates more uniform cell distribution.
        """
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            # Normalize areas
            areas = areas / np.sum(areas)
            # Entropy calculation
            entropy = -np.sum(areas * np.log(areas + 1e-10))
            return entropy
        except:
            # Fallback for cases where SphericalVoronoi fails
            return 0.0

    def combined_fitness(points):
        """Combined fitness function focusing on distance ratio and uniformity"""
        ratio = calculate_min_max_ratio(points)
        uniformity = voronoi_entropy_score(points)
        # Weight ratio more heavily since that's our primary objective
        return ratio * (1.0 + 0.1 * uniformity)

    def force_based_optimization(initial_points, max_iterations=2000):
        """
        Force-based optimization inspired by physics where points move according to 
        forces that maximize the distance ratio.
        """
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = calculate_min_max_ratio(current_points)
        
        # Track convergence
        last_improvement = 0
        improvement_threshold = 0.0001
        
        # Adaptive learning rate that decreases over time
        learning_rate = 0.05
        
        for iteration in range(max_iterations):
            # Calculate gradient for distance ratio
            gradient = calculate_distance_ratio_gradient(current_points)
            
            # Apply gradient with adaptive learning rate
            current_points = current_points + learning_rate * gradient
            
            # Project back to unit sphere
            for i in range(len(current_points)):
                norm = np.linalg.norm(current_points[i])
                if norm > 0:
                    current_points[i] = current_points[i] / norm
            
            # Calculate new ratio
            new_ratio = calculate_min_max_ratio(current_points)
            
            # Update best solution if improved
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
                last_improvement = iteration
            
            # Adaptive learning rate adjustment
            if iteration > 100:
                # If we haven't improved in a while, decrease learning rate
                if iteration - last_improvement > 50:
                    learning_rate *= 0.95
                    if learning_rate < 0.001:
                        learning_rate = 0.001
            
            # Early stopping based on convergence
            if iteration - last_improvement > 200:
                break
                
        return best_points, best_ratio

    def adaptive_lattice_refinement(initial_points, max_iterations=1500):
        """
        Refine using adaptive lattice-based approach.
        Starts with coarse lattice, then refines with finer grids based on convergence.
        """
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = calculate_min_max_ratio(current_points)
        
        # Start with coarser grid
        grid_resolutions = [8, 12, 16]
        
        for idx, resolution in enumerate(grid_resolutions):
            # Generate new lattice points at this resolution
            lattice_points = initialize_lattice_points(14, resolution)
            
            # Try several different initializations from this lattice
            for attempt in range(3):
                np.random.seed(42 + idx * 100 + attempt * 10)
                
                # Add some randomness to the lattice points
                perturbed_points = lattice_points.copy()
                for i in range(len(perturbed_points)):
                    # Small random perturbation
                    perturbation = np.random.normal(0, 0.02, 3)
                    perturbed_points[i] += perturbation
                    # Project back to sphere
                    norm = np.linalg.norm(perturbed_points[i])
                    if norm > 0:
                        perturbed_points[i] = perturbed_points[i] / norm
                
                # Optimize this configuration
                optimized_points, ratio = force_based_optimization(perturbed_points, max_iterations // 3)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            # Reduce max iterations for subsequent refinements
            max_iterations = max_iterations // 2
            
        return best_points, best_ratio

    def local_gradient_refinement(points, iterations=500):
        """Local refinement using gradient-based optimization"""
        def objective(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit sphere
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 0:
                    points_local[i] = points_local[i] / norm
            return -calculate_min_max_ratio(points_local)  # negative for maximization

        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': iterations, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-6
            )
            refined_points = result.x.reshape(-1, 3)
            # Project back to sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 0:
                    refined_points[i] = refined_points[i] / norm
            return refined_points, -result.fun
        except Exception:
            # Fallback to simple iterative refinement
            current_points = points.copy()
            best_ratio = calculate_min_max_ratio(current_points)
            best_points = current_points.copy()
            
            for _ in range(iterations):
                neighbor_points = current_points.copy()
                point_idx = np.random.randint(len(neighbor_points))
                perturbation = np.random.normal(0, 0.001, 3)
                neighbor_points[point_idx] += perturbation
                
                # Project back to sphere
                norm = np.linalg.norm(neighbor_points[point_idx])
                if norm > 0:
                    neighbor_points[point_idx] = neighbor_points[point_idx] / norm
                    
                new_ratio = calculate_min_max_ratio(neighbor_points)
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = neighbor_points.copy()
                    current_points = neighbor_points.copy()
                    
            return best_points, best_ratio

    # Main execution flow
    np.random.seed(42)
    
    # Strategy 1: Initialize from different sources
    strategies = []
    
    # Try lattice-based initialization
    try:
        lattice_points = initialize_lattice_points(14, 12)
        strategies.append(("lattice", lattice_points))
    except:
        pass
    
    # Try fibonacci initialization
    try:
        fib_points = initialize_fibonacci_sphere(14)
        strategies.append(("fibonacci", fib_points))
    except:
        pass
        
    # Try spherical code
    try:
        code_points = initialize_spherical_code()
        strategies.append(("spherical_code", code_points))
    except:
        pass
    
    # Try multiple random initialization
    for i in range(3):
        np.random.seed(42 + i * 100)
        random_points = np.random.uniform(-1, 1, (14, 3))
        # Normalize to unit sphere
        for j in range(len(random_points)):
            norm = np.linalg.norm(random_points[j])
            if norm > 0:
                random_points[j] = random_points[j] / norm
        strategies.append(("random_" + str(i), random_points))
    
    # Run force-based optimization from each starting point
    best_points = None
    best_ratio = 0.0
    
    for strategy_name, initial_points in strategies:
        try:
            # Apply force-based optimization
            optimized_points, ratio = force_based_optimization(initial_points)
            
            # If we're doing better, update the best
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Apply adaptive lattice refinement to further improve
    try:
        if best_points is not None:
            refined_points, refined_ratio = adaptive_lattice_refinement(best_points)
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()
    except Exception:
        pass
    
    # Final local gradient refinement
    try:
        if best_points is not None:
            final_points, final_ratio = local_gradient_refinement(best_points)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()
    except Exception:
        pass
    
    # Normalize to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(best_points)
    
    return points_in_cube

# EVOLVE-BLOCK-END