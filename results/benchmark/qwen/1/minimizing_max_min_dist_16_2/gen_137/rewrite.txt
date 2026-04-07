# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_voronoi_fitness(points):
        """Calculate fitness based on Voronoi cell properties for better distribution."""
        try:
            vor = Voronoi(points)
            areas = []
            
            # Calculate Voronoi cell areas for each point
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 in region or len(region) < 3:
                    # Skip invalid regions
                    continue
                    
                vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                if len(vertices) >= 3:
                    # Calculate area using shoelace formula
                    n = len(vertices)
                    area = 0.5 * abs(sum(vertices[i][0] * vertices[(i+1)%n][1] - 
                                        vertices[(i+1)%n][0] * vertices[i][1] 
                                        for i in range(n)))
                    areas.append(area)
            
            if not areas:
                return 0.0
                
            # Ratio of min/max cell areas (higher is better for uniformity)
            min_area = min(areas)
            max_area = max(areas)
            
            if max_area == 0:
                return 0.0
                
            return min_area / max_area
            
        except Exception:
            return 0.0

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance."""
        if len(points) < 2:
            return 0.0

        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def initialize_voronoi_structured_points():
        """Generate initial points using Voronoi-structured patterns."""
        np.random.seed(42)
        
        # Generate diverse starting configurations based on Voronoi principles
        configurations = []
        
        # 1. Hexagonal grid with perturbation
        hex_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                # Add controlled randomness
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                hex_points.append([x, y])
        hex_points = np.array(hex_points[:16])
        configurations.append(hex_points)
        
        # 2. Fibonacci spiral pattern
        fib_points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(16):
            theta = 2 * np.pi * i / phi
            r = np.sqrt(i / 15) if i > 0 else 0.5
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            # Add noise
            x += np.random.normal(0, 0.005)
            y += np.random.normal(0, 0.005)
            fib_points.append([x, y])
        configurations.append(np.array(fib_points))
        
        # 3. Random initialization with boundary awareness
        random_points = np.random.rand(16, 2)
        configurations.append(random_points)
        
        # Evaluate each configuration and select the best one
        best_config = configurations[0]
        best_fitness = calculate_voronoi_fitness(best_config)
        
        for config in configurations[1:]:
            fitness = calculate_voronoi_fitness(config)
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = config
                
        return best_config

    def voronoi_relax_and_optimize(points, max_iterations=50):
        """Enhanced Voronoi relaxation with gradient-based optimization."""
        current_points = points.copy()
        
        for iteration in range(max_iterations):
            try:
                # Compute Voronoi diagram
                vor = Voronoi(current_points)
                
                # Calculate new positions using Voronoi centroids
                new_points = np.zeros_like(current_points)
                converged = True
                
                for i in range(len(current_points)):
                    # Get Voronoi cell vertices
                    region = vor.regions[vor.point_region[i]]
                    
                    if -1 in region or len(region) < 3:
                        # Handle unbounded regions with boundary reflection
                        new_points[i] = current_points[i] * 0.99 + np.random.normal(0, 0.001, 2)
                        continue
                    
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                    if len(vertices) < 3:
                        new_points[i] = current_points[i]
                        continue
                    
                    # Compute centroid of Voronoi cell
                    centroid = np.mean(vertices, axis=0)
                    
                    # Apply boundary constraints (clip to [0.001, 0.999] to prevent edge effects)
                    centroid = np.clip(centroid, 0.001, 0.999)
                    
                    # Store new point
                    new_points[i] = centroid
                    
                    # Check convergence
                    if np.linalg.norm(new_points[i] - current_points[i]) > 1e-6:
                        converged = False
                
                # Apply momentum for smoother convergence
                momentum_factor = 0.8
                current_points = current_points * (1 - momentum_factor) + new_points * momentum_factor
                
                # Ensure points stay within bounds
                current_points = np.clip(current_points, 0, 1)
                
                # Early stopping if converged
                if converged:
                    break
                    
            except Exception:
                # Fallback to simple perturbation
                current_points += np.random.normal(0, 0.001, current_points.shape)
                current_points = np.clip(current_points, 0, 1)
        
        return current_points

    def objective_function(points_flat):
        """Objective function to minimize (negative ratio to maximize ratio)."""
        points = points_flat.reshape(-1, 2)
        # Focus on minimizing the negative of distance ratio
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -1e10
            
        return -min_dist / max_dist

    def adaptive_local_optimization(initial_points, max_attempts=3):
        """Apply adaptive local optimization to improve point distribution."""
        best_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(best_points)
        
        # Multiple optimization strategies
        strategies = [
            {'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}},
            {'method': 'SLSQP', 'options': {'maxiter': 80, 'ftol': 1e-10}},
            {'method': 'TNC', 'options': {'maxiter': 100, 'ftol': 1e-11}}
        ]
        
        for attempt in range(max_attempts):
            # Add slight perturbation for diversity
            if attempt > 0:
                perturbation = np.random.normal(0, 0.005, best_points.shape)
                perturbed = best_points + perturbation
                perturbed = np.clip(perturbed, 0, 1)
            else:
                perturbed = best_points.copy()
                
            for strategy in strategies:
                try:
                    result = minimize(
                        objective_function,
                        perturbed.flatten(),
                        method=strategy['method'],
                        bounds=[(0, 1) for _ in range(len(perturbed.flatten()))],
                        options=strategy['options']
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 2)
                        optimized_points = np.clip(optimized_points, 0, 1)
                        ratio = calculate_min_max_ratio(optimized_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except Exception:
                    continue
                    
        return best_points, best_ratio

    # Main algorithm flow
    # Phase 1: Initialize with Voronoi-structured configurations
    initial_points = initialize_voronoi_structured_points()
    
    # Phase 2: Voronoi relaxation for global improvement
    relaxed_points = voronoi_relax_and_optimize(initial_points, max_iterations=30)
    relaxed_ratio = calculate_min_max_ratio(relaxed_points)
    
    # Phase 3: Local optimization to refine solution
    final_points, final_ratio = adaptive_local_optimization(relaxed_points)
    
    # Phase 4: Additional Voronoi refinement if beneficial
    if final_ratio < relaxed_ratio * 0.99:  # Only if significant improvement expected
        additional_refinement = voronoi_relax_and_optimize(final_points, max_iterations=20)
        additional_ratio = calculate_min_max_ratio(additional_refinement)
        
        if additional_ratio > final_ratio:
            final_points = additional_refinement
            final_ratio = additional_ratio
    
    # Phase 5: Final adaptive local optimization
    final_points, final_ratio = adaptive_local_optimization(final_points)
    
    return final_points

# EVOLVE-BLOCK-END