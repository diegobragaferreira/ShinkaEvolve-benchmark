# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import time
from typing import Tuple

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses Voronoi lattice initialization with hybrid optimization approach.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Constants
    N_CIRCLES = 26
    MAX_RADIUS = 0.5
    MIN_RADIUS = 0.001
    BOUNDARY_MARGIN = 0.01
    
    def is_valid_configuration(circles_array: np.ndarray) -> bool:
        """Check if configuration respects all constraints."""
        n = len(circles_array)
        
        # Check containment constraints
        for x, y, r in circles_array:
            if not (r >= MIN_RADIUS and 
                   r <= x <= 1 - r and 
                   r <= y <= 1 - r):
                return False
        
        # Check overlap constraints using efficient nearest neighbor search
        if n > 1:
            positions = circles_array[:, :2]
            radii = circles_array[:, 2]
            
            # Use cKDTree for efficient pairwise distance checking
            tree = cKDTree(positions)
            pairs = tree.query_pairs(2 * MAX_RADIUS)
            
            for i, j in pairs:
                if i < j:  # Only check each pair once
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    r1, r2 = radii[i], radii[j]
                    
                    dist_squared = (x1 - x2)**2 + (y1 - y2)**2
                    radius_sum = r1 + r2
                    
                    if dist_squared < radius_sum**2:
                        return False
                        
        return True
    
    def create_voronoi_lattice_initialization() -> np.ndarray:
        """
        Create initial configuration using Voronoi lattice with adaptive spacing.
        This approach leverages Voronoi's natural uniformity property.
        """
        # Create a regular hexagonal lattice pattern (more efficient than square)
        # Generate points in a triangular/hexagonal grid pattern
        points = []
        
        # Hexagonal lattice parameters
        sqrt3 = np.sqrt(3)
        spacing = 1.0 / 5.0  # Adjust based on number of circles needed
        
        # Generate points in a hexagonal pattern
        for i in range(6):
            for j in range(6):
                # Hexagonal offset pattern
                x = (j + 0.5 * (i % 2)) * spacing
                y = i * spacing * sqrt3 / 2
                if x < 1 and y < 1:
                    points.append([x, y])
        
        points = np.array(points[:N_CIRCLES])
        
        # Add some randomness to avoid perfect patterns that might cause issues
        noise_level = 0.03
        points += np.random.uniform(-noise_level, noise_level, points.shape)
        
        # Ensure points are within bounds
        points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
        
        # Create Voronoi diagram and compute centroids
        try:
            vor = Voronoi(points)
            centroids = vor.points[vor.point_region[:-1]]
            selected_centroids = centroids[:N_CIRCLES]
            
            # Create circles with radius estimation based on Voronoi properties
            circles = np.zeros((N_CIRCLES, 3))
            
            for i in range(N_CIRCLES):
                x, y = selected_centroids[i]
                
                # Calculate minimum distance to neighbors
                distances = np.sqrt(np.sum((selected_centroids - [x, y])**2, axis=1))
                distances = distances[distances > 0]  # Exclude self-distance
                
                if len(distances) > 0:
                    # Use minimum distance to estimate appropriate radius
                    avg_distance = np.min(distances)
                    radius = min(0.2, avg_distance * 0.3)
                else:
                    radius = 0.1
                
                # Ensure radius respects boundary constraints
                radius = min(radius, 
                           x - BOUNDARY_MARGIN, 
                           1 - x - BOUNDARY_MARGIN,
                           y - BOUNDARY_MARGIN, 
                           1 - y - BOUNDARY_MARGIN)
                
                radius = max(MIN_RADIUS, min(MAX_RADIUS, radius))
                
                circles[i] = [x, y, radius]
                
            return circles
            
        except Exception:
            # Fallback to simple grid initialization
            circles = np.zeros((N_CIRCLES, 3))
            grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
            spacing = 1.0 / (grid_size + 1)
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= N_CIRCLES:
                        break
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    r = min(spacing * 0.4, 0.2)
                    circles[idx] = [x, y, r]
                    idx += 1
                    
            return circles
    
    def optimize_via_simulated_annealing(initial_circles: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
        """
        Optimize using simulated annealing approach that balances exploration and exploitation.
        """
        current_solution = initial_circles.copy()
        current_fitness = np.sum(current_solution[:, 2])
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        # Annealing parameters
        temperature = 0.1
        cooling_rate = 0.995
        min_temperature = 0.001
        
        # For each iteration, try perturbing one circle
        for iteration in range(max_iterations):
            # Cool down temperature
            if temperature > min_temperature:
                temperature *= cooling_rate
            
            # Select a random circle to modify
            circle_idx = np.random.randint(0, N_CIRCLES)
            
            # Save backup
            original_pos = current_solution[circle_idx, :2].copy()
            original_rad = current_solution[circle_idx, 2]
            
            # Perturb position and radius
            new_pos = original_pos + np.random.normal(0, 0.005, 2)
            new_rad = original_rad * np.random.normal(1, 0.05)
            
            # Apply bounds
            new_pos[0] = np.clip(new_pos[0], original_rad, 1 - original_rad)
            new_pos[1] = np.clip(new_pos[1], original_rad, 1 - original_rad)
            new_rad = np.clip(new_rad, MIN_RADIUS, MAX_RADIUS)
            
            # Update solution temporarily
            temp_solution = current_solution.copy()
            temp_solution[circle_idx, :2] = new_pos
            temp_solution[circle_idx, 2] = new_rad
            
            # Check if new solution is valid
            if is_valid_configuration(temp_solution):
                # Calculate new fitness
                new_fitness = np.sum(temp_solution[:, 2])
                
                # Accept or reject based on Metropolis criterion
                delta_fitness = new_fitness - current_fitness
                if delta_fitness > 0 or np.random.random() < np.exp(delta_fitness / temperature):
                    current_solution = temp_solution
                    current_fitness = new_fitness
                    
                    # Update best solution
                    if current_fitness > best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
            else:
                # Restore original
                current_solution[circle_idx, :2] = original_pos
                current_solution[circle_idx, 2] = original_rad
        
        return best_solution
    
    def local_optimization_refinement(circles: np.ndarray, max_iterations: int = 500) -> np.ndarray:
        """
        Apply local optimization to maximize radii while maintaining constraints.
        """
        circles = circles.copy()
        
        # Precompute neighborhood information for efficiency
        tree = cKDTree(circles[:, :2]) if len(circles) > 1 else None
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try to increase each circle's radius
            for i in range(len(circles)):
                # Store original state
                original_radius = circles[i, 2]
                original_position = circles[i, :2].copy()
                
                # Calculate maximum possible radius
                max_possible_radius = float('inf')
                
                # Check constraints with other circles
                if tree is not None:
                    # Find nearby circles
                    neighbors = tree.query_ball_point(original_position, 2 * MAX_RADIUS)
                    for j in neighbors:
                        if i != j:
                            x1, y1 = original_position
                            x2, y2 = circles[j, :2]
                            r2 = circles[j, 2]
                            
                            # Distance to center of neighbor
                            dist_to_center = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            
                            # Maximum radius so that circles don't overlap
                            max_radius_with_neighbor = dist_to_center - r2
                            max_possible_radius = min(max_possible_radius, max_radius_with_neighbor)
                
                # Also check boundary constraints
                boundary_radius_x = min(original_position[0], 1 - original_position[0])
                boundary_radius_y = min(original_position[1], 1 - original_position[1])
                max_boundary_radius = min(boundary_radius_x, boundary_radius_y)
                max_possible_radius = min(max_possible_radius, max_boundary_radius)
                
                # If there's room to increase radius
                if max_possible_radius > original_radius:
                    # Try increasing radius step by step
                    step_size = max(0.001, max_possible_radius * 0.05)
                    test_radius = min(original_radius + step_size, max_possible_radius)
                    
                    # Test if we can actually increase radius
                    temp_circles = circles.copy()
                    temp_circles[i, 2] = test_radius
                    
                    # Validate all constraints
                    if is_valid_configuration(temp_circles):
                        circles = temp_circles
                        improved = True
                
            # If no improvement, break early
            if not improved:
                break
                
        return circles
    
    # Main algorithm execution
    try:
        # Step 1: Initialize with Voronoi lattice
        initial_circles = create_voronoi_lattice_initialization()
        
        # Step 2: Optimize using simulated annealing
        optimized_circles = optimize_via_simulated_annealing(initial_circles, 1000)
        
        # Step 3: Refine with local optimization
        final_circles = local_optimization_refinement(optimized_circles, 500)
        
        # Final validation
        if not is_valid_configuration(final_circles):
            # Fallback to simple greedy approach
            final_circles = np.zeros((N_CIRCLES, 3))
            grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
            spacing = 1.0 / (grid_size + 1)
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= N_CIRCLES:
                        break
                    x = (i + 0.5) * spacing
                    y = (j + 0.5) * spacing
                    r = min(spacing * 0.3, 0.2)
                    final_circles[idx] = [x, y, r]
                    idx += 1
                    
            # Fill remaining circles with random positions
            for i in range(idx, N_CIRCLES):
                x = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                y = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                r = np.random.uniform(MIN_RADIUS, MAX_RADIUS)
                final_circles[i] = [x, y, r]
        
        return final_circles
        
    except Exception as e:
        # Last resort fallback
        circles = np.zeros((N_CIRCLES, 3))
        grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= N_CIRCLES:
                    break
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                r = min(spacing * 0.3, 0.2)
                circles[idx] = [x, y, r]
                idx += 1
                
        # Fill remaining circles with random positions
        for i in range(idx, N_CIRCLES):
            x = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            y = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            r = np.random.uniform(MIN_RADIUS, MAX_RADIUS)
            circles[i] = [x, y, r]
            
        return circles

# EVOLVE-BLOCK-END