# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Implements a Voronoi-guided evolutionary approach with multi-scale optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into points
        points = x.reshape(-1, 2)

        # Compute pairwise distances using squareform for better numerical stability
        distances = squareform(pdist(points))

        # Zero out diagonal elements (distance to self)
        np.fill_diagonal(distances, np.inf)

        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing the negative)
        if d_max == 0:
            return -1.0
        return -d_min / d_max

    def constraint_func(x):
        # Ensure points are within [0.01, 0.99] x [0.01, 0.99] for numerical stability
        points = x.reshape(-1, 2)
        constraints = []

        # x coordinates in [0.01, 0.99]
        constraints.append(points[:, 0].min() - 0.01)  # x_min >= 0.01
        constraints.append(0.99 - points[:, 0].max())  # x_max <= 0.99

        # y coordinates in [0.01, 0.99]
        constraints.append(points[:, 1].min() - 0.01)  # y_min >= 0.01
        constraints.append(0.99 - points[:, 1].max())  # y_max <= 0.99

        return np.array(constraints)

    def voronoi_based_penalty(x, penalty_weight=1000.0):
        """
        Enhanced penalty function that leverages Voronoi diagram analysis for better spatial distribution
        """
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        d_min = np.min(distances)
        d_max = np.max(distances)

        # If all points are identical or near identical, penalize heavily
        if d_max == 0:
            return -1.0

        # Calculate base ratio to maximize
        ratio = d_min / d_max

        # Boundary penalty - enforce points stay away from edges
        boundary_penalty = 0.0
        margin = 0.01
        for point in points:
            if (point[0] < margin or point[0] > 1-margin or
                point[1] < margin or point[1] > 1-margin):
                boundary_penalty += penalty_weight * (margin - min(point[0], 1-point[0], point[1], 1-point[1]))

        # Clustering penalty - penalize very small distances
        min_distance_penalty = 0.0
        if d_min < 0.05:
            min_distance_penalty = penalty_weight * (0.05 - d_min)

        # Voronoi-based uniformity penalty - analyze how evenly points are distributed
        voronoi_uniformity_penalty = 0.0
        try:
            # Compute Voronoi diagram for current configuration
            vor = Voronoi(points)
            
            # Calculate areas of Voronoi cells
            cell_areas = []
            for i, region in enumerate(vor.regions):
                if len(region) > 0 and -1 not in region:  # Skip infinite regions
                    vertices = vor.vertices[region]
                    if len(vertices) >= 3:
                        # Calculate polygon area using shoelace formula
                        x_coords = vertices[:, 0]
                        y_coords = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
                        cell_areas.append(area)
            
            # If we have valid cell areas, compute penalty based on variance
            if len(cell_areas) > 0:
                mean_area = np.mean(cell_areas)
                std_area = np.std(cell_areas)
                if mean_area > 0:
                    # Penalize high variance in cell areas (unequal distribution)
                    voronoi_uniformity_penalty = penalty_weight * (std_area / mean_area)
                    
        except Exception:
            # If Voronoi computation fails, skip this penalty
            pass

        total_penalty = boundary_penalty + min_distance_penalty + voronoi_uniformity_penalty
        return -(ratio - total_penalty / penalty_weight)

    def create_voronoi_guided_initialization():
        """
        Create initial configuration using Voronoi-guided approach:
        Start with a structured pattern, then iteratively refine using Voronoi analysis
        """
        np.random.seed(42)
        
        # Start with a simple hexagonal grid pattern
        points = []
        rows, cols = 4, 4
        sqrt3 = np.sqrt(3)
        spacing = 0.8
        row_spacing = spacing / sqrt3
        col_spacing = spacing
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                
                # Scale to fit within [0.05, 0.95] range
                x_scaled = 0.05 + (x / (col_spacing * cols)) * 0.9
                y_scaled = 0.05 + (y / (row_spacing * rows)) * 0.9
                
                points.append([x_scaled, y_scaled])
        
        points = np.array(points[:16])
        
        # Add strategic perturbations
        for i in range(len(points)):
            points[i] += np.random.normal(0, 0.01, 2)
        
        points = np.clip(points, 0.05, 0.95)
        
        # Apply iterative refinement using Voronoi analysis
        for _ in range(10):
            try:
                # Create Voronoi diagram
                vor = Voronoi(points)
                
                # Calculate centroids of Voronoi regions
                new_points = []
                for i in range(len(points)):
                    region_indices = np.where(vor.point_region == i)[0]
                    if len(region_indices) > 0:
                        region_id = region_indices[0]
                        vertices = vor.vertices[vor.region[region_id]]
                        
                        if len(vertices) >= 3:
                            # Compute centroid of Voronoi region
                            centroid = np.mean(vertices, axis=0)
                            
                            # Only move if there's significant difference
                            if np.linalg.norm(centroid - points[i]) > 1e-6:
                                # Move point towards centroid with damping
                                direction = centroid - points[i]
                                step_size = 0.05 * np.linalg.norm(direction)
                                new_point = points[i] + step_size * direction / (np.linalg.norm(direction) + 1e-10)
                                new_point = np.clip(new_point, 0.05, 0.95)
                                new_points.append(new_point)
                            else:
                                new_points.append(points[i])
                        else:
                            new_points.append(points[i])
                    else:
                        new_points.append(points[i])
                
                points = np.array(new_points)
                
                # Add small random jitter to avoid getting stuck
                points += np.random.normal(0, 0.005, points.shape)
                points = np.clip(points, 0.05, 0.95)
                
            except Exception:
                # If Voronoi computation fails, just add random jitter
                points += np.random.normal(0, 0.005, points.shape)
                points = np.clip(points, 0.05, 0.95)
        
        return points

    def generate_diverse_initial_configs():
        """Generate multiple diverse initial configurations"""
        configs = []
        
        # 1. Voronoi-guided hexagonal pattern
        configs.append(create_voronoi_guided_initialization())
        
        # 2. Fibonacci spiral pattern
        np.random.seed(123)
        points = []
        for i in range(16):
            if i == 0:
                points.append([0.5, 0.5])
            else:
                angle = i * 2.4
                radius = 0.4 * np.sqrt(i / 15.0) if i > 0 else 0.05
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0.05, 0.95)
        configs.append(points)
        
        # 3. Regular grid with offset rows (more sophisticated than previous versions)
        np.random.seed(456)
        points = np.zeros((16, 2))
        rows, cols = 4, 4
        row_spacing = 0.9 / (rows - 1) if rows > 1 else 0.9
        col_spacing = 0.9 / (cols - 1) if cols > 1 else 0.9
        
        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= 16:
                    break
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                points[i * cols + j] = [x + np.random.normal(0, 0.005), y + np.random.normal(0, 0.005)]
        
        points = np.clip(points, 0.05, 0.95)
        configs.append(points)
        
        # 4. Random configuration with good distribution properties
        np.random.seed(789)
        points = np.random.rand(16, 2) * 0.9 + 0.05
        configs.append(points)
        
        # 5. Corner and center configuration
        np.random.seed(321)
        corner_points = np.array([
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
        ])
        remaining_points = np.random.rand(8, 2) * 0.7 + 0.15
        points = np.vstack([corner_points, remaining_points])
        points = np.clip(points, 0.05, 0.95)
        configs.append(points)
        
        return configs

    def adaptive_local_optimization(x0, bounds, max_iter=500):
        """
        Perform adaptive local optimization with fallback strategies
        """
        # Try L-BFGS-B first (often best for smooth problems)
        try:
            result = minimize(
                voronoi_based_penalty,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.success:
                return result
            
        except Exception:
            pass
        
        # Fallback to SLSQP
        try:
            result = minimize(
                voronoi_based_penalty,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                return result
                
        except Exception:
            pass
        
        # Final fallback to Nelder-Mead with reduced iterations
        try:
            result = minimize(
                voronoi_based_penalty,
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iter // 4, 'adaptive': True}
            )
            
            return result
            
        except Exception:
            pass
            
        # If all else fails, return original
        return minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_configs()
    
    # Define bounds for each coordinate with boundary padding
    bounds = [(0.05, 0.95) for _ in range(32)]  # 16 points * 2 coordinates each
    
    best_points = None
    best_ratio = float('inf')
    
    # Try multiple initial configurations with hybrid optimization
    for i, initial_config in enumerate(initial_configs):
        try:
            x0 = initial_config.flatten()
            
            # Phase 1: Global optimization with Differential Evolution
            # Use enhanced penalty function for better exploration
            de_result = differential_evolution(
                voronoi_based_penalty,
                bounds,
                seed=42+i,
                maxiter=75,   # Reduce iterations for faster exploration
                popsize=20,   # Moderate population size
                tol=1e-8,
                mutation=(0.7, 1.0),
                recombination=0.8,
                disp=False
            )
            
            # Phase 2: Local refinement with adaptive optimization
            refined_result = adaptive_local_optimization(
                de_result.x, bounds, max_iter=400
            )
            
            # Evaluate the refined result
            current_points = refined_result.x.reshape(-1, 2)
            distances = squareform(pdist(current_points))
            np.fill_diagonal(distances, np.inf)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 0:
                current_ratio = d_min / d_max
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    
        except Exception as e:
            warnings.warn(f"Optimization attempt {i} failed: {e}")
            continue
    
    # If we still don't have a good solution, run a focused refinement
    if best_points is None:
        # Try one last optimization from a good random start
        np.random.seed(999)  # Fixed seed for reproducibility
        x0 = np.random.rand(32) * 0.9 + 0.05  # [0.05, 0.95]
        bounds = [(0.05, 0.95) for _ in range(32)]
        
        # Direct optimization from random start
        result = adaptive_local_optimization(x0, bounds, max_iter=600)
        best_points = result.x.reshape(-1, 2)
    
    # Final refinement with standard objective
    try:
        final_result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0.05, 0.95) for _ in range(32)],
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        best_points = final_result.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0.05, 0.95)
    
    return best_points

# EVOLVE-BLOCK-END