# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Voronoi-based geometric optimization approach.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 0:
            return 0
        return d_min / d_max
    
    def create_initial_voronoi_configuration():
        """Create initial configuration based on Voronoi-friendly layout"""
        # Start with hexagonal grid pattern
        points = []
        rows = 4
        cols = 4
        
        spacing_x = 1.0
        spacing_y = np.sqrt(3) / 2

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                points.append([x, y])

        points = np.array(points)
        
        # Normalize to [0,1] x [0,1]
        max_x = (cols - 1) + 0.5
        max_y = (rows - 1) * spacing_y

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y
        
        # Add strategic perturbations to break symmetry and improve Voronoi quality
        noise = np.random.normal(0, 0.01, points.shape)
        
        # Emphasize perturbation on edge points
        edge_indices = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]  # Edge points of 4x4 grid
        noise[edge_indices] *= 1.5
        
        points += noise
        points = np.clip(points, 0, 1)
        
        return points

    def voronoi_energy_penalty(points):
        """Calculate energy penalty based on Voronoi cell properties"""
        if len(points) < 3:
            return 0
            
        try:
            # Compute Voronoi diagram
            vor = Voronoi(points)
            
            # Calculate area variance penalty (more uniform cells are better)
            areas = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    # Compute polygon area
                    vertices = [vor.vertices[i] for i in region if i >= 0]
                    if len(vertices) >= 3:
                        # Simple polygon area calculation
                        polygon_area = 0
                        n = len(vertices)
                        for i in range(n):
                            j = (i + 1) % n
                            polygon_area += vertices[i][0] * vertices[j][1]
                            polygon_area -= vertices[j][0] * vertices[i][1]
                        areas.append(abs(polygon_area) / 2)
            
            if len(areas) == 0:
                return 0
                
            # Penalize high variance in cell areas
            mean_area = np.mean(areas)
            if mean_area == 0:
                return 0
                
            area_variance = np.var(areas)
            area_penalty = area_variance / (mean_area + 1e-8)
            
            return area_penalty
            
        except:
            return 0

    def objective_with_voronoi(points_flat):
        """Combined objective: maximize min/max ratio with Voronoi regularization"""
        points = points_flat.reshape(-1, 2)
        
        # Compute distance ratio
        ratio = compute_min_max_ratio(points)
        
        # Add Voronoi-based penalty to encourage uniform distribution  
        voronoi_penalty = voronoi_energy_penalty(points)
        
        # We want to maximize ratio while minimizing Voronoi penalty
        # Return negative since we minimize in scipy.optimize
        return -(ratio - 0.1 * voronoi_penalty)

    def generate_voronoi_guided_perturbations(points, num_perturbations=8):
        """Generate perturbations that respect Voronoi geometry"""
        # Get Voronoi structure
        try:
            vor = Voronoi(points)
        except:
            return points.copy()
            
        new_points = points.copy()
        
        # For each point, adjust based on its Voronoi cell properties
        for i in range(len(points)):
            # Get neighbors based on Voronoi connectivity
            neighbors = []
            
            # Find neighboring points via Voronoi relationships
            for j in range(len(points)):
                if i != j:
                    neighbors.append(j)
                    
            # If we have neighbors, move towards more uniform distribution
            if len(neighbors) > 0 and np.random.rand() < 0.3:
                # Move point to increase average Voronoi cell area
                # Simple heuristic: move towards center of mass of neighbors
                neighbor_points = points[neighbors]
                center_of_mass = np.mean(neighbor_points, axis=0)
                
                # Move point slightly towards center of mass
                direction = center_of_mass - points[i]
                magnitude = 0.01 * np.linalg.norm(direction)
                if magnitude > 0:
                    direction = direction / magnitude
                    
                # Apply perturbation with reduced magnitude for stability
                new_points[i] += direction * 0.005
                
        return new_points

    def voronoi_based_optimization(initial_points, max_iter=2000):
        """Optimize using Voronoi-guided approach"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Better cooling schedule
        T = 0.3
        cooling_rate = 0.999
        min_temp = 1e-6
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Store recent improvements
        recent_improvements = []
        improvement_window = 30
        
        for iteration in range(max_iter):
            # Adaptive cooling
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-6:
                    T *= 0.95
                elif avg_improvement > 1e-4:
                    T = min(T * 1.02, 0.5)  # Warm up occasionally
            
            T *= cooling_rate
            
            if T < min_temp:
                break
            
            # Generate multiple perturbations and select the best
            best_new_points = current_points.copy()
            best_new_ratio = current_ratio
            
            # Try several perturbation strategies
            for _ in range(5):
                new_points = current_points.copy()
                
                # Strategy 1: Random perturbation
                if np.random.rand() < 0.5:
                    idx = np.random.randint(len(current_points))
                    perturbation_magnitude = T * 0.05
                    new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                    new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)
                else:
                    # Strategy 2: Voronoi-guided perturbation
                    new_points = generate_voronoi_guided_perturbations(current_points, 2)
                
                # Enforce boundaries
                new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
                new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)
                
                # Evaluate
                new_ratio = compute_min_max_ratio(new_points)
                
                if new_ratio > best_new_ratio:
                    best_new_ratio = new_ratio
                    best_new_points = new_points.copy()
            
            # Accept or reject with Metropolis criterion
            if best_new_ratio > current_ratio or np.random.rand() < np.exp((best_new_ratio - current_ratio) / T):
                current_points = best_new_points
                current_ratio = best_new_ratio
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
            
            # Track improvements
            if best_new_ratio > current_ratio:
                recent_improvements.append(best_new_ratio - current_ratio)
                if len(recent_improvements) > improvement_window * 2:
                    recent_improvements.pop(0)
        
        return best_points, best_ratio

    def hybrid_optimization(initial_points):
        """Combine different optimization strategies"""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Strategy 1: Voronoi-based optimization
        try:
            voronoi_points, voronoi_ratio = voronoi_based_optimization(initial_points, max_iter=1500)
            if voronoi_ratio > best_ratio:
                best_ratio = voronoi_ratio
                best_points = voronoi_points.copy()
        except:
            pass
        
        # Strategy 2: Gradient-based refinement with scipy
        try:
            # Flatten points for scipy optimization
            x0 = best_points.flatten()
            
            # Use L-BFGS-B for constrained optimization
            result = minimize(
                objective_with_voronoi,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            # Extract result
            refined_points = result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            refined_ratio = compute_min_max_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()
        except:
            pass
        
        return best_points, best_ratio

    # Generate multiple diverse initial configurations
    initial_configs = []
    
    # Configuration 1: Enhanced hexagonal grid (base case)
    initial_configs.append(create_initial_voronoi_configuration())
    
    # Configuration 2: Random with constraints
    np.random.seed(42)
    initial_configs.append(np.random.rand(16, 2))
    
    # Configuration 3: Perturbed grid
    grid_points = create_initial_voronoi_configuration()
    np.random.seed(43)
    perturbations = np.random.normal(0, 0.005, (16, 2))
    initial_configs.append(np.clip(grid_points + perturbations, 0, 1))
    
    # Configuration 4: Triangular lattice
    triangular_points = []
    rows = 4
    cols = 4
    spacing_x = 1.0
    spacing_y = np.sqrt(3)/2

    for i in range(rows):
        for j in range(cols):
            x = j * spacing_x + (i % 2) * spacing_x * 0.5
            y = i * spacing_y
            triangular_points.append([x, y])

    triangular_points = np.array(triangular_points)
    # Normalize triangular lattice
    max_x = (cols - 1) + 0.5
    max_y = (rows - 1) * spacing_y
    triangular_points[:, 0] = triangular_points[:, 0] / max_x
    triangular_points[:, 1] = triangular_points[:, 1] / max_y
    initial_configs.append(np.clip(triangular_points[:16], 0, 1))
    
    # Run optimization from each configuration
    best_ratio = -np.inf
    best_points = None
    
    for i, initial_config in enumerate(initial_configs):
        try:
            # Clip initial points
            initial_config = np.clip(initial_config, 0, 1)
            
            # Hybrid optimization
            optimized_points, ratio = hybrid_optimization(initial_config)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Final refinement if needed
    if best_points is not None:
        # Do one final optimization with gradient descent
        try:
            x0 = best_points.flatten()
            result = minimize(
                objective_with_voronoi,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 300, 'ftol': 1e-9}
            )
            
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            final_ratio = compute_min_max_ratio(final_points)
            
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()
                
        except:
            pass
    
    # Fallback to initial configuration if nothing works
    if best_points is None:
        best_points = create_initial_voronoi_configuration()
    
    return best_points

# EVOLVE-BLOCK-END