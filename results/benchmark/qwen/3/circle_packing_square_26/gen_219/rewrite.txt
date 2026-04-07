# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List

class VoronoiLagrangianOptimizer:
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.voronoi_cells = []
        self.voronoi_vertices = []
        self.lagrangian_multipliers = None
        
    def construct_initial_voronoi(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct Voronoi diagram from initial points"""
        try:
            # Add boundary points to avoid edge issues
            boundary_points = [
                [-1, -1], [-1, 2], [2, -1], [2, 2],
                [-0.5, 0.5], [0.5, -0.5], [1.5, 0.5], [0.5, 1.5]
            ]
            extended_points = np.vstack([points, boundary_points])
            
            vor = Voronoi(extended_points)
            return vor.vertices, vor.ridge_points
        except:
            # Fallback to basic Voronoi
            return points, np.arange(len(points))
    
    def compute_voronoi_constraints(self, circles: np.ndarray) -> np.ndarray:
        """Compute Voronoi-based constraints for each circle"""
        # Get circle centers
        centers = circles[:, :2]
        
        # Compute Voronoi diagram of circle centers
        voronoi_vertices, ridge_points = self.construct_initial_voronoi(centers)
        
        # For each circle, determine its Voronoi region constraints
        constraints = []
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Boundary constraints (ensure circle fits within unit square)
            boundary_constraint = min([
                x - r,  # Left boundary
                1 - x - r,  # Right boundary
                y - r,  # Bottom boundary
                1 - y - r  # Top boundary
            ])
            
            constraints.append(boundary_constraint)
            
        return np.array(constraints)
    
    def lagrangian_objective(self, params: np.ndarray, circles: np.ndarray, 
                           lambda_mult: float = 100.0) -> float:
        """Objective function with Lagrangian relaxation"""
        # Reshape parameters back to circles
        n_circles = len(circles)
        positions = params[:n_circles*2].reshape((-1, 2))
        radii = params[n_circles*2:]
        
        # Sum of radii (to maximize)
        objective_value = -np.sum(radii)  # Negative because we minimize
        
        # Add penalty for constraint violations
        penalty = 0.0
        
        # Boundary constraints
        for i in range(n_circles):
            x, y = positions[i]
            r = radii[i]
            # Penalty for going outside boundaries
            if x - r < 0:
                penalty += lambda_mult * (x - r)**2
            if x + r > 1:
                penalty += lambda_mult * (x + r - 1)**2
            if y - r < 0:
                penalty += lambda_mult * (y - r)**2
            if y + r > 1:
                penalty += lambda_mult * (y + r - 1)**2
                
        # Overlap constraints (soft constraints)
        for i in range(n_circles):
            x1, y1 = positions[i]
            r1 = radii[i]
            for j in range(i+1, n_circles):
                x2, y2 = positions[j]
                r2 = radii[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                overlap = max(0, r1 + r2 - distance)
                penalty += lambda_mult * overlap**2
                
        return objective_value + penalty
    
    def update_circles_with_voronoi(self, circles: np.ndarray) -> np.ndarray:
        """Improve circle configuration using Voronoi-based geometric insights"""
        updated_circles = circles.copy()
        
        # Compute Voronoi diagrams of current circle centers
        centers = circles[:, :2]
        try:
            vor = Voronoi(centers)
            vertices = vor.vertices
        except:
            # Fallback - just use direct approach
            return circles
            
        # For each circle, adjust based on Voronoi cell geometry
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Find Voronoi cell for this point
            # Simplified approach: adjust position to be more central in Voronoi region
            
            # Constraint-based adjustments
            # Keep within boundary limits
            max_radius = min(x, 1-x, y, 1-y)
            if r > max_radius:
                updated_circles[i, 2] = max_radius * 0.95  # Slightly reduce radius
            
            # Improve positioning based on neighbors
            neighbor_distances = []
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    neighbor_distances.append(dist)
            
            if neighbor_distances:
                avg_neighbor_dist = np.mean(neighbor_distances)
                # Adjust radius to maintain spacing
                ideal_radius = avg_neighbor_dist * 0.4  # Maintain some spacing
                if ideal_radius < max_radius:
                    updated_circles[i, 2] = min(ideal_radius, updated_circles[i, 2])
        
        return updated_circles
    
    def optimize_single_step(self, circles: np.ndarray, iterations: int = 50) -> np.ndarray:
        """Optimize circles using geometric and Lagrangian insights"""
        optimized = circles.copy()
        
        # Use a combination of local geometric adjustments and Voronoi-based refinement
        for iter_num in range(iterations):
            # Store old configuration
            old_circles = optimized.copy()
            
            # Local optimization step
            for i in range(len(optimized)):
                x, y, r = optimized[i]
                
                # Compute maximum radius based on current neighbors and boundaries
                max_radius = min(x, 1-x, y, 1-y)
                
                # Consider neighbors' influence
                for j in range(len(optimized)):
                    if i != j:
                        x2, y2, r2 = optimized[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance > 0:
                            # Adjust radius to maintain proper spacing
                            max_radius = min(max_radius, distance - r2)
                
                # Keep within safe bounds
                optimized[i, 2] = min(max(0.001, max_radius * 0.95), r)
                
                # Adjust position to maintain good Voronoi-like distribution
                if iter_num % 10 == 0:  # Periodic repositioning
                    # Move towards better Voronoi-centered position
                    # Simple heuristic: move to center of mass of neighbors within radius
                    neighbors = []
                    for j in range(len(optimized)):
                        if i != j:
                            x2, y2, r2 = optimized[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < 0.5:  # Within some neighborhood
                                neighbors.append((x2, y2))
                    
                    if neighbors:
                        avg_x = np.mean([n[0] for n in neighbors])
                        avg_y = np.mean([n[1] for n in neighbors])
                        # Move toward average neighbor position
                        optimized[i, 0] = 0.8 * x + 0.2 * avg_x
                        optimized[i, 1] = 0.8 * y + 0.2 * avg_y
                        
                        # Clamp to unit square
                        optimized[i, 0] = np.clip(optimized[i, 0], 0.01, 0.99)
                        optimized[i, 1] = np.clip(optimized[i, 1], 0.01, 0.99)
            
            # Ensure no overlaps
            for i in range(len(optimized)):
                for j in range(i+1, len(optimized)):
                    x1, y1, r1 = optimized[i]
                    x2, y2, r2 = optimized[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if dist < (r1 + r2):
                        # Resolve overlap by moving circles apart
                        dx = x2 - x1
                        dy = y2 - y1
                        if dist > 0:
                            dx = dx / dist
                            dy = dy / dist
                            
                            move_amount = (r1 + r2 - dist) / 2.0
                            optimized[i, 0] -= dx * move_amount * 0.3
                            optimized[i, 1] -= dy * move_amount * 0.3
                            optimized[j, 0] += dx * move_amount * 0.3
                            optimized[j, 1] += dy * move_amount * 0.3
            
            # Clamp to valid ranges
            for i in range(len(optimized)):
                x, y, r = optimized[i]
                optimized[i, 0] = np.clip(x, r + 0.01, 1 - r - 0.01)
                optimized[i, 1] = np.clip(y, r + 0.01, 1 - r - 0.01)
                optimized[i, 2] = max(0.001, min(0.49, r))
        
        return optimized

def generate_initial_config(n_circles: int) -> np.ndarray:
    """Generate an initial good configuration using spatial partitioning"""
    # Create a hexagonal-like pattern for initial distribution
    circles = np.zeros((n_circles, 3))
    
    # Try to place circles in a quasi-hexagonal grid
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Base radius
    base_radius = min(spacing_x, spacing_y) * 0.3
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
                
            # Offset odd rows for better packing
            x_offset = spacing_x * (j + 1)
            if i % 2 == 1:
                x_offset += spacing_x * 0.5
                
            x = np.clip(x_offset, base_radius, 1 - base_radius)
            y = np.clip(spacing_y * (i + 1), base_radius, 1 - base_radius)
            
            # Adjust radius based on proximity to edges
            max_radius = min(x, 1-x, y, 1-y)
            radius = min(base_radius, max_radius)
            
            circles[idx] = [x, y, radius]
            idx += 1
    
    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    np.random.seed(42)
    
    n_circles = 26
    
    # Generate initial configuration
    circles = generate_initial_config(n_circles)
    
    # Create optimizer
    optimizer = VoronoiLagrangianOptimizer(n_circles)
    
    # Iterative optimization with Voronoi-Lagrangian approach
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Multiple optimization stages
    for stage in range(3):
        # Stage 1: Rough layout optimization
        if stage == 0:
            optimized_circles = optimizer.optimize_single_step(circles, iterations=30)
        # Stage 2: Fine-tuning
        elif stage == 1:
            optimized_circles = optimizer.optimize_single_step(circles, iterations=50)
        # Stage 3: Final refinement
        else:
            optimized_circles = optimizer.optimize_single_step(circles, iterations=70)
        
        # Apply Voronoi improvement
        improved_circles = optimizer.update_circles_with_voronoi(optimized_circles)
        
        # Check if this is better
        current_sum = np.sum(improved_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = improved_circles.copy()
        
        # Continue with improved configuration
        circles = improved_circles.copy()
    
    # Final validation and cleanup
    final_circles = best_circles.copy()
    
    # Ensure all circles are valid
    for i in range(n_circles):
        x, y, r = final_circles[i]
        
        # Check boundary constraints
        max_radius = min(x, 1-x, y, 1-y)
        final_circles[i, 2] = min(r, max_radius * 0.95)
        
        # Re-clamp positions to be within valid boundaries
        final_circles[i, 0] = np.clip(x, final_circles[i, 2] + 0.01, 1 - final_circles[i, 2] - 0.01)
        final_circles[i, 1] = np.clip(y, final_circles[i, 2] + 0.01, 1 - final_circles[i, 2] - 0.01)
    
    # Final overlap resolution
    for _ in range(10):
        resolved = False
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1, r1 = final_circles[i]
                x2, y2, r2 = final_circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if dist < (r1 + r2):
                    # Resolve overlap by moving apart
                    dx = x2 - x1
                    dy = y2 - y1
                    if dist > 0:
                        dx = dx / dist
                        dy = dy / dist
                    
                    move_amount = (r1 + r2 - dist) / 2.0
                    final_circles[i, 0] -= dx * move_amount * 0.2
                    final_circles[i, 1] -= dy * move_amount * 0.2
                    final_circles[j, 0] += dx * move_amount * 0.2
                    final_circles[j, 1] += dy * move_amount * 0.2
                    resolved = True
        
        if not resolved:
            break
    
    return final_circles

# EVOLVE-BLOCK-END