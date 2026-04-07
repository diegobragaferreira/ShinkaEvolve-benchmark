# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional

class VoronoiPointOptimizer:
    """Optimizes point distribution using Voronoi-based geometric principles."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        
    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio along with actual values."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0, min_dist, max_dist
            
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist
    
    def voronoi_energy(self, points: np.ndarray) -> float:
        """Compute Voronoi-based energy that penalizes poor point distributions."""
        try:
            vor = Voronoi(points)
            energy = 0.0
            
            # Calculate energy based on Voronoi cell properties
            for i, region in enumerate(vor.point_region):
                if region != len(vor.regions) - 1:  # Skip infinite regions
                    vertices = vor.regions[region]
                    if len(vertices) > 0 and -1 not in vertices:
                        # Calculate area of Voronoi cell
                        cell_vertices = vor.vertices[vertices]
                        if len(cell_vertices) >= 3:
                            # Simple polygon area calculation
                            area = 0.0
                            n = len(cell_vertices)
                            for j in range(n):
                                k = (j + 1) % n
                                area += cell_vertices[j][0] * cell_vertices[k][1]
                                area -= cell_vertices[k][0] * cell_vertices[j][1]
                            area = abs(area) / 2.0
                            
                            # Penalize very small or very large cells
                            # This encourages more uniform cell sizes
                            energy += 1.0 / (area + 1e-8)
                            
            return energy
        except:
            return float('inf')
    
    def geometric_objective(self, points: np.ndarray) -> float:
        """Combined objective that balances distance ratio with Voronoi quality."""
        ratio, min_dist, max_dist = self.calculate_ratio(points)
        
        # The main objective is to maximize ratio
        # We also include a penalty for poor Voronoi tessellation
        voronoi_penalty = self.voronoi_energy(points)
        
        # Combine objectives - minimize negative ratio + some Voronoi penalty
        # The Voronoi penalty helps avoid degenerate configurations
        return -(ratio - 0.001 * min(voronoi_penalty, 1000.0))
    
    def generate_initial_voronoi_config(self) -> np.ndarray:
        """Generate initial configuration with good Voronoi properties."""
        points = []
        
        # Start with a structured approach but with geometric awareness
        # Use a modified hexagonal grid with special corner treatment
        
        # Create a 4x4 grid with offset rows for better Voronoi properties
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Slightly adjust spacing to encourage better Voronoi cells
        spacing_x *= 0.9
        spacing_y *= 0.9
        
        for i in range(rows):
            for j in range(cols):
                # Offset every other row to form better Voronoi patterns
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Ensure bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        # Select exactly 16 points (should already be 16)
        return np.array(points[:16])
    
    def generate_geometrically_balanced(self) -> np.ndarray:
        """Generate points that are balanced geometrically to avoid clustering."""
        points = []
        
        # Place points in a way that maximizes minimum separation
        # Start with a simple but structured approach
        
        # Use a combination of grid and radial placement
        # Grid points first
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                grid_points.append([x, y])
        
        # Add some strategic radial points
        radial_points = []
        radii = [0.3, 0.6]
        angles = [math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4, 2*math.pi]
        
        for r in radii:
            for angle in angles:
                if len(radial_points) >= 16 - 16:  # Reserve some for grid
                    break
                x = 0.5 + r * math.cos(angle)
                y = 0.5 + r * math.sin(angle)
                if 0.001 <= x <= 0.999 and 0.001 <= y <= 0.999:
                    radial_points.append([x, y])
        
        points = grid_points[:16]  # Just use grid for simplicity and ensure 16 points
        points = np.array(points)
        
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.01, points.shape)
        points += perturbation
        points = np.clip(points, 0.001, 0.999)
        
        return points
    
    def adaptive_local_search(self, points: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Perform adaptive local search with geometric constraints."""
        current_points = points.copy()
        best_points = points.copy()
        best_ratio, _, _ = self.calculate_ratio(best_points)
        
        # For each iteration, try to improve the solution
        for i in range(max_iter):
            # Store current state
            current_points = best_points.copy()
            
            # Try optimizations of individual points
            # Use a simple coordinate-wise optimization approach
            for j in range(len(current_points)):
                # Save original point
                original_point = current_points[j].copy()
                
                # Try different small perturbations
                best_improvement = 0.0
                best_new_point = original_point.copy()
                
                # Test different small moves
                for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                        test_point = original_point.copy()
                        test_point[0] += dx
                        test_point[1] += dy
                        
                        # Clip to bounds
                        test_point[0] = np.clip(test_point[0], 0.001, 0.999)
                        test_point[1] = np.clip(test_point[1], 0.001, 0.999)
                        
                        # Temporarily update this point
                        temp_points = current_points.copy()
                        temp_points[j] = test_point
                        
                        # Check if this gives better ratio
                        ratio, _, _ = self.calculate_ratio(temp_points)
                        improvement = ratio - best_ratio
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_new_point = test_point.copy()
                
                # Apply the best improvement found
                if best_improvement > 0:
                    current_points[j] = best_new_point
                    ratio, _, _ = self.calculate_ratio(current_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()
            
            # Early stopping if no significant improvement
            if best_improvement <= 1e-8:
                break
        
        return best_points
    
    def geometric_refinement_step(self, points: np.ndarray) -> np.ndarray:
        """Apply geometric refinement that improves Voronoi quality and distance balance."""
        # First, try local search
        refined_points = self.adaptive_local_search(points, max_iter=50)
        
        # Then, apply constrained optimization using scipy's optimizers
        try:
            # Try a few different optimization approaches
            x0 = refined_points.flatten()
            
            # Try L-BFGS-B with tighter tolerances
            result = minimize(
                lambda x: self.geometric_objective(x.reshape(-1, 2)),
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio, _, _ = self.calculate_ratio(optimized_points)
                refined_ratio, _, _ = self.calculate_ratio(refined_points)
                
                if ratio > refined_ratio:
                    return optimized_points
        except:
            pass
            
        return refined_points
    
    def get_best_solution(self) -> np.ndarray:
        """Get the best solution using Voronoi-based optimization."""
        # Generate several different initial configurations with geometric awareness
        initial_configs = []
        
        # Config 1: Voronoi-friendly structured grid
        initial_configs.append(self.generate_initial_voronoi_config())
        
        # Config 2: Geometrically balanced distribution  
        initial_configs.append(self.generate_geometrically_balanced())
        
        # Config 3: Random but well-distributed with constraints
        np.random.seed(42)
        random_points = np.random.uniform(0.001, 0.999, (16, 2))
        initial_configs.append(random_points)
        
        # Config 4: Highly structured grid with random perturbations
        structured_points = self.generate_initial_voronoi_config()
        perturbed = structured_points + np.random.normal(0, 0.02, structured_points.shape)
        perturbed = np.clip(perturbed, 0.001, 0.999)
        initial_configs.append(perturbed)
        
        best_ratio = -np.inf
        best_points = None
        
        # Try each configuration with geometric refinement
        for config in initial_configs:
            # Apply geometric refinement
            refined_points = self.geometric_refinement_step(config)
            
            ratio, _, _ = self.calculate_ratio(refined_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        
        # Do final refinement on the best
        if best_points is not None:
            final_points = self.geometric_refinement_step(best_points)
            final_ratio, _, _ = self.calculate_ratio(final_points)
            
            if final_ratio > best_ratio:
                return final_points
        
        return best_points if best_points is not None else initial_configs[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer
    optimizer = VoronoiPointOptimizer(16, 2)
    
    # Get best solution using Voronoi-based approach
    best_points = optimizer.get_best_solution()
    
    return best_points

# EVOLVE-BLOCK-END