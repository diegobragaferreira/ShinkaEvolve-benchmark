# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import warnings
from typing import Tuple, Optional, List
warnings.filterwarnings('ignore')

class HexagonPacker:
    """Efficient hexagon packing optimizer with evolutionary approach."""
    
    def __init__(self):
        self.HEX_RADIUS = 1.0
        self.HEX_APO = self.HEX_RADIUS * np.sqrt(3) / 2
        self.TARGET_RATIO = 0.2537
        self.MAX_ITERATIONS = 1000
        
    def generate_hexagon_vertices(self, center_x: float, center_y: float, 
                                angle_degrees: float) -> np.ndarray:
        """Generate vertices of a unit regular hexagon given center and rotation."""
        angle_rad = np.radians(angle_degrees)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = self.HEX_RADIUS * np.cos(theta)
            y = self.HEX_RADIUS * np.sin(theta)
            vertices.append((x, y))
        
        # Transform to global coordinates
        global_verts = []
        for x, y in vertices:
            # Rotate and translate
            rot_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            rot_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            global_verts.append((rot_x + center_x, rot_y + center_y))
            
        return np.array(global_verts)
    
    def create_hexagon_polygon(self, center_x: float, center_y: float, 
                             angle_degrees: float) -> Polygon:
        """Create Shapely polygon representation of a hexagon."""
        vertices = self.generate_hexagon_vertices(center_x, center_y, angle_degrees)
        return Polygon(vertices)
    
    def check_containment(self, hex_poly: Polygon, outer_hex_poly: Polygon) -> bool:
        """Check if hexagon is fully contained within outer hexagon."""
        return outer_hex_poly.contains(hex_poly)
    
    def check_overlap(self, hex1_poly: Polygon, hex2_poly: Polygon) -> bool:
        """Check if two hexagons overlap."""
        return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)
    
    def compute_outer_hexagon_side_length(self, inner_hex_data: np.ndarray) -> float:
        """Compute the minimal side length needed for the outer hexagon to contain all inner hexagons."""
        if len(inner_hex_data) == 0:
            return 1e6
            
        # Find all vertices of all hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle_degrees = inner_hex_data[i]
            vertices = self.generate_hexagon_vertices(center_x, center_y, angle_degrees)
            all_vertices.extend(vertices)
        
        if len(all_vertices) == 0:
            return 1e6
        
        # Calculate bounding rectangle
        min_x = min(v[0] for v in all_vertices)
        max_x = max(v[0] for v in all_vertices)
        min_y = min(v[1] for v in all_vertices)
        max_y = max(v[1] for v in all_vertices)
        
        # Compute required side length for outer hexagon
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        max_dist_sq = 0
        for x, y in all_vertices:
            dist_sq = (x - center_x)**2 + (y - center_y)**2
            max_dist_sq = max(max_dist_sq, dist_sq)
        
        # Side length = sqrt(max_dist) * 2 / sqrt(3)
        return np.sqrt(max_dist_sq) * 2 / np.sqrt(3)
    
    def evaluate_fitness(self, solution: np.ndarray, outer_hex_center=(0, 0), 
                        outer_hex_angle=0) -> Tuple[float, float]:
        """Evaluate fitness of a solution: higher is better."""
        # Reshape solution back to 12 hexagons with (x, y, angle)
        positions = solution.reshape(-1, 3)
        
        # Create polygons for all inner hexagons
        inner_polygons = []
        for i in range(len(positions)):
            center_x, center_y, angle_degrees = positions[i]
            poly = self.create_hexagon_polygon(center_x, center_y, angle_degrees)
            inner_polygons.append(poly)
        
        # Find outer hexagon side length
        side_length = self.compute_outer_hexagon_side_length(positions)
        
        # Check containment and overlaps
        total_penalty = 0
        num_inner = len(inner_polygons)
        
        # Outer hexagon polygon (centered at origin with given angle)
        outer_poly = self.create_hexagon_polygon(outer_hex_center[0], outer_hex_center[1], outer_hex_angle)
        
        # Check containment
        for i in range(num_inner):
            if not self.check_containment(inner_polygons[i], outer_poly):
                # Large penalty for containment violations
                total_penalty += 1e6
        
        # Check overlaps
        for i in range(num_inner):
            for j in range(i+1, num_inner):
                if self.check_overlap(inner_polygons[i], inner_polygons[j]):
                    # Penalize overlap more heavily
                    total_penalty += 1e5
        
        # Objective: maximize 1/side_length
        # So minimize negative log of side_length plus penalties
        fitness = -np.log(side_length) - total_penalty
        
        return fitness, side_length
    
    def get_symmetric_initial_config(self) -> np.ndarray:
        """Generate initial configuration with symmetry properties."""
        # Start with symmetric pattern around circle
        positions = []
        # Add central hexagon
        positions.append([0, 0, 0])
        
        # Place others in circular pattern
        # Ring 1: 6 hexagons around center
        for i in range(6):
            angle = np.radians(60 * i)
            radius = 2.0  # approximate radius
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append([x, y, 0])
            
        # Ring 2: 5 hexagons at larger radius
        for i in range(5):
            angle = np.radians(360/5 * i)
            radius = 3.5  # larger radius
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append([x, y, 0])
        
        return np.array(positions).flatten()
    
    def refine_with_local_search(self, initial_solution: np.ndarray, 
                               max_iterations: int = 50) -> np.ndarray:
        """Use local optimization to refine the solution."""
        # Flatten the solution
        x0 = initial_solution.copy()
        
        def objective(x):
            # Reshape for evaluation
            positions = x.reshape(-1, 3)
            # Add small penalty to encourage staying close to initial
            initial_penalty = 0.1 * np.sum((x - initial_solution)**2)
            _, side_length = self.evaluate_fitness(x)
            return -np.log(side_length) - initial_penalty
        
        bounds = [(-20, 20)] * len(x0)  # Reasonable bounds
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': max_iterations})
            return result.x
        except:
            return x0
    
    def optimize_hexagon_packing(self) -> Tuple[np.ndarray, float]:
        """Main optimization function."""
        # Generate initial configuration with some symmetry
        initial_positions = self.get_symmetric_initial_config()
        
        # First pass with local optimization
        refined_solution = self.refine_with_local_search(initial_positions, 30)
        
        # Convert back to structured format for fitness evaluation
        positions = refined_solution.reshape(-1, 3)
        
        # Evaluate initial solution  
        initial_fitness, initial_side_length = self.evaluate_fitness(refined_solution)
        
        # Run differential evolution for global optimization
        bounds = [(-15, 15)] * len(refined_solution)  # reasonable bounds for hexagon positions
        
        def de_objective(x):
            _, side_length = self.evaluate_fitness(x)
            return -np.log(side_length)  # Minimize negative log side length
        
        try:
            # Run with multiple attempts for better results
            best_solution = None
            best_side_length = float('inf')
            
            for _ in range(3):  # Multiple runs
                result = differential_evolution(de_objective, bounds, 
                                              maxiter=100, popsize=15, 
                                              seed=None, disp=False)
                
                if result.success:
                    _, side_length = self.evaluate_fitness(result.x)
                    if side_length < best_side_length:
                        best_side_length = side_length
                        best_solution = result.x
            
            if best_solution is not None:
                final_positions = best_solution.reshape(-1, 3)
                _, final_side_length = self.evaluate_fitness(best_solution)
            else:
                final_positions = positions
                final_side_length = initial_side_length
                
        except Exception as e:
            print(f"Differential evolution error: {e}")
            final_positions = positions
            final_side_length = initial_side_length
        
        # Final local refinement
        flattened_final = final_positions.flatten()
        final_refined = self.refine_with_local_search(flattened_final, 20)
        final_positions = final_refined.reshape(-1, 3)
        _, final_side_length = self.evaluate_fitness(final_refined)
        
        return final_positions, final_side_length

def hexagon_packing_12() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize packer
    packer = HexagonPacker()
    
    # Optimize
    inner_hex_data, outer_hex_side_length = packer.optimize_hexagon_packing()
    
    # Ensure we have exactly 12 hexagons
    if len(inner_hex_data) != 12:
        # Fallback to simple grid if optimization fails
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer_hex_side_length = 8
    
    # Set outer hexagon parameters (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    # Final validation and adjustment
    try:
        # Double-check the solution
        _, computed_side_length = packer.evaluate_fitness(inner_hex_data.flatten())
        outer_hex_side_length = computed_side_length
    except:
        pass
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Calculate benchmark ratio (inverse side length vs target)
    benchmark_ratio = 1 / outer_hex_side_length / 0.2537
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
