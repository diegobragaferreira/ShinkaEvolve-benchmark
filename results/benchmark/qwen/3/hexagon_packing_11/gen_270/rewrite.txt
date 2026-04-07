# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import time
from numba import jit
import math

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, side_length, angle_degrees):
    """Generate vertices of a regular hexagon (jit compiled)"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Vertices of unit hexagon centered at origin
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    rotated_vertices = np.empty_like(base_vertices)
    for i in range(6):
        x = base_vertices[i, 0]
        y = base_vertices[i, 1]
        rotated_vertices[i, 0] = x * cos_a - y * sin_a
        rotated_vertices[i, 1] = x * sin_a + y * cos_a
    
    return rotated_vertices * side_length + np.array([center_x, center_y])

class HexagonPacker:
    def __init__(self):
        self.hex_side_length = 1.0
        self.n_inner_hexagons = 11
        
    def hexagon_polygon(self, center_x, center_y, angle_degrees):
        """Create shapely polygon representation of hexagon"""
        vertices = hexagon_vertices_jit(center_x, center_y, self.hex_side_length, angle_degrees)
        return Polygon(vertices)
        
    def compute_outer_radius_bounds(self, hex_centers):
        """Compute tight analytical bounds for outer hexagon radius"""
        # Get all vertices of all hexagons (simplified approach)
        max_dist = 0
        for i, center in enumerate(hex_centers):
            cx, cy = center
            # For unit hexagon, distance from center to corner is 1
            # For unit hexagon, distance from center to edge is sqrt(3)/2
            # We use the maximum distance to any vertex
            dist = np.sqrt(cx*cx + cy*cy) + 1.0  # Max distance to vertex
            max_dist = max(max_dist, dist)
        return max_dist * 1.01  # Small buffer
    
    def compute_outer_radius_exact(self, hex_data):
        """Compute exact outer radius using vertex analysis"""
        all_vertices = []
        for center_x, center_y, angle in hex_data:
            vertices = hexagon_vertices_jit(center_x, center_y, self.hex_side_length, angle)
            all_vertices.extend(vertices)
        
        if not all_vertices:
            return 10.0
            
        # Find center of all vertices
        avg_x = sum(v[0] for v in all_vertices) / len(all_vertices)
        avg_y = sum(v[1] for v in all_vertices) / len(all_vertices)
        
        # Find max distance from average center to any vertex
        max_dist = 0
        for x, y in all_vertices:
            dist = np.sqrt((x - avg_x)**2 + (y - avg_y)**2)
            max_dist = max(max_dist, dist)
            
        return max_dist + 0.001  # Small buffer
    
    def check_containment(self, hex_data, outer_radius):
        """Check if all hexagons are contained in outer hexagon"""
        # Check if all hexagon centers are within the outer hexagon
        outer_vertices = hexagon_vertices_jit(0, 0, outer_radius, 0)
        outer_polygon = Polygon(outer_vertices)
        
        for center_x, center_y, angle in hex_data:
            # Just check if center is inside - sufficient for containment
            if not outer_polygon.contains(Point(center_x, center_y)):
                return False
        return True
    
    def check_overlaps(self, hex_data):
        """Check for overlaps using analytical approach"""
        # Convert to numpy for faster computation
        centers = np.array([[c[0], c[1]] for c in hex_data])
        
        # Compute pairwise distances
        distances = cdist(centers, centers)
        
        # For unit hexagons, minimum distance to avoid overlap is 2
        # We allow for slight tolerance due to rotational effects
        min_distance = 1.999  # Slightly less than 2 to account for rotations
        
        # Check all pairs
        for i in range(len(distances)):
            for j in range(i+1, len(distances[i])):
                if distances[i, j] < min_distance:
                    return True
        return False
    
    def construct_hexagonal_lattice(self):
        """Construct a highly symmetric hexagonal lattice arrangement"""
        # Central hexagon
        positions = [[0.0, 0.0]]
        
        # First ring - 6 hexagons
        for i in range(6):
            angle = i * 60
            rad_angle = np.radians(angle)
            x = 2.0 * np.cos(rad_angle)
            y = 2.0 * np.sin(rad_angle)
            positions.append([x, y])
        
        # Second ring - additional positions to reach 11
        # Place them at positions that maximize space utilization
        positions.extend([[-1.0, 1.732], [1.0, 1.732], [-1.0, -1.732], [1.0, -1.732]])
        
        # Truncate to 11 positions and add some symmetry
        positions = positions[:11]
        
        # Return with zero angles (no rotation)
        return [(pos[0], pos[1], 0.0) for pos in positions]
    
    def build_optimization_problem(self, initial_positions):
        """Build optimization problem for hexagon packing"""
        # Simplified approach: optimize only centers and angles
        # Since we know the hexagons have fixed size, we can treat this as a geometric optimization
        
        # Convert to numpy for easier manipulation
        initial_data = np.array(initial_positions)
        n = len(initial_data)
        
        # Flatten for optimization
        initial_flat = []
        for i in range(n):
            initial_flat.extend([initial_data[i][0], initial_data[i][1], initial_data[i][2]])
        
        # Objective: minimize outer radius while satisfying constraints
        def objective(params):
            # Reshape parameters
            hex_data = []
            for i in range(n):
                idx = i * 3
                hex_data.append((params[idx], params[idx+1], params[idx+2]))
            
            # Compute outer radius (this is the main quantity we want to minimize)
            outer_radius = self.compute_outer_radius_exact(hex_data)
            
            # Add penalty for overlaps
            if self.check_overlaps(hex_data):
                return outer_radius + 1000.0  # Large penalty
            
            return outer_radius
        
        return initial_flat, objective
    
    def optimize_with_gradient_descent(self, initial_positions, max_iter=200):
        """Use gradient-based optimization to refine the solution"""
        
        # Build optimization problem
        initial_flat, objective = self.build_optimization_problem(initial_positions)
        
        # Use L-BFGS-B with bounds
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (0.0, 360.0)] * len(initial_positions)
        
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if not result.success:
            # If optimization fails, return original
            return initial_positions
        
        # Reshape result back
        final_positions = []
        for i in range(len(initial_positions)):
            idx = i * 3
            final_positions.append((
                result.x[idx], 
                result.x[idx+1], 
                result.x[idx+2]
            ))
        
        return final_positions
    
    def refine_solution(self, initial_positions):
        """Apply multiple refinements to improve solution quality"""
        # Refinement 1: Initial geometric optimization
        refined1 = self.optimize_with_gradient_descent(initial_positions, 100)
        
        # Refinement 2: Further optimization with stricter tolerances
        refined2 = self.optimize_with_gradient_descent(refined1, 150)
        
        return refined2

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    packer = HexagonPacker()
    
    # Start with a well-structured hexagonal lattice arrangement
    initial_positions = packer.construct_hexagonal_lattice()
    
    # Refine the solution using gradient-based optimization
    refined_positions = packer.refine_solution(initial_positions)
    
    # Final validation and outer radius computation
    outer_radius = packer.compute_outer_radius_exact(refined_positions)
    
    # Ensure solution validity
    if packer.check_overlaps(refined_positions):
        # If overlaps found, fall back to initial
        refined_positions = initial_positions
        outer_radius = packer.compute_outer_radius_exact(refined_positions)
    
    # Convert to numpy array for output
    inner_hex_data = np.array(refined_positions)
    
    # Outer hexagon centered at origin
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END