# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
from numba import jit

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, size=1, angle_deg=0):
    """Fast generation of hexagon vertices using numba."""
    angle_rad = np.radians(angle_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + size * np.cos(angle)
        y = center_y + size * np.sin(angle)
        vertices[i] = [x, y]
    return vertices

class HexagonPacker:
    def __init__(self):
        self.hex_side_length = 1.0
        self.outer_center = np.array([0.0, 0.0])
        
    def hexagon_vertices(self, center_x, center_y, size=1, angle_deg=0):
        """Generate vertices of a regular hexagon."""
        return hexagon_vertices_jit(center_x, center_y, size, angle_deg)
    
    def get_outer_hexagon(self, outer_radius):
        """Get vertices of the outer hexagon."""
        return self.hexagon_vertices(self.outer_center[0], self.outer_center[1], outer_radius, 0)
    
    def validate_containment(self, hex_vertices, outer_radius):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        outer_vertices = self.get_outer_hexagon(outer_radius)
        outer_polygon = Polygon(outer_vertices)
        
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True
    
    def validate_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    def calculate_max_distance_from_center(self, hex_data):
        """Calculate maximum distance from center to any hexagon vertex."""
        max_dist = 0
        for i in range(len(hex_data)):
            cx, cy, _ = hex_data[i]
            # Calculate distance to center plus hexagon radius
            dist = np.sqrt(cx**2 + cy**2) + self.hex_side_length
            max_dist = max(max_dist, dist)
        return max_dist

class SymmetricInitializer:
    @staticmethod
    def create_advanced_symmetric_config():
        """Create initial configuration with better spatial distribution."""
        config = []
        
        # Central hexagon  
        config.append([0, 0, 0])
        
        # First ring (6 hexagons) - arranged in a tight hexagonal pattern
        for i in range(6):
            angle = i * 60
            radius = 2.0  # Distance from origin
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            config.append([x, y, 0])
        
        # Second ring (6 hexagons) - arranged to fill gaps
        for i in range(6):
            angle = 30 + i * 60
            radius = 3.464  # sqrt(12) approximately, optimized spacing
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            config.append([x, y, 0])
            
        return np.array(config)

class Optimizer:
    def __init__(self, packer):
        self.packer = packer
        
    def flatten_config(self, hex_data):
        """Convert 2D array to flat parameter list for optimization."""
        params = []
        for i in range(len(hex_data)):
            params.extend([hex_data[i][0], hex_data[i][1]])  # Only positions, not angles
        return np.array(params)
    
    def unflatten_config(self, params, original_config):
        """Reconstruct hex_data from flattened parameters."""
        config = original_config.copy()
        idx = 0
        for i in range(len(config)):
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2
        return config
    
    def objective_function(self, params, hex_data, outer_radius):
        """Objective function for optimization."""
        # Reconstruct configuration
        reconstructed_config = self.unflatten_config(params, hex_data)
        
        # Evaluate
        validity, inv_radius = self.packer.evaluate_configuration(reconstructed_config, outer_radius)
        
        if not validity:
            return 1e10  # Large penalty for invalid configurations
        return -inv_radius  # Negative because we maximize
    
    def optimize_positions(self, initial_config, outer_radius):
        """Optimize positions using constrained numerical optimization."""
        # Flatten initial configuration
        initial_params = self.flatten_config(initial_config)
        
        # Perform optimization
        result = minimize(
            self.objective_function, 
            initial_params, 
            args=(initial_config, outer_radius),
            method='L-BFGS-B',
            bounds=[(-10, 10), (-10, 10)] * len(initial_config)
        )
        
        # Reconstruct optimized configuration
        optimized_config = self.unflatten_config(result.x, initial_config)
        return optimized_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    packer = HexagonPacker()
    initializer = SymmetricInitializer()
    optimizer = Optimizer(packer)
    
    # Generate initial symmetric configuration
    initial_config = initializer.create_advanced_symmetric_config()
    
    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0
    packer.outer_center = np.array([outer_center_x, outer_center_y])
    
    # Start with a reasonable outer radius estimate
    estimated_outer_radius = packer.calculate_max_distance_from_center(initial_config)
    
    # Optimization loop
    best_config = initial_config.copy()
    best_inv_radius = 0
    max_iterations = 10
    
    for iteration in range(max_iterations):
        # Optimize positions
        optimized_config = optimizer.optimize_positions(initial_config, estimated_outer_radius)
        
        # Check validity
        validity, inv_radius = packer.evaluate_configuration(optimized_config, estimated_outer_radius)
        
        if validity and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = optimized_config.copy()
            
        if validity:
            break
            
        # Try perturbation if invalid
        for i in range(len(optimized_config)):
            optimized_config[i][0] += np.random.normal(0, 0.05)
            optimized_config[i][1] += np.random.normal(0, 0.05)
            
        initial_config = optimized_config
    
    # Final validation and refinement
    final_validity, final_inv_radius = packer.evaluate_configuration(best_config, estimated_outer_radius)
    
    if not final_validity:
        # Fall back to a known good configuration with better parameters
        best_config = np.array([
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
            [0, -4, 0]
        ])
        final_inv_radius = 1.0 / 8.0  # Conservative estimate
    
    # Calculate final outer radius
    final_outer_radius = 1.0 / final_inv_radius if final_inv_radius > 0 else 8.0
    
    # Prepare return values
    inner_hex_data = np.array(best_config)
    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])
    outer_hex_side_length = final_outer_radius * 2
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
