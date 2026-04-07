# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time
import copy

# Constants
UNIT_HEXAGON_RADIUS = 1.0
UNIT_HEXAGON_APOGEE = np.sqrt(3)/2
UNIT_HEXAGON_VERTEX_ANGLE = np.pi/3
PI_3 = np.pi/3
SQRT_3 = np.sqrt(3)

class HexagonPacker:
    def __init__(self):
        self.hexagon_vertices_cache = {}
        
    def create_unit_hexagon_vertices(self, center=(0,0), rotation=0):
        """Create vertices of a unit regular hexagon with caching."""
        cache_key = (center, rotation)
        if cache_key in self.hexagon_vertices_cache:
            return self.hexagon_vertices_cache[cache_key]
            
        vertices = []
        for i in range(6):
            angle = rotation + i * UNIT_HEXAGON_VERTEX_ANGLE
            x = center[0] + UNIT_HEXAGON_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEXAGON_RADIUS * np.sin(angle)
            vertices.append((x, y))
        result = np.array(vertices)
        self.hexagon_vertices_cache[cache_key] = result
        return result
    
    def compute_outer_hexagon_vertices(self, center=(0,0), side_length=1.0, rotation=0):
        """Create vertices of the outer hexagon."""
        vertices = []
        for i in range(6):
            angle = rotation + i * UNIT_HEXAGON_VERTEX_ANGLE
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
    
    def check_containment(self, inner_vertices, outer_vertices):
        """Check if all vertices of inner hexagon are within outer hexagon."""
        inner_polygon = Polygon(inner_vertices)
        outer_polygon = Polygon(outer_vertices)
        return outer_polygon.contains(inner_polygon)
    
    def check_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    def validate_configuration(self, positions, side_length):
        """Validate a configuration for all constraints."""
        # Create outer hexagon
        outer_hex = self.compute_outer_hexagon_vertices((0,0), side_length)
        
        # Check containment for all inner hexagons
        for i, pos in enumerate(positions):
            x, y, angle = pos
            inner_hex = self.create_unit_hexagon_vertices((x, y), np.radians(angle))
            if not self.check_containment(inner_hex, outer_hex):
                return False
        
        # Check overlaps between all pairs
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                x1, y1, angle1 = positions[i]
                x2, y2, angle2 = positions[j]
                hex1 = self.create_unit_hexagon_vertices((x1, y1), np.radians(angle1))
                hex2 = self.create_unit_hexagon_vertices((x2, y2), np.radians(angle2))
                if self.check_overlap(hex1, hex2):
                    return False
                    
        return True

def generate_symmetric_initial_config():
    """Generate a symmetric initial configuration for 12 hexagons."""
    positions = []
    
    # Center hexagon
    positions.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * PI_3
        x = UNIT_HEXAGON_RADIUS * np.cos(angle)
        y = UNIT_HEXAGON_RADIUS * np.sin(angle)
        positions.append([x, y, 0.0])
    
    # Second ring (5 hexagons)
    for i in range(5):
        angle = i * PI_3
        x = 2 * UNIT_HEXAGON_RADIUS * np.cos(angle)
        y = 2 * UNIT_HEXAGON_RADIUS * np.sin(angle)
        positions.append([x, y, 0.0])
    
    # Bottom center
    positions.append([0.0, -2 * UNIT_HEXAGON_RADIUS, 0.0])
    
    return np.array(positions)

def evaluate_fitness(config):
    """Evaluate fitness of a configuration."""
    # Separate positions and side length
    positions = config[:-1].reshape(-1, 3)
    side_length = config[-1]
    
    # Validate configuration
    packer = HexagonPacker()
    if not packer.validate_configuration(positions, side_length):
        return -1e10  # Invalid configuration
    
    # Return fitness (inverse of side length for maximization)
    return 1.0 / side_length if side_length > 0 else -1e10

def objective_for_evolution(x):
    """Wrapper for evolutionary algorithm that returns negative fitness."""
    # Reshape back to correct dimensions
    positions = x[:-1].reshape(-1, 3)
    side_length = x[-1]
    
    # Validate and calculate fitness
    packer = HexagonPacker() 
    if not packer.validate_configuration(positions, side_length):
        return 1e10  # Penalize invalid configs heavily
    
    # Return negative of inverse side length for minimization
    return -1.0 / side_length if side_length > 0 else 1e10

def initialize_evolution_bounds():
    """Initialize bounds for evolutionary algorithm."""
    # Position bounds: x,y in [-10, 10], angle in [0, 360)
    bounds = [(-10, 10), (-10, 10), (0, 360)] * 12
    # Side length bound: [1, 10]
    bounds.append((1.0, 10.0))
    return bounds

def optimize_with_evolution():
    """Optimize using evolutionary algorithm."""
    # Generate initial configuration
    initial_positions = generate_symmetric_initial_config()
    initial_side_length = 5.0  # Reasonable starting point
    
    # Flatten for evolutionary algorithm
    initial_flat = initial_positions.flatten()
    initial_config = np.append(initial_flat, initial_side_length)
    
    # Set up bounds
    bounds = initialize_evolution_bounds()
    
    # Run evolutionary optimization
    result = differential_evolution(
        objective_for_evolution,
        bounds,
        maxiter=500,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )
    
    if result.success:
        # Reshape results
        final_positions = result.x[:-1].reshape(-1, 3)
        final_side_length = result.x[-1]
        return final_positions, final_side_length
    else:
        # Fallback to initial if optimization fails
        return initial_positions, initial_side_length

def generate_fallback_config():
    """Generate a fallback configuration."""
    inner_hex_data = np.array([
        [0, 0, 0],          # center
        [-2.5, 0, 0],       # left
        [2.5, 0, 0],        # right
        [-1.25, 2.17, 0],   # top-left
        [1.25, 2.17, 0],    # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],   # bottom-right
        [-3.75, 2.17, 0],   # far top-left
        [3.75, 2.17, 0],    # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],   # far bottom-right
        [0, -4, 0],         # far bottom-center
    ])
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Perform evolutionary optimization
        final_positions, final_side_length = optimize_with_evolution()
        
        # Prepare output in required format
        inner_hex_data = final_positions.copy()
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered
        
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to simple configuration
        inner_hex_data, outer_hex_data, final_side_length = generate_fallback_config()
    
    end_time = time.time()
    
    # Calculate performance metrics
    inv_outer_hex_side_length = 1.0 / final_side_length if final_side_length > 0 else 0.0
    benchmark_ratio = inv_outer_hex_side_length / 0.2537
    
    print(f"Optimized result: inverse_side_length={inv_outer_hex_side_length:.6f}, "
          f"benchmark_ratio={benchmark_ratio:.6f}, eval_time={(end_time-start_time):.3f}s")
    
    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END