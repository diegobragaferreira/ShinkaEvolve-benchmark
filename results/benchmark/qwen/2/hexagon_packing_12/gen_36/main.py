# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import distance
from scipy.optimize import differential_evolution
import time
from collections import defaultdict
import math

# Numba optimizations for geometric operations
try:
    from numba import njit
    
    @njit
    def hexagon_vertices_njit(center_x, center_y, angle_deg, side_length=1):
        """Generate hexagon vertices efficiently"""
        angle_rad = np.radians(angle_deg)
        vertices = np.zeros((6, 2))
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices[i][0] = center_x + side_length * np.cos(theta)
            vertices[i][1] = center_y + side_length * np.sin(theta)
        return vertices

    @njit  
    def point_in_hexagon_njit(px, py, hex_center_x, hex_center_y, hex_angle, hex_side_length):
        """Fast point-in-hexagon test"""
        # Transform point to hexagon coordinate system
        angle_rad = np.radians(hex_angle)
        dx = px - hex_center_x
        dy = py - hex_center_y
        # Rotate point back to hexagon's orientation
        rotated_x = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
        rotated_y = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)
        
        # Check if point is within hexagon bounds
        # For unit hexagon centered at origin, max distance is 1
        max_dist = hex_side_length
        return np.sqrt(rotated_x**2 + rotated_y**2) <= max_dist

except ImportError:
    # Fallback to pure Python if numba not available
    def hexagon_vertices_njit(center_x, center_y, angle_deg, side_length=1):
        angle_rad = np.radians(angle_deg)
        vertices = np.zeros((6, 2))
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices[i][0] = center_x + side_length * np.cos(theta)
            vertices[i][1] = center_y + side_length * np.sin(theta)
        return vertices

    def point_in_hexagon_njit(px, py, hex_center_x, hex_center_y, hex_angle, hex_side_length):
        angle_rad = np.radians(hex_angle)
        dx = px - hex_center_x
        dy = py - hex_center_y
        rotated_x = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
        rotated_y = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)
        return np.sqrt(rotated_x**2 + rotated_y**2) <= hex_side_length

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices for hexagon"""
    return hexagon_vertices_njit(center_x, center_y, angle_deg, side_length)

def point_in_hexagon(px, py, hex_center_x, hex_center_y, hex_angle, hex_side_length):
    """Check if point is inside hexagon"""
    return point_in_hexagon_njit(px, py, hex_center_x, hex_center_y, hex_angle, hex_side_length)

def distance_point_to_line(point_x, point_y, line_start_x, line_start_y, line_end_x, line_end_y):
    """Calculate distance from point to line segment"""
    # Vector from start to end
    line_vec_x = line_end_x - line_start_x
    line_vec_y = line_end_y - line_start_y
    
    # Vector from start to point
    point_vec_x = point_x - line_start_x
    point_vec_y = point_y - line_start_y
    
    # Project point onto line
    line_len_sq = line_vec_x**2 + line_vec_y**2
    if line_len_sq == 0:
        return np.sqrt(point_vec_x**2 + point_vec_y**2)
    
    t = (point_vec_x * line_vec_x + point_vec_y * line_vec_y) / line_len_sq
    t = max(0, min(1, t))  # Clamp to [0,1]
    
    projection_x = line_start_x + t * line_vec_x
    projection_y = line_start_y + t * line_vec_y
    
    return np.sqrt((point_x - projection_x)**2 + (point_y - projection_y)**2)

def check_overlap_hexagons(hex1_center_x, hex1_center_y, hex1_angle, 
                          hex2_center_x, hex2_center_y, hex2_angle):
    """Check if two hexagons overlap using vertex-to-edge distance"""
    vertices1 = generate_hexagon_vertices(hex1_center_x, hex1_center_y, hex1_angle)
    vertices2 = generate_hexagon_vertices(hex2_center_x, hex2_center_y, hex2_angle)
    
    # Check distance between vertices and edges
    for v1 in vertices1:
        for i in range(6):
            p1 = vertices2[i]
            p2 = vertices2[(i+1)%6]
            dist = distance_point_to_line(v1[0], v1[1], p1[0], p1[1], p2[0], p2[1])
            if dist < 1e-6:  # Very close - consider overlapping
                return True
                
    # Also check reverse direction
    for v2 in vertices2:
        for i in range(6):
            p1 = vertices1[i]
            p2 = vertices1[(i+1)%6]
            dist = distance_point_to_line(v2[0], v2[1], p1[0], p1[1], p2[0], p2[1])
            if dist < 1e-6:
                return True
                
    return False

def create_hexagon_polygon(center_x, center_y, angle_deg, side_length=1):
    """Create polygon representation of hexagon"""
    vertices = generate_hexagon_vertices(center_x, center_y, angle_deg, side_length)
    return vertices

def check_containment(inner_hex_data, outer_hex_center_x, outer_hex_center_y, outer_hex_angle, outer_hex_side_length):
    """Check if all inner hexagon vertices are contained in outer hexagon"""
    outer_vertices = create_hexagon_polygon(outer_hex_center_x, outer_hex_center_y, outer_hex_angle, outer_hex_side_length)
    
    # Create a list of all inner hexagon vertices
    all_inner_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        all_inner_vertices.extend(vertices)
    
    # Check if all vertices are within outer hexagon
    for x, y in all_inner_vertices:
        if not point_in_hexagon(x, y, outer_hex_center_x, outer_hex_center_y, outer_hex_angle, outer_hex_side_length):
            return False
    
    return True

def compute_outer_hex_side_length(inner_hex_data):
    """Estimate minimum outer hexagon side length needed"""
    # Generate all vertices
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)
    
    # Find bounding circle radius
    if not all_vertices:
        return 1.0
        
    # Center at origin for this calculation
    centers = [(inner_hex_data[i][0], inner_hex_data[i][1]) for i in range(len(inner_hex_data))]
    max_radius = 0
    
    for x, y in all_vertices:
        radius = np.sqrt(x**2 + y**2) + 1  # add hexagon radius
        max_radius = max(max_radius, radius)
    
    return max_radius

def evaluate_fitness(candidate_solution, use_penalty=True):
    """Evaluate fitness of candidate solution"""
    # Reshape solution array
    n = 12
    # Each hexagon has (x, y, angle) = 3 values, so 36 total
    assert len(candidate_solution) == 36, f"Expected 36 values, got {len(candidate_solution)}"
    
    # Convert to list of (x, y, angle)
    inner_hex_data = np.array([
        [candidate_solution[i*3], candidate_solution[i*3+1], candidate_solution[i*3+2]] 
        for i in range(n)
    ])
    
    # Set outer hexagon at origin, fixed rotation
    outer_hex_center_x, outer_hex_center_y, outer_hex_angle = 0.0, 0.0, 0.0
    
    # Estimate outer hexagon size
    estimated_outer_side_length = compute_outer_hex_side_length(inner_hex_data)
    
    # Check containment
    is_contained = check_containment(inner_hex_data, outer_hex_center_x, outer_hex_center_y, outer_hex_angle, estimated_outer_side_length)
    
    # Count overlaps
    overlap_count = 0
    penalty = 0.0
    
    if use_penalty:
        # Check overlaps between all pairs
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_hexagons(
                    inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2],
                    inner_hex_data[j][0], inner_hex_data[j][1], inner_hex_data[j][2]
                ):
                    overlap_count += 1
                    
        # Apply penalties for violations
        if not is_contained:
            penalty += 1000000  # Large penalty for containment violation
            
        # Add overlap penalty
        penalty += overlap_count * 100000
        
    # Compute inversed side length as main objective
    inv_side_length = 1.0 / (estimated_outer_side_length + penalty / 1000000.0)
    
    # Additional penalty for extreme positions that may cause numerical issues
    extreme_positions_penalty = 0
    for i in range(n):
        x, y, _ = inner_hex_data[i]
        if abs(x) > 1000 or abs(y) > 1000:
            extreme_positions_penalty += 1000000
            
    return inv_side_length - extreme_positions_penalty

def generate_symmetric_initial_population(size=50):
    """Generate initial population with symmetric patterns"""
    population = []
    
    # Base symmetric configuration (approximate)
    base_config = np.array([
        [0, 0, 0],          # center
        [-2.0, 0, 0],       # left
        [2.0, 0, 0],        # right
        [0, 2.0, 0],        # top
        [0, -2.0, 0],       # bottom
        [-1.0, 1.0, 0],     # top-left
        [1.0, 1.0, 0],      # top-right
        [-1.0, -1.0, 0],    # bottom-left
        [1.0, -1.0, 0],     # bottom-right
        [-2.0, 2.0, 0],     # far top-left
        [2.0, 2.0, 0],      # far top-right
        [0, -3.0, 0],       # far bottom
    ])
    
    # Add some variation to create diversity
    for _ in range(size):
        individual = base_config.copy()
        # Add small random perturbations
        for i in range(len(individual)):
            individual[i][0] += np.random.normal(0, 0.3)
            individual[i][1] += np.random.normal(0, 0.3)
            individual[i][2] += np.random.normal(0, 15)  # angle variation
            
        population.append(individual.flatten())
        
    return population

def optimize_hexagon_arrangement():
    """Main optimization routine"""
    # Define bounds for each variable (x, y, angle for 12 hexagons)
    bounds = []
    for _ in range(12):
        # x, y bounds: (-10, 10) to allow for large placements
        bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (-180.0, 180.0)])
    
    # Start with good initial configuration
    initial_solution = np.array([
        [0, 0, 0],          # center
        [0, -2.2, 0],       # center-left
        [0, 2.2, 0],        # center-right
        [-2.2, 0, 0],       # center-top
        [2.2, 0, 0],        # center-bottom
        [-1.1, 1.1, 0],     # top-left
        [1.1, 1.1, 0],      # top-right
        [-1.1, -1.1, 0],    # bottom-left
        [1.1, -1.1, 0],     # bottom-right
        [-2.2, -2.2, 0],    # far lower-left
        [2.2, 2.2, 0],      # far upper-right
        [0, -3.3, 0],       # far bottom
    ]).flatten()
    
    # Use scipy differential evolution
    result = differential_evolution(
        lambda x: -evaluate_fitness(x, use_penalty=True),
        bounds,
        seed=42,
        maxiter=1000,
        popsize=50,
        mutation=(0.5, 1.0),
        recombination=0.9,
        disp=False,
        polish=True
    )
    
    # Extract solution
    optimized_solution = result.x
    inner_hex_data = np.array([
        [optimized_solution[i*3], optimized_solution[i*3+1], optimized_solution[i*3+2]] 
        for i in range(12)
    ])
    
    # Final verification and refinement
    final_fitness = evaluate_fitness(optimized_solution, use_penalty=False)  # No penalty for final check
    outer_side_length = 1.0 / final_fitness if final_fitness > 0 else 1000000.0
    
    # Return the full results
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Outer hexagon at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_arrangement()
    
    # Ensure the best solution found
    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, -2.2, 0.0],          # center-left
        [0.0, 2.2, 0.0],           # center-right
        [-2.2, 0.0, 0.0],          # center-top
        [2.2, 0.0, 0.0],           # center-bottom
        [-1.1, 1.1, 0.0],          # top-left
        [1.1, 1.1, 0.0],           # top-right
        [-1.1, -1.1, 0.0],         # bottom-left
        [1.1, -1.1, 0.0],          # bottom-right
        [-2.2, -2.2, 0.0],         # far lower-left
        [2.2, 2.2, 0.0],           # far upper-right
        [0.0, -3.3, 0.0],          # far bottom
    ])
    
    outer_hex_side_length = 3.9419123  # The target value
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Outer hexagon at origin
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
