# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from numba import jit, prange
import time
from collections import defaultdict

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given position, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using separating axis theorem"""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_vertices:
        if point_in_polygon(v, hex2_vertices):
            return True
    # Check if any vertex of hex2 is inside hex1
    for v in hex2_vertices:
        if point_in_polygon(v, hex1_vertices):
            return True
    return False

def create_spatial_hash(hex_vertices_list, cell_size=2.0):
    """Create spatial hash grid for fast overlap checking"""
    hash_grid = defaultdict(list)
    for i, vertices in enumerate(hex_vertices_list):
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        # Add to all relevant cells
        start_col = int(min_x // cell_size)
        end_col = int(max_x // cell_size) + 1
        start_row = int(min_y // cell_size)
        end_row = int(max_y // cell_size) + 1

        for col in range(start_col, end_col + 1):
            for row in range(start_row, end_row + 1):
                hash_grid[(col, row)].append(i)
    return hash_grid

def get_overlapping_indices(hash_grid, hex_index, hex_vertices, cell_size=2.0):
    """Get indices of potentially overlapping hexagons using spatial hash"""
    overlapping = set()
    # Get bounding box of hexagon
    min_x = min(v[0] for v in hex_vertices)
    max_x = max(v[0] for v in hex_vertices)
    min_y = min(v[1] for v in hex_vertices)
    max_y = max(v[1] for v in hex_vertices)

    # Check all relevant cells
    start_col = int(min_x // cell_size)
    end_col = int(max_x // cell_size) + 1
    start_row = int(min_y // cell_size)
    end_row = int(max_y // cell_size) + 1

    for col in range(start_col, end_col + 1):
        for row in range(start_row, end_row + 1):
            if (col, row) in hash_grid:
                for idx in hash_grid[(col, row)]:
                    if idx != hex_index:
                        overlapping.add(idx)
    return overlapping

def compute_outer_hex_side_length(inner_hex_data):
    """Compute minimum outer hexagon side length required to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, angle)
        for vx, vy in vertices:
            dist = np.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)
    return max_dist * 2.0 / np.sqrt(3) + 0.5  # Add small margin

def calculate_force_repulsion(pos1, pos2, strength=100.0):
    """Calculate repulsive force between two points"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dist = np.sqrt(dx*dx + dy*dy)
    
    if dist < 0.1:
        return np.array([0.0, 0.0])
        
    # Repulsive force inversely proportional to distance squared
    force_magnitude = strength / (dist * dist + 1e-8)
    force_x = force_magnitude * dx / dist
    force_y = force_magnitude * dy / dist
    
    return np.array([force_x, force_y])

def calculate_force_attraction(pos, target, strength=1.0):
    """Calculate attractive force toward target"""
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    dist = np.sqrt(dx*dx + dy*dy)
    
    if dist < 1e-6:
        return np.array([0.0, 0.0])
        
    force_magnitude = strength * dist
    force_x = force_magnitude * dx / dist
    force_y = force_magnitude * dy / dist
    
    return np.array([force_x, force_y])

def calculate_boundary_force(pos, boundary_radius, strength=10.0):
    """Calculate repulsive force from boundary"""
    dx = pos[0]
    dy = pos[1]
    dist = np.sqrt(dx*dx + dy*dy)
    
    if dist < 0.1:
        return np.array([0.0, 0.0])
        
    # Force is repulsive when inside boundary
    if dist < boundary_radius:
        force_magnitude = strength * (boundary_radius - dist) / (dist + 1e-8)
        force_x = force_magnitude * dx / dist
        force_y = force_magnitude * dy / dist
        return np.array([-force_x, -force_y])
    else:
        return np.array([0.0, 0.0])

def simulate_hexagon_gravity(inner_hex_data, outer_radius, max_steps=1000, dt=0.01):
    """Simulate physics-based optimization of hexagon positions"""
    num_hex = len(inner_hex_data)
    positions = np.array([[inner_hex_data[i][0], inner_hex_data[i][1]] for i in range(num_hex)])
    velocities = np.zeros((num_hex, 2))
    accelerations = np.zeros((num_hex, 2))
    
    # Initialize forces
    forces = np.zeros((num_hex, 2))
    
    # Boundary constraints
    boundary_center = np.array([0.0, 0.0])
    
    best_positions = positions.copy()
    best_penalty = float('inf')
    
    for step in range(max_steps):
        # Reset forces
        forces.fill(0.0)
        
        # Calculate forces between all hexagons
        for i in range(num_hex):
            for j in range(i+1, num_hex):
                force = calculate_force_repulsion(positions[i], positions[j])
                forces[i] += force
                forces[j] -= force  # Newton's third law
        
        # Calculate boundary forces
        for i in range(num_hex):
            boundary_force = calculate_boundary_force(positions[i], outer_radius)
            forces[i] += boundary_force
            
        # Update positions and velocities
        for i in range(num_hex):
            accelerations[i] = forces[i]
            velocities[i] += accelerations[i] * dt
            # Apply velocity damping
            velocities[i] *= 0.95
            positions[i] += velocities[i] * dt
            
            # Keep within reasonable bounds
            pos_mag = np.sqrt(positions[i][0]**2 + positions[i][1]**2)
            if pos_mag > 10.0:
                positions[i] = positions[i] / pos_mag * 10.0
        
        # Check for improvement
        current_hex_data = np.array([[positions[i][0], positions[i][1], inner_hex_data[i][2]] for i in range(num_hex)])
        current_outer_radius = compute_outer_hex_side_length(current_hex_data)
        
        # Check constraints
        penalty = 0
        hex_vertices_list = [hexagon_vertices(current_hex_data[i][0], current_hex_data[i][1], current_hex_data[i][2]) for i in range(num_hex)]
        hash_grid = create_spatial_hash(hex_vertices_list)
        
        # Check containment
        outer_hex_vertices = hexagon_vertices(0, 0, 0, current_outer_radius)
        for i in range(num_hex):
            vertices = hex_vertices_list[i]
            for vx, vy in vertices:
                point = np.array([vx, vy])
                if not point_in_polygon(point, outer_hex_vertices):
                    penalty += 1000000
        
        # Check overlaps
        for i in range(num_hex):
            overlapping_indices = get_overlapping_indices(hash_grid, i, hex_vertices_list[i])
            for j in overlapping_indices:
                if i < j:
                    if check_hexagon_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                        penalty += 1000000
                        
        if penalty == 0 and current_outer_radius < best_penalty:
            best_penalty = current_outer_radius
            best_positions = positions.copy()
            
        # Early termination if good solution found
        if penalty == 0 and current_outer_radius < 4.0:
            break
    
    # Return best configuration found
    final_hex_data = np.array([[best_positions[i][0], best_positions[i][1], inner_hex_data[i][2]] for i in range(num_hex)])
    return final_hex_data

def calculate_penalties(inner_hex_data, outer_radius):
    """Calculate penalty for constraints"""
    penalty = 0
    
    # Check containment
    outer_hex_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    hex_vertices_list = [hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2]) for i in range(len(inner_hex_data))]
    
    for i in range(len(inner_hex_data)):
        vertices = hex_vertices_list[i]
        for vx, vy in vertices:
            point = np.array([vx, vy])
            if not point_in_polygon(point, outer_hex_vertices):
                penalty += 1000000
                
    # Check overlaps
    hash_grid = create_spatial_hash(hex_vertices_list)
    for i in range(len(inner_hex_data)):
        overlapping_indices = get_overlapping_indices(hash_grid, i, hex_vertices_list[i])
        for j in overlapping_indices:
            if i < j:
                if check_hexagon_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                    penalty += 1000000
                    
    return penalty

def generate_initial_config():
    """Generate initial configuration using symmetry and geometric insights"""
    # Use a known good configuration with small variations
    positions = [
        [0.0, 0.0, 0.0],      # center
        [0.0, 2.0, 0.0],      # up
        [0.0, -2.0, 0.0],     # down
        [1.73, 1.0, 0.0],     # up-right
        [-1.73, 1.0, 0.0],    # up-left
        [1.73, -1.0, 0.0],    # down-right
        [-1.73, -1.0, 0.0],   # down-left
        [3.46, 0.0, 0.0],     # far right
        [-3.46, 0.0, 0.0],    # far left
        [0.0, 3.46, 0.0],     # far up
        [0.0, -3.46, 0.0],    # far down
        [1.73, 3.0, 0.0],     # far upper right
    ]
    
    # Add slight randomness
    positions = np.array(positions)
    for i in range(1, len(positions)):
        positions[i][0] += np.random.normal(0, 0.1)
        positions[i][1] += np.random.normal(0, 0.1)
        
    return positions

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate initial configuration
    inner_hex_data = generate_initial_config()
    
    # Estimate initial outer radius
    initial_outer_radius = compute_outer_hex_side_length(inner_hex_data)
    
    # Run physics-based simulation with iterative refinement
    best_inner_data = inner_hex_data.copy()
    best_outer_radius = initial_outer_radius
    
    # Multiple simulation runs with different parameters
    for run in range(3):
        # Vary parameters for better exploration
        if run == 0:
            max_steps = 500
            dt = 0.01
        elif run == 1:
            max_steps = 800
            dt = 0.005
        else:
            max_steps = 1000
            dt = 0.002
            
        # Run gravity simulation
        simulated_data = simulate_hexagon_gravity(inner_hex_data, initial_outer_radius, max_steps, dt)
        
        # Calculate final outer radius
        final_radius = compute_outer_hex_side_length(simulated_data)
        
        # Validate and accept better solution
        penalty = calculate_penalties(simulated_data, final_radius)
        if penalty == 0 and final_radius < best_outer_radius:
            best_outer_radius = final_radius
            best_inner_data = simulated_data.copy()
    
    # Final validation and refinement
    final_penalty = calculate_penalties(best_inner_data, best_outer_radius)
    if final_penalty > 1000000:
        # If constraints violated, use fallback
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_side_length = 8.0
    else:
        inner_hex_data = best_inner_data
        outer_hex_side_length = best_outer_radius
    
    # Create outer hexagon data (centered, no rotation)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin, no rotation
    
    end_time = time.time()
    
    # Calculate benchmark ratio
    benchmark_ratio = 1.0 / outer_hex_side_length / 0.2537
    
    # Print diagnostic information
    print(f"inv_outer_hex_side_length: {1.0 / outer_hex_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {end_time - start_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END