# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import minimize
import time

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = math.radians(angle_deg)
    # Vertices of a regular hexagon with side_length=1 centered at origin
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = math.cos(theta)
        y = math.sin(theta)
        base_vertices.append((x, y))

    # Scale and translate
    vertices = [(center_x + side_length * vx, center_y + side_length * vy) for vx, vy in base_vertices]
    return vertices

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hexagon_vertices)
    return outer_poly.contains(inner_poly)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def hexagon_distance_squared(h1_center, h2_center):
    """Calculate squared distance between hexagon centers."""
    return (h1_center[0] - h2_center[0])**2 + (h1_center[1] - h2_center[1])**2

def calculate_outer_hexagon_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed to contain all inner hexagons."""
    max_dist = 0
    outer_center = (0, 0)
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = generate_hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)

    # Find maximum distance from center
    for vertex in all_vertices:
        dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)

    # Add buffer for safety
    return max_dist * 1.1

def compute_energy_for_hexagon(hex_idx, positions, angles, outer_radius):
    """Compute energy contribution for a specific hexagon."""
    # Energy components
    energy = 0.0
    
    # Distance penalty - push hexagons away from each other
    for i in range(len(positions)):
        if i != hex_idx:
            dist_sq = hexagon_distance_squared(positions[hex_idx], positions[i])
            # Avoid division by zero
            if dist_sq > 0.01:
                # Repulsive force (inverse square law)
                energy += 1.0 / dist_sq
    
    # Boundary penalty - hexagon gets energy penalty when near boundary
    hex_center = positions[hex_idx]
    distance_from_center = math.sqrt(hex_center[0]**2 + hex_center[1]**2)
    
    # If hexagon center is beyond outer radius, give strong penalty
    if distance_from_center > outer_radius * 0.95:
        energy += 1000.0 * (distance_from_center - outer_radius * 0.95)
    
    return energy

def compute_total_energy(positions, angles, outer_radius):
    """Compute total system energy."""
    total_energy = 0.0
    for i in range(len(positions)):
        total_energy += compute_energy_for_hexagon(i, positions, angles, outer_radius)
    return total_energy

def compute_force_on_hexagon(hex_idx, positions, angles, outer_radius):
    """Compute force on a hexagon based on energy gradient."""
    forces = []
    eps = 1e-6
    
    # Compute approximate gradient numerically
    base_energy = compute_energy_for_hexagon(hex_idx, positions, angles, outer_radius)
    
    # Compute force in x direction
    test_positions = [list(pos) for pos in positions]
    test_positions[hex_idx][0] += eps
    test_energy = compute_energy_for_hexagon(hex_idx, test_positions, angles, outer_radius)
    fx = -(test_energy - base_energy) / eps
    
    # Compute force in y direction  
    test_positions[hex_idx][0] -= eps
    test_positions[hex_idx][1] += eps
    test_energy = compute_energy_for_hexagon(hex_idx, test_positions, angles, outer_radius)
    fy = -(test_energy - base_energy) / eps
    
    return np.array([fx, fy])

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid geometric energy optimization approach.
    """
    start_time = time.time()
    
    # Phase 1: Create initial geometric configuration based on hexagonal tiling principles
    # Place hexagons in a natural tiling pattern with some randomness to escape local minima
    positions = []
    angles = []
    
    # Central hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Surrounding hexagons in 6 directions (like a honeycomb)
    directions = [
        (0, 1),   # up
        (0.866, 0.5),  # up-right
        (0.866, -0.5), # down-right
        (0, -1),  # down
        (-0.866, -0.5), # down-left
        (-0.866, 0.5)   # up-left
    ]
    
    # Place 6 hexagons around center
    for i, (dx, dy) in enumerate(directions):
        positions.append([dx * 2.0, dy * 2.0])
        angles.append(0.0)
    
    # Place 4 additional hexagons in strategic positions
    additional_positions = [
        (-1.5, 1.5),
        (1.5, 1.5),
        (-1.5, -1.5),
        (1.5, -1.5)
    ]
    
    for pos in additional_positions:
        positions.append(list(pos))
        angles.append(0.0)
        
    # Ensure we have exactly 11 positions
    positions = positions[:11]
    angles = angles[:11]

    # Phase 2: Optimization using energy-based approach with geometric constraints
    max_iter = 1000
    
    # Find initial outer radius
    outer_radius = calculate_outer_hexagon_radius(positions, angles)
    
    # Use scipy.optimize for continuous optimization
    def objective_function(params):
        # Reshape params into positions and angles
        current_positions = []
        current_angles = []
        
        for i in range(11):
            current_positions.append([params[2*i], params[2*i+1]])
            current_angles.append(params[22+i])
        
        # Check constraints
        penalty = 0.0
        
        # Check containment: Create outer hexagon vertices
        outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_radius)
        
        # Check if all inner hexagons are contained
        for i in range(len(current_positions)):
            hex_vertices = generate_hexagon_vertices(current_positions[i][0], current_positions[i][1], current_angles[i])
            if not check_containment(hex_vertices, outer_vertices):
                penalty += 1000000.0  # Strong penalty for containment violation
                
        # Check overlaps
        for i in range(len(current_positions)):
            for j in range(i+1, len(current_positions)):
                hex1_vertices = generate_hexagon_vertices(current_positions[i][0], current_positions[i][1], current_angles[i])
                hex2_vertices = generate_hexagon_vertices(current_positions[j][0], current_positions[j][1], current_angles[j])
                if check_overlap(hex1_vertices, hex2_vertices):
                    penalty += 1000000.0  # Strong penalty for overlap
        
        # Objective function: minimize 1/(outer_radius) + penalty 
        # But we want to maximize 1/outer_radius, so we're minimizing -1/outer_radius
        # However, we want to minimize the entire cost including constraints
        if penalty > 0:
            return penalty + 1.0 / outer_radius
        
        # Calculate the actual geometric packing quality
        # We want to minimize the outer radius, which means maximizing 1/outer_radius
        # So we return -1.0 / outer_radius (as negative because we use minimization)
        return -1.0 / outer_radius
    
    # Prepare initial parameters (flatten positions and angles)
    initial_params = []
    for i in range(11):
        initial_params.extend([positions[i][0], positions[i][1], angles[i]])
    
    # Phase 3: Run optimization with bounds
    bounds = []
    # Position bounds
    for i in range(22):
        bounds.append((-15.0, 15.0))
    # Angle bounds
    for i in range(11):
        bounds.append((-180.0, 180.0))
    
    # Use L-BFGS-B optimization method which handles bounds well
    try:
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=None
        )
        
        if result.success:
            # Extract optimized parameters
            final_positions = []
            final_angles = []
            for i in range(11):
                final_positions.append([result.x[2*i], result.x[2*i+1]])
                final_angles.append(result.x[22+i])
        else:
            # Fallback to initial positions if optimization fails
            final_positions = positions
            final_angles = angles
    except Exception as e:
        # Fallback to initial positions
        final_positions = positions
        final_angles = angles
    
    # Final refinement: recompute outer radius with optimized positions
    outer_radius_final = calculate_outer_hexagon_radius(final_positions, final_angles)
    
    # Construct output arrays
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [final_positions[i][0], final_positions[i][1], final_angles[i]]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    
    # Convert radius back to side length for regular hexagon
    # For a regular hexagon, side length = radius
    outer_hex_side_length = outer_radius_final
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END