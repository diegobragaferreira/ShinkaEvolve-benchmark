# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time
from numba import jit

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

def get_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Get shapely polygon representation of hexagon"""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(hex_poly, outer_poly):
    """Check if hexagon is completely contained within outer hexagon"""
    return outer_poly.contains(hex_poly) or (outer_poly.intersects(hex_poly) and
                                           outer_poly.intersection(hex_poly).area == hex_poly.area)

def calculate_outer_hexagon_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed to contain all inner hexagons"""
    max_dist = 0
    outer_center = (0, 0)

    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)

    # Find maximum distance from center
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)

    # Add buffer for safety and account for hexagon shape
    return max_dist * 1.1  # Safety factor

def hexagon_distance(pos1, pos2):
    """Calculate Euclidean distance between two hexagon centers"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def compute_forces(positions, angles, outer_radius):
    """Compute net forces on each hexagon based on physics simulation"""
    n = len(positions)
    forces = np.zeros((n, 2))  # Forces on each hexagon (dx, dy)
    
    # Repulsion forces between hexagons (inverse square law)
    repulsion_strength = 100.0
    min_distance = 1.5  # Minimum distance to avoid singularity
    
    for i in range(n):
        for j in range(i+1, n):
            dist = hexagon_distance(positions[i], positions[j])
            if dist > 0 and dist < 5.0:  # Only consider nearby hexagons
                # Repulsive force (inverse square law)
                force_magnitude = repulsion_strength / (dist * dist + 0.1)
                # Direction from j to i
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                # Normalize and apply force
                if dist > 0:
                    norm = np.sqrt(dx*dx + dy*dy)
                    dx /= norm
                    dy /= norm
                    forces[i][0] += force_magnitude * dx
                    forces[i][1] += force_magnitude * dy
                    forces[j][0] -= force_magnitude * dx
                    forces[j][1] -= force_magnitude * dy
    
    # Gravitational pull towards center
    gravity_strength = 0.1
    for i in range(n):
        dx = 0 - positions[i][0]
        dy = 0 - positions[i][1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            norm = np.sqrt(dx*dx + dy*dy)
            dx /= norm
            dy /= norm
            forces[i][0] += gravity_strength * dx
            forces[i][1] += gravity_strength * dy
    
    # Boundary constraints (penalty for going outside outer hexagon)
    boundary_penalty = 50.0
    for i in range(n):
        dx = 0 - positions[i][0]
        dy = 0 - positions[i][1]
        dist = np.sqrt(dx*dx + dy*dy)
        # Push back if too far from center (outside the outer hexagon)
        if dist > outer_radius - 0.5:  # Add some buffer
            # Force away from center
            if dist > 0:
                norm = np.sqrt(dx*dx + dy*dy)
                dx /= norm
                dy /= norm
                forces[i][0] -= boundary_penalty * dx
                forces[i][1] -= boundary_penalty * dy
    
    return forces

def simulate_physics_step(positions, velocities, angles, forces, dt=0.01, damping=0.9):
    """Advance the physics simulation by one step"""
    new_positions = positions.copy()
    new_velocities = velocities.copy()
    
    for i in range(len(positions)):
        # Update velocity with forces
        new_velocities[i][0] += forces[i][0] * dt
        new_velocities[i][1] += forces[i][1] * dt
        
        # Apply damping
        new_velocities[i][0] *= damping
        new_velocities[i][1] *= damping
        
        # Update position
        new_positions[i][0] += new_velocities[i][0] * dt
        new_positions[i][1] += new_velocities[i][1] * dt
    
    return new_positions, new_velocities

def check_solution_feasibility(positions, angles, outer_radius):
    """Check if the current solution is feasible (no overlaps, fully contained)"""
    # Create hexagon polygons
    hexagons = []
    for i in range(len(positions)):
        hex_poly = get_hexagon_polygon(positions[i][0], positions[i][1], angles[i])
        hexagons.append(hex_poly)
    
    # Check containment
    outer_hexagon = get_hexagon_polygon(0, 0, 0, outer_radius)
    
    for hex_poly in hexagons:
        if not check_containment(hex_poly, outer_hexagon):
            return False
    
    # Check overlaps
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if hexagons[i].intersects(hexagons[j]):
                return False
    
    return True

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a gravity-based physics simulation approach.
    
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    max_time = 175  # Leave some margin for cleanup

    # Initialize with a good starting configuration
    initial_positions = [
        [0.0, 0.0],     # center
        [-2.0, 0.0],    # left
        [2.0, 0.0],     # right
        [0.0, 2.0],     # top
        [0.0, -2.0],    # bottom
        [-1.0, 1.0],    # top-left
        [1.0, 1.0],     # top-right
        [-1.0, -1.0],   # bottom-left
        [1.0, -1.0],    # bottom-right
        [-2.0, 1.5],    # further top-left
        [2.0, 1.5]      # further top-right
    ]
    
    # Adjust spacing to allow for better packing and make it more compact initially
    for i in range(len(initial_positions)):
        initial_positions[i][0] *= 0.8
        initial_positions[i][1] *= 0.8
    
    initial_angles = [0.0] * 11
    positions = np.array(initial_positions)
    angles = np.array(initial_angles)
    
    # Initialize velocities
    velocities = np.zeros_like(positions)
    
    # Calculate initial outer radius
    outer_radius = calculate_outer_hexagon_radius(positions, angles)
    
    # Physics simulation parameters
    dt = 0.005
    damping = 0.99
    max_steps = 5000
    
    # Early stopping criteria
    prev_energy = float('inf')
    energy_threshold = 1e-6
    consecutive_stable_steps = 0
    max_consecutive_stable = 100
    
    # Main physics simulation loop
    for step in range(max_steps):
        if time.time() - start_time > max_time:
            break
            
        # Compute forces
        forces = compute_forces(positions, angles, outer_radius)
        
        # Update physics
        positions, velocities = simulate_physics_step(positions, velocities, angles, forces, dt, damping)
        
        # Periodically recalculate outer radius
        if step % 50 == 0:
            outer_radius = calculate_outer_hexagon_radius(positions, angles)
        
        # Check for convergence (low energy)
        energy = np.sum(forces**2)
        if abs(prev_energy - energy) < energy_threshold:
            consecutive_stable_steps += 1
        else:
            consecutive_stable_steps = 0
            prev_energy = energy
        
        # Early termination if stable
        if consecutive_stable_steps >= max_consecutive_stable:
            break
    
    # Perform final feasibility check and fix if needed
    if not check_solution_feasibility(positions, angles, outer_radius):
        # If solution is invalid, return initial configuration
        positions = np.array([
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
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_radius = 8.0
        angles = np.array([0.0] * 11)
        # Recalculate final outer radius
        outer_radius = calculate_outer_hexagon_radius(positions[:, :2], angles)
    
    # Create inner hex data
    inner_hex_data = np.column_stack([positions, angles])
    
    # Create outer hex data (centered)
    outer_hex_data = np.array([0, 0, 0])
    
    # Convert to side length for regular hexagon (radius = side_length * sqrt(3)/2)
    outer_hex_side_length = outer_radius / (np.sqrt(3) / 2)
    
    elapsed_time = time.time() - start_time
    print(f"Physics simulation completed in {elapsed_time:.2f} seconds")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END