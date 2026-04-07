# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon with given center and rotation"""
    angle_offset = math.radians(rotation)
    radius = 1
    vertices = []
    for i in range(6):
        angle = angle_offset + i * math.pi / 3
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices

def calculate_outer_hex_radius(inner_hex_data):
    """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        hexagon = create_unit_hexagon(center, rotation)
        all_vertices.extend(hexagon)

    max_distance = 0
    for vertex in all_vertices:
        distance = math.sqrt(vertex[0]**2 + vertex[1]**2)
        max_distance = max(max_distance, distance)

    return max_distance + 0.1

def hexagon_vertices(center, rotation):
    """Get vertices of a unit hexagon at given center and rotation"""
    angle_offset = math.radians(rotation)
    radius = 1
    vertices = []
    for i in range(6):
        angle = angle_offset + i * math.pi / 3
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def distance_to_outer_hexagon(center, outer_radius):
    """Distance from center to boundary of outer hexagon"""
    # For a regular hexagon centered at origin with side length R
    # The distance to the boundary in any direction is R
    return max(0, np.linalg.norm(center) - outer_radius)

def hexagon_distance(h1, h2):
    """Minimum distance between two hexagons"""
    # Simple approximation: distance between centers minus sum of radii
    # For unit hexagons, effective radius is about 1.0
    c1 = np.mean(h1, axis=0)
    c2 = np.mean(h2, axis=0)
    dist = np.linalg.norm(c1 - c2)
    return dist - 2.0  # subtract 2*radius for unit hexagons

def check_constraints(hex_data, outer_radius):
    """Check if hexagon configuration satisfies all constraints"""
    # Check containment
    for i in range(len(hex_data)):
        center = (hex_data[i][0], hex_data[i][1])
        if np.linalg.norm(np.array(center)) > outer_radius - 1.0:
            return False
    
    # Check overlaps - simple pairwise check for now
    for i in range(len(hex_data)):
        h1 = hexagon_vertices((hex_data[i][0], hex_data[i][1]), hex_data[i][2])
        for j in range(i+1, len(hex_data)):
            h2 = hexagon_vertices((hex_data[j][0], hex_data[j][1]), hex_data[j][2])
            if hexagon_distance(h1, h2) < 0:
                return False
    return True

def objective_and_gradient(hex_params, outer_radius):
    """Objective function and gradient for optimization"""
    # Extract positions (we'll assume 12 hexagons with symmetric positioning)
    # Parameterization: positions in polar coordinates, then convert to Cartesian
    n = 12
    # First 12 values are x-coordinates, next 12 are y-coordinates
    positions = hex_params.reshape(2, -1).T  # Shape: (12, 2)
    
    # Calculate current outer radius needed
    max_dist = 0
    for i in range(12):
        dist = np.linalg.norm(positions[i])
        max_dist = max(max_dist, dist)
    
    # Simple objective: minimize outer radius (maximize 1/outer_radius)
    objective_value = max_dist + 0.1  # add small buffer
    
    # Gradient calculation (simplified)
    gradient = np.zeros_like(hex_params)
    for i in range(12):
        if max_dist > 0:
            grad_x = positions[i][0] / max_dist
            grad_y = positions[i][1] / max_dist
            gradient[2*i] = grad_x  # d/dx
            gradient[2*i+1] = grad_y  # d/dy
    
    return objective_value, gradient

def generate_initial_symmetric_config():
    """Generate high-quality symmetric configuration"""
    # Core idea: arrange hexagons in concentric rings with rotational symmetry
    # This reduces the number of independent variables significantly
    
    # Center hexagon
    hex_data = [[0, 0, 0]]  # center
    
    # Ring 1: 6 hexagons around center (at distance 2)
    for i in range(6):
        angle = i * math.pi / 3  # 60 degree increments
        x = 2 * math.cos(angle)
        y = 2 * math.sin(angle)
        hex_data.append([x, y, 0])
    
    # Ring 2: 5 hexagons (not including the one opposite to center)
    for i in range(5):
        angle = i * 2 * math.pi / 5  # 72 degree increments
        x = 3.464 * math.cos(angle)  # sqrt(3) * 2
        y = 3.464 * math.sin(angle)
        hex_data.append([x, y, 0])
    
    # Trim to exactly 12
    return np.array(hex_data[:12])

def project_onto_hexagon_boundary(positions, outer_radius):
    """Project positions onto outer hexagon boundary if they exceed it"""
    # For a hexagon centered at origin with radius outer_radius
    # Each side is at 60 degree intervals
    projected = []
    for pos in positions:
        norm = np.linalg.norm(pos)
        if norm > outer_radius:
            # Project onto boundary
            unit_vec = pos / norm
            projected_pos = unit_vec * outer_radius
            projected.append(projected_pos)
        else:
            projected.append(pos)
    return np.array(projected)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Generate initial symmetric configuration
    hex_data = generate_initial_symmetric_config()
    
    # Initialize optimization variables
    # Only optimize positions, keeping rotations fixed at 0 for now
    optimization_vars = hex_data[:, :2].flatten()  # x, y coordinates
    
    # Optimization parameters
    max_iterations = 100
    tolerance = 1e-6
    
    # Direct optimization approach
    def objective_func(params):
        # Reshape parameters back to positions
        positions = params.reshape(-1, 2)
        temp_data = hex_data.copy()
        temp_data[:, 0] = positions[:, 0]
        temp_data[:, 1] = positions[:, 1]
        
        # Calculate outer radius
        outer_radius = calculate_outer_hex_radius(temp_data)
        return outer_radius
    
    def constraint_func(params):
        # Reshape parameters back to positions
        positions = params.reshape(-1, 2)
        temp_data = hex_data.copy()
        temp_data[:, 0] = positions[:, 0]
        temp_data[:, 1] = positions[:, 1]
        
        # Check constraints
        outer_radius = calculate_outer_hex_radius(temp_data)
        valid = check_constraints(temp_data, outer_radius)
        
        # Return penalty for constraint violations
        if valid:
            return 0
        else:
            return 1000  # Large penalty
    
    # Apply bounds for positions  
    bounds = [(-5, 5) for _ in range(24)]
    
    # Use scipy optimization with analytical approach
    try:
        # First optimize positions only
        result = minimize(
            objective_func, 
            optimization_vars, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )
        
        if result.success:
            optimized_positions = result.x.reshape(-1, 2)
            hex_data[:, 0] = optimized_positions[:, 0]
            hex_data[:, 1] = optimized_positions[:, 1]
    except Exception as e:
        # Fall back to original configuration if optimization fails
        pass
    
    # Calculate final outer hexagon size
    final_outer_radius = calculate_outer_hex_radius(hex_data)
    outer_hex_side_length = final_outer_radius + 0.2  # Add margin
    
    # Return result
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END