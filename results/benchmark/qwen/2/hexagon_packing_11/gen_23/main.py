# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create vertices of a regular hexagon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    vertices = np.column_stack([center[0] + side_length * np.cos(angles),
                               center[1] + side_length * np.sin(angles)])
    return vertices[:-1]  # Remove last vertex to close the polygon

def hexagon_vertices(position, rotation_deg, side_length=1):
    """Get vertices of a hexagon at given position and rotation"""
    return create_regular_hexagon(position, side_length, rotation_deg)

def point_in_hexagon(point, hex_vertices):
    """Check if point is inside hexagon using ray casting method"""
    x, y = point
    n = len(hex_vertices)
    inside = False

    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def hexagons_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using bounding box and then SAT"""
    # Quick bounding box check
    min1 = np.min(hex1_vertices, axis=0)
    max1 = np.max(hex1_vertices, axis=0)
    min2 = np.min(hex2_vertices, axis=0)
    max2 = np.max(hex2_vertices, axis=0)

    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False

    # More precise SAT check
    # Collect all edges of both hexagons
    edges1 = []
    edges2 = []

    for i in range(len(hex1_vertices)):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1) % len(hex1_vertices)]
        edge = p2 - p1
        edges1.append(edge)

    for i in range(len(hex2_vertices)):
        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i+1) % len(hex2_vertices)]
        edge = p2 - p1
        edges2.append(edge)

    # Check separating axes (normals to edges)
    all_edges = edges1 + edges2

    for edge in all_edges:
        # Normal to this edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:  # Avoid zero vectors
            normal = normal / norm

        # Project both hexagons onto this axis
        proj1 = [np.dot(vertex, normal) for vertex in hex1_vertices]
        proj2 = [np.dot(vertex, normal) for vertex in hex2_vertices]

        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        # If projections don't overlap, there's a separating axis
        if max1 < min2 or max2 < min1:
            return False

    return True

def get_outer_hexagon_radius(inner_hex_data, outer_center=(0,0)):
    """Estimate required radius for outer hexagon"""
    # Get all vertices of inner hexagons
    max_dist = 0
    for i in range(len(inner_hex_data)):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        vertices = hexagon_vertices(pos, rot, 1)

        # Find max distance from center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist + 0.1  # Add small margin

def evaluate_packing(inner_hex_data, outer_center=(0,0)):
    """Evaluate if a packing is valid and return penalty score"""
    n = len(inner_hex_data)
    penalty = 0

    # Get all hexagon vertices
    hex_vertices = []
    for i in range(n):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        vertices = hexagon_vertices(pos, rot, 1)
        hex_vertices.append(vertices)

    # Check pairwise overlaps
    for i in range(n):
        for j in range(i+1, n):
            if hexagons_overlap(hex_vertices[i], hex_vertices[j]):
                penalty += 1000  # Large penalty for overlap

    # Check containment in outer hexagon
    outer_radius = get_outer_hexagon_radius(inner_hex_data, outer_center)
    outer_hex = create_regular_hexagon(outer_center, outer_radius, 0)

    # Each vertex must be inside outer hexagon
    for i in range(n):
        for vertex in hex_vertices[i]:
            if not point_in_hexagon(vertex, outer_hex):
                # Calculate how much it exceeds
                dist_to_center = np.linalg.norm(np.array(vertex) - np.array(outer_center))
                penalty += (dist_to_center - outer_radius)**2 * 100

    # Return penalty (smaller means better)
    return penalty

def generate_random_valid_config():
    """Generate a random valid configuration of 11 hexagons"""
    # Generate random positions and rotations for 11 hexagons
    configs = []
    
    # Try several random configurations
    for attempt in range(10000):
        # Random positions within a reasonable bounds
        positions = np.random.uniform(low=-5, high=5, size=(11, 2))
        rotations = np.random.uniform(low=0, high=360, size=11)
        
        # Create configuration array
        config = np.column_stack([positions, rotations])
        
        # Check if this is a valid configuration
        try:
            penalty = evaluate_packing(config)
            if penalty < 100:  # Only keep valid configurations
                configs.append((config, penalty))
        except:
            continue
    
    if configs:
        # Return the best valid configuration found
        best_config, _ = min(configs, key=lambda x: x[1])
        return best_config
    else:
        # Fallback to some reasonable configuration
        return np.array([
            [0, 0, 0],
            [1.732, 0, 0],
            [-1.732, 0, 0],
            [0.866, 1.5, 0],
            [-0.866, 1.5, 0],
            [0.866, -1.5, 0],
            [-0.866, -1.5, 0],
            [2.598, 1.5, 0],
            [-2.598, 1.5, 0],
            [2.598, -1.5, 0],
            [-2.598, -1.5, 0]
        ])

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use Monte Carlo approach to sample random configurations
    best_config = generate_random_valid_config()
    
    # Refine the best configuration found
    best_penalty = evaluate_packing(best_config)
    
    # Try to improve further with random search
    for _ in range(5000):
        # Small random perturbations
        test_config = best_config.copy()
        # Perturb positions slightly
        test_config[:, :2] += np.random.normal(0, 0.1, (11, 2))
        # Perturb rotations slightly
        test_config[:, 2] += np.random.normal(0, 5, 11)
        test_config[:, 2] = test_config[:, 2] % 360  # Keep rotations in [0,360]
        
        penalty = evaluate_packing(test_config)
        if penalty < best_penalty and penalty < 100:  # Valid configuration
            best_config = test_config
            best_penalty = penalty
    
    # Calculate final outer hexagon size
    outer_radius = get_outer_hexagon_radius(best_config)
    outer_hex_side_length = outer_radius
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin

    print(f"Monte Carlo optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Outer hex side length: {outer_hex_side_length:.6f}")
    print(f"Inverse outer hex side length: {1/outer_hex_side_length:.6f}")

    return best_config, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
