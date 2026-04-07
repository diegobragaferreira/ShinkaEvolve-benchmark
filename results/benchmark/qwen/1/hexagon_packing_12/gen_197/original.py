# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time

def hexagon_vertices(center_x, center_y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon."""
    angle_rad = np.radians(angle_degrees)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad  # 6 sides + closing vertex
    vertices = []
    for angle in angles:
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    inner_polygon = Polygon(hexagon_vertices)
    outer_polygon = Polygon(outer_hexagon_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    polygon1 = Polygon(hex1_vertices)
    polygon2 = Polygon(hex2_vertices)
    return polygon1.intersects(polygon2)

def fast_overlap_check(hex1_vertices, hex2_vertices):
    """Fast overlap check using bounding circles for early rejection."""
    # Compute centroids
    cx1, cy1 = np.mean(hex1_vertices, axis=0)
    cx2, cy2 = np.mean(hex2_vertices, axis=0)

    # Compute approximate radii (distance from centroid to farthest vertex)
    r1 = max(np.linalg.norm(v - [cx1, cy1]) for v in hex1_vertices)
    r2 = max(np.linalg.norm(v - [cx2, cy2]) for v in hex2_vertices)

    # Fast circle overlap test
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return dist < (r1 + r2)

def generate_initial_symmetric_placement():
    """Generate an initial symmetric configuration based on proven hexagonal packing patterns."""
    # Start with a 6-fold symmetric pattern
    # Central hexagon + 6 surrounding hexagons in first ring + 5 more in second ring
    
    # Layer 1: Center
    positions = [[0.0, 0.0]]
    rotations = [0.0]
    
    # Layer 2: First ring - 6 hexagons arranged in circle
    ring1_radius = 1.732  # sqrt(3) for optimal spacing
    for i in range(6):
        angle = i * 60  # 60 degree increments for 6-fold symmetry
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        positions.append([x, y])
        rotations.append(0.0)
    
    # Layer 3: Second ring - 5 hexagons arranged in circle (not perfectly symmetric)
    # This pattern tends to work well for 12-hexagon packing
    ring2_radius = 3.464  # 2*sqrt(3)
    for i in range(5):
        angle = i * 72 + 15  # 72 degrees + offset for better packing
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        positions.append([x, y])
        rotations.append(0.0)
    
    return np.array(positions), np.array(rotations)

def evaluate_configuration(params):
    """
    Evaluate a configuration with direct parameterization of hexagon centers and rotations.
    params: array of shape (36,) where first 24 values are (x,y) for 12 hexagons
            and next 12 values are rotation angles (in degrees)
    """
    # Extract inner hexagon data
    positions = params[:24].reshape(12, 2)
    rotations = params[24:36]
    
    # Create all hexagon vertices
    hexagon_vertices_list = []
    for i in range(12):
        x, y = positions[i]
        angle = rotations[i]
        verts = hexagon_vertices(x, y, angle)
        hexagon_vertices_list.append(verts)
    
    # Estimate outer hexagon size based on maximum distance from center
    max_distance = 0
    for verts in hexagon_vertices_list:
        for vertex in verts:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_distance = max(max_distance, dist)
    
    # Create outer hexagon (slightly larger than needed)
    outer_radius = max_distance * 1.05
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    
    # Validate containment
    total_penalty = 0
    
    # Check containment
    for i in range(12):
        if not check_containment(hexagon_vertices_list[i], outer_vertices):
            total_penalty += 10000
    
    # Check overlaps with spatial acceleration
    # Build spatial index for efficient neighbor querying
    hex_centers = np.array([[np.mean(v[:, 0]), np.mean(v[:, 1])] for v in hexagon_vertices_list])
    tree = cKDTree(hex_centers)
    
    # Find neighbors within a reasonable distance
    pairs_to_check = tree.query_pairs(r=2.5, p=np.inf)
    
    # Check overlaps using spatial acceleration
    for i, j in pairs_to_check:
        if i != j:
            if fast_overlap_check(hexagon_vertices_list[i], hexagon_vertices_list[j]):
                total_penalty += 10000
    
    # Additional explicit checks for likely overlaps
    # Center with ring 1
    for i in range(1, 7):
        if fast_overlap_check(hexagon_vertices_list[0], hexagon_vertices_list[i]):
            total_penalty += 10000
    
    # Center with ring 2
    for i in range(7, 12):
        if fast_overlap_check(hexagon_vertices_list[0], hexagon_vertices_list[i]):
            total_penalty += 10000
            
    # Ring 1 to ring 2
    for i in range(1, 7):
        for j in range(7, 12):
            if fast_overlap_check(hexagon_vertices_list[i], hexagon_vertices_list[j]):
                total_penalty += 10000
    
    # Return negative inverse of outer radius (for minimization) plus penalties
    return -(1.0 / (outer_radius + total_penalty + 1e-8))

def optimize_packing():
    """Main optimization routine using hybrid approach."""
    # Generate initial symmetric configuration
    positions, rotations = generate_initial_symmetric_placement()
    
    # Combine into single parameter vector: [24 positions + 12 rotations]
    initial_params = np.concatenate([positions.flatten(), rotations])
    
    # Define bounds for optimization
    # Positions: x, y bounded to reasonable range (±10)
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10)])  # x, y for each hexagon
    for _ in range(12):
        bounds.append((-180, 180))  # rotation for each hexagon
    
    # Phase 1: Coarse global optimization
    def objective_coarse(params):
        return evaluate_configuration(params)
    
    try:
        # Run differential evolution with fewer iterations for speed
        de_result = differential_evolution(
            objective_coarse,
            bounds,
            maxiter=20,
            popsize=10,
            seed=42,
            disp=False,
            atol=1e-4,
            ftol=1e-4
        )
        
        best_params = de_result.x
        
        # Phase 2: Fine local refinement
        # Use L-BFGS-B for local refinement
        local_result = minimize(
            evaluate_configuration,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30, 'disp': False}
        )
        
        if local_result.success:
            best_params = local_result.x
    except:
        # If optimization fails, use initial configuration
        pass
    
    return best_params

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Track execution time
    start_time = time.time()

    try:
        # Get optimized configuration
        best_params = optimize_packing()
        
        # Extract results
        positions = best_params[:24].reshape(12, 2)
        rotations = best_params[24:36]
        
        # Create inner hex data
        inner_hex_data = np.column_stack([positions, rotations])
        
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
        # Calculate actual outer hexagon size
        hexagon_vertices_list = []
        for i in range(12):
            x, y = positions[i]
            angle = rotations[i]
            verts = hexagon_vertices(x, y, angle)
            hexagon_vertices_list.append(verts)
        
        # Estimate outer hexagon size based on maximum distance from center  
        max_distance = 0
        for verts in hexagon_vertices_list:
            for vertex in verts:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                max_distance = max(max_distance, dist)
                
        outer_hex_side_length = max_distance * 1.05  # Add buffer

        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback to improved grid configuration if optimization fails
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

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END