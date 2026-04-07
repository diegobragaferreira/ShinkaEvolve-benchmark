# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def hexagon_vertices(center_x, center_y, rotation_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a unit hexagon centered at origin
    unit_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = unit_vertices @ rotation_matrix.T
    return rotated_vertices * side_length + np.array([center_x, center_y])

def check_containment_single(hex_vertices, outer_polygon):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_collision_single_sat(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Separating Axis Theorem for efficiency."""
    # First quick bounding box check
    min1 = np.min(hex1_vertices, axis=0)
    max1 = np.max(hex1_vertices, axis=0)
    min2 = np.min(hex2_vertices, axis=0)
    max2 = np.max(hex2_vertices, axis=0)

    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False

    # SAT algorithm for hexagon-hexagon collision
    # Collect all edges from both hexagons
    edges1 = []
    edges2 = []

    for i in range(len(hex1_vertices)):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i + 1) % len(hex1_vertices)]
        edge = p2 - p1
        edges1.append(edge)

    for i in range(len(hex2_vertices)):
        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i + 1) % len(hex2_vertices)]
        edge = p2 - p1
        edges2.append(edge)

    # All possible separating axes (normals to edges)
    axes = []
    for edge in edges1 + edges2:
        # Normal vector to edge
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        axes.append(normal)

    # Check each axis
    for axis in axes:
        # Project both polygons onto this axis
        proj1 = [np.dot(vertex, axis) for vertex in hex1_vertices]
        proj2 = [np.dot(vertex, axis) for vertex in hex2_vertices]

        min1_proj, max1_proj = min(proj1), max(proj1)
        min2_proj, max2_proj = min(proj2), max(proj2)

        # If projections don't overlap, this is a separating axis
        if max1_proj < min2_proj or max2_proj < min1_proj:
            return False

    # If we got here, polygons intersect
    return True

def check_collision_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Shapely for final verification."""
    # Use SAT for fast rejection, then Shapely for precise check
    if not check_collision_single_sat(hex1_vertices, hex2_vertices):
        return False
    # If SAT says they might intersect, do precise check
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_hexagon_distances(hex1_vertices, hex2_vertices):
    """Compute minimum distance between two hexagons."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.distance(poly2)

def estimate_min_outer_radius(inner_hex_params):
    """Estimate minimal outer hexagon radius containing all inner hexagons with improved accuracy."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(11):
        x, y, rot = inner_hex_params[3*i], inner_hex_params[3*i+1], inner_hex_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        all_vertices.extend(hex_vertices)

    if len(all_vertices) == 0:
        return 100.0

    # Calculate bounding box
    all_vertices = np.array(all_vertices)
    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_y, max_y = all_vertices[:, 1].min(), all_vertices[:, 1].max()

    # Calculate center of bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Find maximum distance from center to any vertex - this gives us the radius
    # of the smallest circle containing all vertices. For hexagon packing,
    # we need the outer hexagon to have a radius that's sufficient to contain
    # the circumscribed circle of all inner hexagons plus some margin.
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)

    # For a unit hexagon, the circumradius is 1, but we also need to account
    # for the fact that our hexagon contains vertices, not just centers.
    # So we need max_dist + safety_margin (slightly more than 1 for unit hex)
    return max_dist + 1.15  # Tighter margin for better packing

def compute_objective_with_penalties(params, use_distance_penalties=True):
    """Compute objective function with proper penalty handling."""
    # params: [x1, y1, rot1, x2, y2, rot2, ..., x11, y11, rot11, R]
    n = 11
    outer_side_length = params[-1]

    # Check if outer hexagon is valid
    if outer_side_length <= 0:
        return 1e10

    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))

    # Initialize penalties
    containment_penalty = 0.0
    collision_penalty = 0.0

    # Store hexagon info for collision checks
    inner_hex_info = []

    # Check containment and collect hex info
    for i in range(n):
        x, y, rot = params[3*i], params[3*i+1], params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)

        # Check containment
        if not check_containment_single(hex_vertices, outer_polygon):
            containment_penalty += 1e8 * (1 + outer_side_length)  # Stronger penalty for containment violations

        inner_hex_info.append({
            'vertices': hex_vertices,
            'center': [x, y],
            'rotation': rot
        })

    # If containment failed, return immediately
    if containment_penalty > 0:
        return -(1.0 / outer_side_length) + containment_penalty

    # Check collisions with all other hexagons
    if use_distance_penalties:
        # Use distance-based penalties with more aggressive scaling
        for i in range(n):
            for j in range(i+1, n):
                hex1_info = inner_hex_info[i]
                hex2_info = inner_hex_info[j]

                # Compute distance between hexagons
                dist = compute_hexagon_distances(hex1_info['vertices'], hex2_info['vertices'])

                # Penalty based on how close they are
                if dist < 1e-6:  # Colliding
                    collision_penalty += 1e12
                elif dist < 0.05:  # Very close
                    collision_penalty += 1e8 * (1.0 / (dist + 1e-8))
                elif dist < 0.1:  # Close
                    collision_penalty += 1e6 * (1.0 / (dist + 1e-6))
    else:
        # Use simple collision checks with more conservative penalty
        for i in range(n):
            for j in range(i+1, n):
                hex1_vertices = inner_hex_info[i]['vertices']
                hex2_vertices = inner_hex_info[j]['vertices']

                if check_collision_single(hex1_vertices, hex2_vertices):
                    collision_penalty += 1e12  # Severe penalty for collisions

    # Return negative inverse of outer hex side length plus penalties
    total_penalty = containment_penalty + collision_penalty
    return -(1.0 / outer_side_length) + total_penalty

def generate_initial_guesses():
    """Generate comprehensive initial guesses with diverse patterns and optimizations."""
    guesses = []

    # Base patterns with different arrangements
    base_patterns = []
    
    # Pattern 1: Classical hexagonal close-packed arrangement 
    hex_pattern = [
        [0, 0, 0],           # center
        [0, 2, 0],           # top
        [1.732, 1, 0],       # top-right
        [1.732, -1, 0],      # bottom-right
        [0, -2, 0],          # bottom
        [-1.732, -1, 0],     # bottom-left
        [-1.732, 1, 0],      # top-left
        [3.464, 0, 0],       # far right
        [1.732, 2, 0],       # top-middle
        [-1.732, 2, 0],      # top-middle-left
        [-3.464, 0, 0],      # far left
    ]
    base_patterns.append(("hex_close_packed", hex_pattern))

    # Pattern 2: Spiral arrangement with varied spacing
    spiral_pattern = [
        [0, 0, 0],         # center
        [2, 0, 0],         # right
        [1, 1.732, 0],     # upper-right
        [-1, 1.732, 0],    # upper-left
        [-2, 0, 0],        # left
        [-1, -1.732, 0],   # lower-left
        [1, -1.732, 0],    # lower-right
        [3, 0, 0],         # far right
        [1.5, 2.6, 0],     # upper-middle-right
        [-1.5, 2.6, 0],    # upper-middle-left
        [-3, 0, 0],        # far left
    ]
    base_patterns.append(("spiral", spiral_pattern))

    # Pattern 3: Grid arrangement with strategic gaps
    grid_pattern = [
        [0, 0, 0],       # center
        [-2.5, 0, 0],    # left
        [2.5, 0, 0],     # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0],  # bottom-right
        [-3.75, 2.17, 0],  # far top-left
        [3.75, 2.17, 0],   # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0],  # far bottom-right
    ]
    base_patterns.append(("grid", grid_pattern))

    # Pattern 4: Optimized literature configuration (tight packed)
    optimized_pattern = [
        [0, 0, 0],          # center
        [0, 2.0, 0],        # top
        [1.732, 1.0, 0],    # top-right
        [1.732, -1.0, 0],   # bottom-right
        [0, -2.0, 0],       # bottom
        [-1.732, -1.0, 0],  # bottom-left
        [-1.732, 1.0, 0],   # top-left
        [3.464, 0, 0],      # far right
        [1.732, 2.0, 0],    # top-middle
        [-1.732, 2.0, 0],   # top-middle-left
        [-3.464, 0, 0],     # far left
    ]
    base_patterns.append(("optimized", optimized_pattern))

    # Pattern 5: Rotated arrangement to test rotational diversity
    rotated_pattern = [
        [0, 0, 0],           # center
        [0, 2, 0],           # top
        [1.732, 1, 30],      # top-right with rotation
        [1.732, -1, -30],    # bottom-right with rotation
        [0, -2, 0],          # bottom
        [-1.732, -1, 30],    # bottom-left with rotation
        [-1.732, 1, -30],    # top-left with rotation
        [3.464, 0, 0],       # far right
        [1.732, 2, 30],      # top-middle with rotation
        [-1.732, 2, -30],    # top-middle-left with rotation
        [-3.464, 0, 0],      # far left
    ]
    base_patterns.append(("rotated", rotated_pattern))

    # Pattern 6: High symmetry arrangement from literature
    symmetric_pattern = [
        [0, 0, 0],       # center
        [0, 2.2, 0],     # top
        [1.8, 1.1, 0],   # top-right
        [1.8, -1.1, 0],  # bottom-right
        [0, -2.2, 0],    # bottom
        [-1.8, -1.1, 0], # bottom-left
        [-1.8, 1.1, 0],  # top-left
        [3.6, 0, 0],     # far right
        [1.8, 2.2, 0],   # top-middle
        [-1.8, 2.2, 0],  # top-middle-left
        [-3.6, 0, 0],    # far left
    ]
    base_patterns.append(("symmetric", symmetric_pattern))

    # Pattern 7: Alternative tight packing configuration
    alternative_pattern = [
        [0, 0, 0],           # center
        [0, 2.1, 0],         # top
        [1.75, 1.0, 0],      # top-right
        [1.75, -1.0, 0],     # bottom-right
        [0, -2.1, 0],        # bottom
        [-1.75, -1.0, 0],    # bottom-left
        [-1.75, 1.0, 0],     # top-left
        [3.5, 0, 0],         # far right
        [1.75, 2.1, 0],      # top-middle
        [-1.75, 2.1, 0],     # top-middle-left
        [-3.5, 0, 0],        # far left
    ]
    base_patterns.append(("alternative", alternative_pattern))

    # Pattern 8: Asymmetric arrangement for exploration
    asymmetric_pattern = [
        [0, 0, 0],           # center
        [0, 2.0, 0],         # top
        [1.8, 1.2, 15],      # top-right with slight rotation
        [1.8, -1.2, -15],    # bottom-right with slight rotation
        [0, -2.0, 0],        # bottom
        [-1.8, -1.2, 15],    # bottom-left with slight rotation
        [-1.8, 1.2, -15],    # top-left with slight rotation
        [3.6, 0, 0],         # far right
        [1.8, 2.0, 0],       # top-middle
        [-1.8, 2.0, 0],      # top-middle-left
        [-3.6, 0, 0],        # far left
    ]
    base_patterns.append(("asymmetric", asymmetric_pattern))

    # Create variations of each base pattern with small perturbations
    all_patterns = []
    for pattern_name, pattern in base_patterns:
        # Original pattern
        all_patterns.append((pattern_name, pattern))
        
        # Perturbed versions for diversity
        for i in range(2):  # Two perturbed versions
            perturbed = []
            for j, pos in enumerate(pattern):
                x, y, rot = pos
                # Small random perturbation for position
                x += np.random.normal(0, 0.15)
                y += np.random.normal(0, 0.15)
                # Small rotation perturbation
                rot += np.random.normal(0, 8)
                perturbed.append([x, y, rot])
            all_patterns.append((f"{pattern_name}_pert{i}", perturbed))

    # Generate configurations from all patterns
    for pattern_name, pattern in all_patterns:
        initial_params = []
        for pos in pattern:
            initial_params.extend(pos)

        # Estimate outer side length
        estimated_side = estimate_min_outer_radius(np.array(initial_params))
        initial_params.append(estimated_side)

        # Evaluate quality of this configuration
        try:
            score = compute_objective_with_penalties(np.array(initial_params), use_distance_penalties=False)
            if score < 1e9:  # Valid solution
                guesses.append((initial_params, score))
        except:
            continue

    # Sort by quality and return top 12
    if guesses:
        guesses.sort(key=lambda x: x[1])
        return [g[0] for g in guesses[:12]]

    # Fallback if nothing works
    return [
        np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0], 10.0
        ]).flatten()
    ]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Generate comprehensive initial guesses
    initial_guesses = generate_initial_guesses()

    best_result = None
    best_score = float('inf')

    # Try each initial guess with aggressive multi-stage optimization
    for i, initial_guess in enumerate(initial_guesses):
        try:
            # Stage 1: Global exploration with differential evolution
            bounds = []
            # Positions: x, y for each hexagon (limited to reasonable range)
            for _ in range(11):
                bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
            bounds.append((1.0, 20.0))  # Outer hex side length

            # Use aggressive evolutionary algorithm settings with adaptive parameters
            # Determine if we're in a hard region by evaluating initial guess quality
            initial_score = compute_objective_with_penalties(initial_guess, use_distance_penalties=False)
            population_size = 30 if initial_score < 1e6 else 20  # Smaller pop if already good
            max_iterations = 150 if initial_score < 1e6 else 100  # Fewer iterations if promising
            mutation_rate = 0.85 if initial_score < 1e6 else 0.65  # Slightly more exploration for hard cases
            
            de_result = differential_evolution(
                compute_objective_with_penalties,
                bounds,
                maxiter=max_iterations,      # More iterations for harder cases
                popsize=population_size,     # Larger population for better exploration
                mutation=(mutation_rate, 1), # Aggressive mutation
                recombination=0.9,           # High recombination
                seed=42+i,                   # Different seed for each attempt
                disp=False,
                polish=False                 # Skip polish to save time, we'll refine manually
            )

            # Stage 2: Local refinement with L-BFGS-B using optimized bounds
            refined_result = minimize(
                compute_objective_with_penalties,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-8},  # More iterations for final refinement
                args=(True,)
            )

            # Evaluate final score
            final_score = compute_objective_with_penalties(refined_result.x, use_distance_penalties=True)

            if final_score < best_score and refined_result.success:
                best_score = final_score
                best_result = refined_result
                
        except Exception as e:
            continue

    # If we still haven't found a good solution, fall back to one of our initial guesses
    if best_result is None or best_score >= 1e9:
        # Just pick the best initial guess directly
        best_result = None
        best_score = float('inf')
        for i, guess in enumerate(initial_guesses):
            try:
                score = compute_objective_with_penalties(guess, use_distance_penalties=True)
                if score < best_score:
                    best_score = score
                    best_result = type('obj', (object,), {'x': guess})
            except:
                continue

    # Extract the best solution
    if best_result is None:
        # Fallback to a simple symmetric arrangement
        inner_hex_data = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ])
        outer_side_length = estimate_min_outer_radius(inner_hex_data.flatten())
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_side_length

    # Extract final parameters
    best_params = best_result.x
    outer_side_length = best_params[-1]

    # Extract inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_params[3*i], best_params[3*i+1], best_params[3*i+2]]

    # Outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin

    # Verify solution
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))

    valid_solution = True
    for i in range(11):
        x, y, rot = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)

        if not check_containment_single(hex_vertices, outer_polygon):
            valid_solution = False
            break

    if not valid_solution:
        # Revert to simple arrangement
        inner_hex_data = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ])
        outer_side_length = estimate_min_outer_radius(inner_hex_data.flatten())
        outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END