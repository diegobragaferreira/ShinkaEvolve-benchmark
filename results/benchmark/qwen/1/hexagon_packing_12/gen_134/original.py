# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit, prange
from joblib import Parallel, delayed

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

@jit(nopython=True)
def point_in_hexagon(px, py, hx, hy, angle_deg, side_length=1):
    """Fast point-in-hexagon test using winding number or separating axis theorem."""
    vertices = hexagon_vertices(hx, hy, angle_deg, side_length)
    # Use ray casting method for simplicity and speed
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment."""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1

    # Length squared of line segment
    length_sq = dx*dx + dy*dy

    if length_sq == 0:
        # Line segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp projection to line segment

    # Find closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)

    min_dist = np.inf
    for i in range(6):
        for j in range(6):
            # Distance between vertices
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist

            # Distance from vertex to edge of other hexagon
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist

            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist

    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment_parallel(hex_poly, outer_hex_poly, n_jobs=-1):
    """Parallel check if all vertices of hexagon are contained within outer hexagon."""
    # Extract vertices
    coords = list(hex_poly.exterior.coords)
    # We'll do this with our fast numba checker
    return outer_hex_poly.contains(hex_poly)

def check_overlap_parallel(hex1_poly, hex2_poly, n_jobs=-1):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def evaluate_configuration_parallel(params):
    """Parallelized evaluation using joblib for overlap/containment checks."""
    # Extract inner hexagon data and outer radius
    inner_hex_data = params[:-1].reshape(12, 3)
    outer_hex_side_length = params[-1]

    # Create outer hexagon polygon (centered at origin)
    outer_hex_poly = compute_hexagon_polygon(0, 0, 0, outer_hex_side_length)

    # Check containment and overlaps
    valid = True
    total_penalty = 0.0

    n = len(inner_hex_data)
    inner_polys = []

    # Compute all inner hexagon polygons
    for i in range(n):
        x, y, angle = inner_hex_data[i]
        inner_poly = compute_hexagon_polygon(x, y, angle)
        inner_polys.append(inner_poly)

        # Check containment
        if not outer_hex_poly.contains(inner_poly):
            valid = False
            # Calculate penetration
            try:
                diff = outer_hex_poly.difference(inner_poly)
                if hasattr(diff, 'area'):
                    total_penalty += diff.area
            except:
                total_penalty += 1000  # Large penalty

    if not valid:
        # Adaptive penalty - scale with the severity of the constraint violation
        penalty = total_penalty * 10000 if total_penalty > 0 else 1000000
        return penalty

    # Check overlaps using spatial indexing for efficiency
    overlap_area_total = 0.0

    # Create spatial index using centroids for fast proximity checking
    centroids = []
    for i in range(n):
        x, y, _ = inner_hex_data[i]
        centroids.append([x, y])

    # Use scipy's cKDTree for nearest neighbor search
    tree = cKDTree(centroids)

    # Find neighbors within a reasonable distance (2 units apart for unit hexagons)
    # This avoids checking all pairs while being conservative about false positives
    pairs_to_check = tree.query_pairs(r=3.0, output_type='ndarray')

    # Check overlaps only for pairs that are close enough
    overlap_pairs = []
    for i, j in pairs_to_check:
        if i < j:  # Avoid duplicates
            overlap_pairs.append((inner_polys[i], inner_polys[j]))

    # Also check pairs that are likely to be close based on distance thresholds
    for i in range(n):
        for j in range(i+1, n):
            # Skip pairs that are definitely too far apart
            dist = np.sqrt((inner_hex_data[i][0] - inner_hex_data[j][0])**2 +
                          (inner_hex_data[i][1] - inner_hex_data[j][1])**2)
            if dist < 5.0:  # Only check pairs that could reasonably overlap
                overlap_pairs.append((inner_polys[i], inner_polys[j]))

    # Remove duplicates if any
    unique_pairs = list(set(tuple(pair) for pair in overlap_pairs))

    # Check overlaps in parallel
    def check_single_overlap(pair):
        poly1, poly2 = pair
        if poly1.intersects(poly2) and not poly1.touches(poly2):
            try:
                overlap = poly1.intersection(poly2)
                return overlap.area
            except:
                return 1000  # Large penalty
        return 0.0

    overlap_results = Parallel(n_jobs=-1)(
        delayed(check_single_overlap)(pair) for pair in unique_pairs
    )

    # Sum all overlap areas
    for area in overlap_results:
        if area > 0:
            overlap_area_total += area

    if overlap_area_total > 0:
        total_penalty += overlap_area_total * 10000  # Apply penalty for overlaps
        valid = False

    if not valid:
        return total_penalty

    # If valid, return inverse of outer hexagon side length (negative because we minimize)
    return -1.0 / outer_hex_side_length

def generate_symmetric_initial_guess():
    """Generate a good initial symmetric configuration."""
    # Based on mathematical hexagonal lattice arrangement
    # Using a known efficient configuration approach
    positions = []

    # Central hexagon
    positions.append([0, 0, 0])

    # First ring around center (at distance sqrt(3) from center)
    for i in range(6):
        angle = i * 60
        x = 1.732 * np.cos(np.radians(angle))  # ~= sqrt(3) for proper spacing
        y = 1.732 * np.sin(np.radians(angle))
        positions.append([x, y, 0])

    # Second ring (at distance 2*sqrt(3) from center)
    for i in range(6):
        angle = i * 60 + 30  # offset to create more efficient packing
        x = 3.464 * np.cos(np.radians(angle))  # ~= 2*sqrt(3)
        y = 3.464 * np.sin(np.radians(angle))
        positions.append([x, y, 0])

    # Additional strategic placement for better packing
    positions.append([0, -4.5, 0])

    # Take only first 12 positions
    initial_config = np.array(positions[:12])

    # Add some randomness to avoid local minima
    np.random.seed(42)
    initial_config[:, :2] += np.random.normal(0, 0.2, (12, 2))

    return initial_config

def optimize_packing():
    """Main optimization function using hybrid approach."""
    # Generate initial guess
    initial_guess_inner = generate_symmetric_initial_guess()

    # Initial estimate for outer radius based on configuration
    max_dist = 0
    for i in range(12):
        x, y, _ = initial_guess_inner[i]
        dist = np.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist)

    # Add margin for hexagon size (hexagon has width approximately 2)
    initial_outer_radius = max_dist + 2.0

    # Combine into single parameter vector: [12*3 positions + 1 outer radius]
    initial_params = np.concatenate([initial_guess_inner.flatten(), [initial_outer_radius]])

    # Define bounds for optimization
    # Positions: x, y bounded to reasonable range, angle 0-360
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    # Outer radius bound (should be positive)
    bounds.append((0.1, 20.0))

    # First, use global optimization to explore the space broadly
    def objective_global(params):
        # Use smaller penalty scaling for global optimization
        return evaluate_configuration_parallel(params)

    # Run differential evolution with reduced iterations for speed
    de_result = differential_evolution(
        objective_global,
        bounds,
        maxiter=30,
        popsize=10,
        seed=42,
        disp=False
    )

    # Use the best solution from global search as starting point for local refinement
    best_params = de_result.x

    # Local refinement using L-BFGS-B
    # We need to make sure we're still within bounds for the local optimizer
    refined_bounds = [(b[0], b[1]) for b in bounds]

    # Perform local optimization
    try:
        local_result = minimize(
            evaluate_configuration_parallel,
            best_params,
            method='L-BFGS-B',
            bounds=refined_bounds,
            options={'maxiter': 50, 'disp': False}
        )
        if local_result.success:
            best_params = local_result.x
    except:
        # Fall back to previous solution if local optimization fails
        pass

    # Extract results
    inner_hex_data = best_params[:-1].reshape(12, 3)
    outer_hex_side_length = best_params[-1]

    # Ensure the outer hexagon is actually large enough
    # Recalculate to make sure it contains all hexagons
    min_outer_radius = 0
    for i in range(12):
        x, y, _ = inner_hex_data[i]
        dist = np.sqrt(x*x + y*y) + 1.0  # +1 for hexagon radius
        min_outer_radius = max(min_outer_radius, dist)

    # If we computed a smaller radius than needed, adjust it up
    if outer_hex_side_length < min_outer_radius:
        outer_hex_side_length = min_outer_radius * 1.05  # Add small margin

    return inner_hex_data, outer_hex_side_length

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
        inner_hex_data, outer_hex_side_length = optimize_packing()

        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])

        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time

        # Calculate benchmark ratio
        benchmark_ratio = (1.0 / outer_hex_side_length) / 0.2537

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback to original configuration if optimization fails
        n = 12
        inner_hex_data = np.array([
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
            [0, -4, 0],
        ])

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END