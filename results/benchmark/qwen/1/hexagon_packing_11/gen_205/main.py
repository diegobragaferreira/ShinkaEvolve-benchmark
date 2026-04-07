# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.validation import make_valid
import time

def create_regular_hexagon(center=(0, 0), side_length=1, rotation=0):
    """Create a regular hexagon as a shapely polygon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + side_length * np.cos(a),
               center[1] + side_length * np.sin(a)) for a in angles]
    return Polygon(points)

def get_hexagon_vertices(center, side_length, rotation):
    """Get all vertices of a hexagon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + side_length * np.cos(a),
             center[1] + side_length * np.sin(a)) for a in angles]

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    try:
        return outer_hex_poly.contains(hexagon_poly)
    except:
        try:
            valid_outer = make_valid(outer_hex_poly)
            valid_hex = make_valid(hexagon_poly)
            return valid_outer.contains(valid_hex)
        except:
            return False

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using spatial grid indexing for better performance"""
    # Simple bounding box intersection test first
    hex1_box = [min(p[0] for p in hex1_vertices), max(p[0] for p in hex1_vertices),
                min(p[1] for p in hex1_vertices), max(p[1] for p in hex1_vertices)]
    hex2_box = [min(p[0] for p in hex2_vertices), max(p[0] for p in hex2_vertices),
                min(p[1] for p in hex2_vertices), max(p[1] for p in hex2_vertices)]
    
    # Check if bounding boxes overlap
    if (hex1_box[1] < hex2_box[0] or hex2_box[1] < hex1_box[0] or
        hex1_box[3] < hex2_box[2] or hex2_box[3] < hex1_box[2]):
        return False
    
    # Fall back to full polygon intersection if bounding boxes overlap
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        try:
            valid_poly1 = make_valid(Polygon(hex1_vertices))
            valid_poly2 = make_valid(Polygon(hex2_vertices))
            return valid_poly1.intersects(valid_poly2)
        except:
            return True  # if we can't validate, assume they overlap

class HexGridIndex:
    """A spatial grid to quickly query nearby hexagons for collision detection"""
    def __init__(self, cell_size=4.0):
        self.cell_size = cell_size
        self.grid = {}
        
    def _get_cell_key(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def insert(self, hex_id, vertices):
        """Insert a hexagon into the grid"""
        # Get all cells this hexagon touches
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        
        min_cell_x = int(min_x // self.cell_size)
        max_cell_x = int(max_x // self.cell_size)
        min_cell_y = int(min_y // self.cell_size)
        max_cell_y = int(max_y // self.cell_size)
        
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                if (cx, cy) not in self.grid:
                    self.grid[(cx, cy)] = []
                self.grid[(cx, cy)].append(hex_id)
    
    def get_candidates(self, vertices):
        """Get candidate hexagon IDs that might collide with given hexagon"""
        candidates = set()
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        
        min_cell_x = int(min_x // self.cell_size)
        max_cell_x = int(max_x // self.cell_size)
        min_cell_y = int(min_y // self.cell_size)
        max_cell_y = int(max_y // self.cell_size)
        
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                if (cx, cy) in self.grid:
                    candidates.update(self.grid[(cx, cy)])
        return list(candidates)

def evaluate_solution(params, outer_hex_radius=None):
    """Evaluate a solution with optimized collision detection and adaptive penalties"""
    # Parse parameters
    inner_positions = params[:22].reshape(-1, 2)  # x,y pairs
    inner_rotations = params[22:33]  # 11 rotations
    if outer_hex_radius is None:
        outer_radius = params[33]  # outer hex radius
    else:
        outer_radius = outer_hex_radius

    # Create outer hexagon
    outer_hex = create_regular_hexagon((0, 0), outer_radius, 0)

    # Check if all inner hexagons fit inside outer hexagon
    num_inner_hexes = len(inner_positions)

    # Create all inner hexagon polygons
    inner_hexes = []
    for i in range(num_inner_hexes):
        pos = tuple(inner_positions[i])
        rot = inner_rotations[i]
        vertices = get_hexagon_vertices(pos, 1, rot)
        inner_hexes.append(vertices)

    # Use spatial indexing for collision detection
    grid_index = HexGridIndex(cell_size=2.0)
    
    # Insert all hexagons into the grid
    for i, vertices in enumerate(inner_hexes):
        grid_index.insert(i, vertices)
    
    # Check containment and overlaps using optimized grid-based approach
    total_penalty = 0
    containment_violations = 0
    
    for i, hex_vertices in enumerate(inner_hexes):
        hex_poly = Polygon(hex_vertices)
        if not check_containment(hex_poly, outer_hex):
            containment_violations += 1
            total_penalty += 1000 * (1 + containment_violations * 0.1)  # Increasing penalty

        # Use grid index to find nearby candidates
        candidates = grid_index.get_candidates(hex_vertices)
        for j in candidates:
            if i >= j:  # Avoid double-checking and self-checking
                continue
            if check_overlap_fast(hex_vertices, inner_hexes[j]):
                total_penalty += 10000 * (1 + j * 0.01)  # Penalize based on index

    # Calculate combined score (negative since we want to minimize)
    inv_radius = 1.0 / outer_radius if outer_radius > 0 else 0
    return -inv_radius + total_penalty

def generate_initial_guess():
    """Generate a good initial guess using symmetry-aware pattern"""
    # Generate initial configuration using hexagonal packing principles
    # Inspired by the densest packing arrangements of circles/hexagons
    
    positions = []
    rotations = []
    
    # Central hexagon
    positions.append([0, 0])
    rotations.append(0)
    
    # First ring - 6 hexagons at distance sqrt(3) ≈ 1.732
    # These positions maintain equal distance between centers
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles, skip last to avoid duplication
    distance = np.sqrt(3)  # Distance between centers for touching hexagons
    for angle in angles:
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)  # No rotation initially
        
    # Second ring - 4 hexagons at distance 2*sqrt(3) ≈ 3.464
    angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 7)[:-1]  # Offset by π/6
    distance = 2 * np.sqrt(3)
    placed_count = 0
    for i, angle in enumerate(angles):
        if placed_count >= 4:
            break
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)
        placed_count += 1
    
    # Ensure exactly 11 positions/rotations
    positions = positions[:11]
    rotations = rotations[:11]
    
    # Add slight randomness to avoid getting stuck in poor local minima
    for i in range(len(positions)):
        positions[i][0] += np.random.normal(0, 0.1)
        positions[i][1] += np.random.normal(0, 0.1)
        rotations[i] += np.random.uniform(-10, 10)
    
    # Flatten for parameter vector
    flat_params = []
    for pos in positions:
        flat_params.extend(pos)
    flat_params.extend(rotations)
    flat_params.append(4.0)  # Initial outer radius guess, smaller than previous versions
    
    return np.array(flat_params)

def adaptive_local_optimization(initial_params, bounds, max_iter=30):
    """Perform local optimization with adaptive tolerance"""
    def objective(params):
        return evaluate_solution(params)
    
    # Use L-BFGS-B with adaptive tolerances
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-5},
        jac=False
    )
    
    return result.x if result.success else initial_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid evolutionary approach with spatial indexing for enhanced performance.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate initial guess
    initial_guess = generate_initial_guess()
    
    # Define bounds for optimization
    bounds = []
    # Position bounds (-10, 10) for each of 11 hexagons
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10)])
    # Rotation bounds (0, 360) for each of 11 hexagons
    for _ in range(11):
        bounds.append((0, 360))
    # Outer radius bounds (1, 20)
    bounds.append((1, 20))
    
    # Phase 1: Coarse global search with differential evolution
    result = differential_evolution(
        evaluate_solution,
        bounds,
        maxiter=60,
        popsize=15,
        tol=1e-5,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    # Phase 2: Refine using adaptive local optimization
    refined_solution = adaptive_local_optimization(result.x, bounds, max_iter=40)
    
    # Phase 3: Final boundary refinement with lower tolerance
    final_result = minimize(
        lambda x: evaluate_solution(x),
        refined_solution,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6},
        jac=False
    )
    
    if final_result.success:
        best_solution = final_result.x
    else:
        best_solution = refined_solution

    # Extract results
    inner_positions = best_solution[:22].reshape(-1, 2)
    inner_rotations = best_solution[22:33]
    outer_radius = best_solution[33]

    # Create final data arrays
    inner_hex_data = np.column_stack([
        inner_positions[:, 0],
        inner_positions[:, 1],
        inner_rotations
    ])

    outer_hex_data = np.array([0, 0, 0])  # Centered at origin

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END