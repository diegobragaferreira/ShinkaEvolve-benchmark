# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit
import warnings
import random
from collections import defaultdict

warnings.filterwarnings('ignore')

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

def get_bounding_box(hex_poly):
    """Get bounding box of hexagon for spatial indexing"""
    bounds = hex_poly.bounds
    return (bounds[0], bounds[1], bounds[2], bounds[3])

def build_spatial_grid(hexagons, grid_size=5.0):
    """Build a simple spatial grid for fast collision detection"""
    grid = defaultdict(list)
    for i, hex_poly in enumerate(hexagons):
        bbox = get_bounding_box(hex_poly)
        min_x, min_y, max_x, max_y = bbox
        for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
            for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
                grid[(x,y)].append(i)
    return grid

def fast_collision_check(hexagons, grid, i, j):
    """Fast collision check using spatial grid"""
    if i == j:
        return False

    # Check if hexagons are in same or adjacent grid cells
    bbox_i = get_bounding_box(hexagons[i])
    bbox_j = get_bounding_box(hexagons[j])

    # Simple overlap test first
    if not hexagons[i].intersects(hexagons[j]):
        return False

    # More precise check
    return hexagons[i].intersects(hexagons[j])

def evaluate_solution(solution, use_spatial_index=True):
    """Evaluate a solution and return negative of objective (since we minimize)"""
    # Reshape solution into positions and angles
    positions = solution[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates each
    angles = solution[22:]  # 11 angles

    # Create inner hexagons
    inner_hexagons = []
    for i in range(11):
        pos = positions[i]
        angle = angles[i]
        hex_poly = get_hexagon_polygon(pos[0], pos[1], angle)
        inner_hexagons.append(hex_poly)

    # Check containment
    outer_radius = calculate_outer_hexagon_radius(positions, angles)
    # Outer hexagon with center at origin and calculated radius
    outer_hexagon = get_hexagon_polygon(0, 0, 0, outer_radius)

    # Check containment for all inner hexagons
    for hex_poly in inner_hexagons:
        if not check_containment(hex_poly, outer_hexagon):
            return 1e10  # Penalty for non-containment

    # Build spatial grid for collision detection if requested
    if use_spatial_index:
        grid = build_spatial_grid(inner_hexagons)

    # Check for overlaps
    for i in range(11):
        for j in range(i+1, 11):
            # Use fast collision check with spatial indexing
            if use_spatial_index:
                if not fast_collision_check(inner_hexagons, grid, i, j):
                    continue
            else:
                if not inner_hexagons[i].intersects(inner_hexagons[j]):
                    continue

            # If we reach here, there's a collision
            return 1e10  # Penalty for overlap

    # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
    return -1.0 / outer_radius

def generate_initial_population(num_starts=5):
    """Generate multiple initial configurations"""
    initial_populations = []

    # Base pattern: hexagonal arrangement
    base_positions = []
    base_angles = []

    # Center hexagon
    base_positions.append([0.0, 0.0])
    base_angles.append(0.0)

    # Surrounding hexagons in ring
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        base_positions.append([x, y])
        base_angles.append(0.0)

    # Additional positions for remaining hexagons
    additional_positions = [
        (-3.0, 1.0), (3.0, 1.0),
        (-3.0, -1.0), (3.0, -1.0),
        (0.0, 3.0), (0.0, -3.0),
        (1.5, 2.6), (-1.5, -2.6),
        (-1.5, 2.6), (1.5, -2.6)
    ]

    for pos in additional_positions:
        if len(base_positions) < 11:
            base_positions.append(list(pos))
            base_angles.append(0.0)

    # Ensure we have exactly 11 positions
    while len(base_positions) < 11:
        base_positions.append([0.0, 0.0])
        base_angles.append(0.0)

    # Generate different variations
    for start in range(num_starts):
        # Create a slightly different initial configuration for each start
        initial_positions = [pos[:] for pos in base_positions]  # Copy
        initial_angles = [ang for ang in base_angles]  # Copy

        # Add small random perturbations
        for i in range(len(initial_positions)):
            if i > 0:  # Don't perturb center hexagon significantly
                initial_positions[i][0] += random.uniform(-0.2, 0.2)
                initial_positions[i][1] += random.uniform(-0.2, 0.2)
                initial_angles[i] += random.uniform(-5, 5)

        # Flatten initial solution
        initial_solution = []
        for pos in initial_positions[:11]:
            initial_solution.extend(pos)
        initial_solution.extend(initial_angles[:11])
        initial_solution = np.array(initial_solution)

        initial_populations.append(initial_solution)

    return initial_populations

def optimize_hexagon_packing():
    """Main optimization function with multi-start approach"""
    # Generate multiple initial populations
    initial_populations = generate_initial_population(5)

    best_result = None
    best_score = float('inf')

    # Run optimization from multiple starting points
    for i, initial_solution in enumerate(initial_populations):
        try:
            # Set bounds for optimization
            bounds = []
            # Position bounds
            for _ in range(22):
                bounds.append((-10.0, 10.0))  # X and Y coordinates
            # Angle bounds
            for _ in range(11):
                bounds.append((0.0, 360.0))   # Rotation angles

            # Use adaptive DE parameters
            maxiter = 100
            popsize = 15

            # Run differential evolution
            result = differential_evolution(
                lambda sol: evaluate_solution(sol, use_spatial_index=(i < 3)),  # Use spatial index for first 3 starts
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=42+i,  # Different seed for each start
                disp=False,
                tol=1e-6,
                adaptive=True  # Enable adaptive parameters
            )

            # Evaluate final result
            final_score = evaluate_solution(result.x, use_spatial_index=False)

            if final_score < best_score:
                best_score = final_score
                best_result = result

        except Exception as e:
            print(f"Start {i} failed: {e}")
            continue

    if best_result is None:
        # Fallback to simple solution
        raise RuntimeError("All optimization attempts failed")

    # Extract final solution
    final_positions = best_result.x[:22].reshape(-1, 2)
    final_angles = best_result.x[22:]

    # Refine the solution with adaptive local search
    refined_solution = adaptive_local_search(final_positions, final_angles)

    return refined_solution

def adaptive_local_search(positions, angles):
    """Apply adaptive local search with systematic perturbations"""
    # Simple gradient descent-like refinement with better perturbations
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))

    # Use adaptive step sizes that decrease over iterations
    step_sizes = [0.1, 0.05, 0.02, 0.01]
    max_iterations = 200
    patience = 15
    patience_counter = 0

    for iteration in range(max_iterations):
        improved = False

        # Try different step sizes in order
        for step_size in step_sizes:
            # Try small perturbations to each position and angle
            for i in range(11):
                # Perturb position in both directions
                for dim in range(2):
                    old_val = best_positions[i][dim]
                    for delta in [-step_size, step_size]:
                        best_positions[i][dim] = old_val + delta
                        new_score = evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_positions[i][dim] = old_val

                # Perturb angle in both directions
                old_angle = best_angles[i]
                for delta in [-5.0, 5.0]:  # Larger angle steps
                    best_angles[i] = old_angle + delta
                    new_score = evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_angles[i] = old_angle

            # If we made an improvement with this step size, break to next iteration
            if improved:
                break

        # Check for improvement
        if not improved:
            patience_counter += 1
            if patience_counter >= patience:
                break
        else:
            patience_counter = 0

    return best_positions, best_angles

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Run optimization
        final_positions, final_angles = optimize_hexagon_packing()

        # Create inner hex data
        inner_hex_data = np.column_stack([final_positions, final_angles])

        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])

        # Calculate outer hex side length
        outer_radius = calculate_outer_hexagon_radius(final_positions, final_angles)
        # Convert to side length for regular hexagon
        outer_hex_side_length = outer_radius / (np.sqrt(3) / 2)

        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial solution
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
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END