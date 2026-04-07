# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from numba import jit, prange
import math

# Set seed for reproducibility
np.random.seed(42)

@jit(nopython=True)
def hexagon_vertices_fast(center_x, center_y, side_length, rotation_rad):
    """Fast vertex calculation using numba"""
    vertices_x = np.empty(6)
    vertices_y = np.empty(6)
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        vertices_x[i] = center_x + side_length * math.cos(angle)
        vertices_y[i] = center_y + side_length * math.sin(angle)
    return vertices_x, vertices_y

@jit(nopython=True)
def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Fast distance calculation from point to line segment"""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx*dx + dy*dy
    if len_sq == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0, min(1, t))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon."""
    rotation_rad = math.radians(rotation_degrees)
    vertices_x, vertices_y = hexagon_vertices_fast(center_x, center_y, side_length, rotation_rad)
    return np.column_stack([vertices_x, vertices_y])

def create_hexagon_polygon(center_x, center_y, side_length, rotation_degrees):
    """Create Shapely polygon representation of a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)

def check_overlap_fast(hex1, hex2):
    """Fast overlap check using Shapely."""
    poly1 = create_hexagon_polygon(hex1[0], hex1[1], 1, hex1[2])
    poly2 = create_hexagon_polygon(hex2[0], hex2[1], 1, hex2[2])
    return poly1.intersects(poly2)

def check_containment_fast(hex_data, outer_hex_radius):
    """Fast containment check using Shapely."""
    outer_poly = create_hexagon_polygon(0, 0, outer_hex_radius, 0)
    
    for hex_params in hex_data:
        inner_poly = create_hexagon_polygon(hex_params[0], hex_params[1], 1, hex_params[2])
        if not outer_poly.contains(inner_poly):
            return False
    return True

def compute_outer_hexagon_radius_fast(hex_data, tolerance=1e-6):
    """Compute minimum outer hexagon radius using optimized approach."""
    # Get all vertices efficiently
    all_vertices_x = []
    all_vertices_y = []
    
    for hex_params in hex_data:
        vertices = hexagon_vertices(hex_params[0], hex_params[1], 1, hex_params[2])
        all_vertices_x.extend(vertices[:, 0])
        all_vertices_y.extend(vertices[:, 1])
    
    if not all_vertices_x:
        return tolerance
    
    # Calculate center point
    center_x = sum(all_vertices_x) / len(all_vertices_x)
    center_y = sum(all_vertices_y) / len(all_vertices_y)
    
    # Calculate maximum distance
    max_dist = 0
    for i in range(len(all_vertices_x)):
        dist = math.sqrt((all_vertices_x[i] - center_x)**2 + (all_vertices_y[i] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add small buffer
    max_dist += tolerance
    
    # Binary search for exact radius
    low = max_dist
    high = max_dist * 2.0
    
    while high - low > tolerance:
        mid = (low + high) / 2.0
        if check_containment_fast(hex_data, mid):
            high = mid
        else:
            low = mid
            
    return (low + high) / 2.0

def evaluate_fitness_fast(hex_data):
    """Fast fitness evaluation."""
    try:
        radius = compute_outer_hexagon_radius_fast(hex_data)
        # Inverse of radius (higher is better)
        return 1.0 / radius
    except:
        # If there's an error, return a very bad fitness
        return 0.0

def validate_individual_fast(hex_data):
    """Fast validation of individual (no overlaps)."""
    n = len(hex_data)
    # Early exit if too few elements
    if n <= 1:
        return True
    
    # Check for overlaps using pairwise comparison
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap_fast(hex_data[i], hex_data[j]):
                return False
    return True

def generate_better_initial_config():
    """Generate a better initial configuration for 11 hexagons using optimized hexagonal packing."""
    # Create a structured hexagonal lattice with strategic positioning
    # Following mathematical principles of optimal hexagonal packing
    
    # Central hexagon
    config = [[0.0, 0.0, 0.0]]
    
    # First ring - 6 hexagons arranged in perfect hexagonal pattern
    # Distance of 2 units between touching hexagon centers
    for i in range(6):
        angle = i * 60  # 60 degree increments
        rad = math.radians(angle)
        x = 2.0 * math.cos(rad)
        y = 2.0 * math.sin(rad)
        config.append([x, y, 0.0])
    
    # Second ring - strategic placement to maximize packing efficiency
    # Place at approximately sqrt(3) * 2 ≈ 3.464 units radius
    ring_radius = 2.0 * math.sqrt(3)  # About 3.464 units
    # Use just 4 positions to get 11 total (1 + 6 + 4 = 11)
    positions = [
        (0.0, ring_radius, 0.0),        # Top
        (ring_radius * 0.5, -ring_radius * 0.5, 0.0),  # Bottom-right
        (-ring_radius * 0.5, -ring_radius * 0.5, 0.0), # Bottom-left
        (ring_radius * 0.5, ring_radius * 0.5, 0.0),   # Top-right
    ]
    
    # Add strategic positions for the second ring
    for x, y, angle in positions[:4]:
        config.append([x, y, angle])
    
    # Trim to exactly 11 positions
    config = config[:11]
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    for i in range(len(config)):
        config[i][0] += np.random.normal(0, 0.02)
        config[i][1] += np.random.normal(0, 0.02)
    
    return np.array(config)

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
        # Generate better initial configuration
        initial_positions = generate_better_initial_config()
        inner_hex_data = initial_positions.copy()
        
        # Optimization bounds for each parameter (x, y, angle)
        bounds = []
        for i in range(11):
            # Reasonable bounds to keep solutions in a practical region
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # angle in degrees

        # Define objective function for optimization
        def objective(params):
            # Reshape parameters into positions and angles
            positions_angles = []
            for i in range(11):
                x = params[i*3]
                y = params[i*3 + 1]
                angle = params[i*3 + 2]
                positions_angles.append([x, y, angle])

            score, side_length = evaluate_layout(positions_angles)
            return score  # Negative since we minimize -score = maximize score

        def evaluate_layout(inner_positions_angles):
            """Evaluate the layout quality"""
            # Convert to hexagon polygons efficiently
            inner_hexagons = []
            for pos_angle in inner_positions_angles:
                x, y, angle = pos_angle
                hex_poly = create_hexagon_polygon(x, y, 1, angle)
                inner_hexagons.append(hex_poly)

            # Compute outer hexagon radius
            outer_radius = compute_outer_hexagon_radius_fast(inner_positions_angles)
            outer_hexagon = create_hexagon_polygon(0, 0, outer_radius, 0)

            # Validate constraints
            valid = check_containment_fast(inner_positions_angles, outer_radius)
            valid &= validate_individual_fast(inner_positions_angles)

            # Return negative because we want to maximize 1/R (minimize R)
            outer_side_length = outer_radius
            inv_radius = 1.0 / outer_side_length if valid else 0.0

            return -inv_radius, outer_side_length

        # Two-stage optimization approach
        best_score = float('inf')
        best_inner_data = inner_hex_data.copy()
        best_outer_side_length = 10.0
        time_limit = 170  # seconds

        # Stage 1: Coarse global optimization with differential evolution  
        try:
            print("Starting coarse global optimization...")
            result = differential_evolution(
                objective,
                bounds,
                maxiter=60,
                popsize=15,
                seed=42,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False,
                timeout=time_limit
            )

            # Extract best solution from coarse optimization
            best_params = result.x
            final_positions_angles = []
            for i in range(11):
                x = best_params[i*3]
                y = best_params[i*3 + 1]
                angle = best_params[i*3 + 2]
                final_positions_angles.append([x, y, angle])

            # Evaluate final result
            final_score, final_side_length = evaluate_layout(final_positions_angles)

            if final_score < best_score and final_side_length > 0:
                best_score = final_score
                best_inner_data = np.array(final_positions_angles)
                best_outer_side_length = final_side_length

        except Exception as e:
            print(f"Coarse optimization failed: {e}")
            pass

        # Stage 2: Fine-grained local optimization with L-BFGS-B
        if best_score < float('inf') and (time.time() - start_time) < time_limit:
            print("Starting fine-grained local optimization...")

            # Convert to flat array for scipy optimization
            initial_flat = []
            for pos_angle in best_inner_data:
                initial_flat.extend(pos_angle)

            # Local refinement with L-BFGS-B
            try:
                result_local = minimize(
                    objective,
                    initial_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-8},
                    callback=None,
                    timeout=time_limit - (time.time() - start_time)
                )

                # Extract refined solution
                refined_params = result_local.x
                refined_positions_angles = []
                for i in range(11):
                    x = refined_params[i*3]
                    y = refined_params[i*3 + 1]
                    angle = refined_params[i*3 + 2]
                    refined_positions_angles.append([x, y, angle])

                # Evaluate refined result
                refined_score, refined_side_length = evaluate_layout(refined_positions_angles)

                if refined_score < best_score and refined_side_length > 0:
                    best_score = refined_score
                    best_inner_data = np.array(refined_positions_angles)
                    best_outer_side_length = refined_side_length

            except Exception as e:
                print(f"Fine optimization failed: {e}")
                pass

        # Final validation and refinement
        print("Performing final validation...")

        # Always validate the result
        if (time.time() - start_time) < time_limit:
            # Recompute outer hexagon size carefully
            outer_radius = compute_outer_hexagon_radius_fast(best_inner_data, 1e-6)
            outer_hexagon = create_hexagon_polygon(0, 0, outer_radius, 0)

            # Final validation check
            if not check_containment_fast(best_inner_data, outer_radius) or not validate_individual_fast(best_inner_data):
                print("Final validation failed, using fallback...")
                # Fall back to initial configuration
                best_inner_data = initial_positions.copy()
                outer_radius = compute_outer_hexagon_radius_fast(best_inner_data, 1e-6)
            else:
                outer_radius = outer_radius
        else:
            # Time limit reached - use current best
            outer_radius = compute_outer_hexagon_radius_fast(best_inner_data, 1e-6)

        # Ensure we're returning the correct data format
        outer_hex_data = np.array([0, 0, 0])  # centered at origin

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to baseline approach
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
        ])
        outer_radius = 8.0
        outer_hex_data = np.array([0, 0, 0])

    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END