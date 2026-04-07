# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import math
from scipy.spatial.distance import cdist
from numba import jit
import time

# Set seed for reproducibility
np.random.seed(42)

# Constants
UNIT_HEX_SIDE = 1.0
PI = np.pi

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon efficiently using numba."""
    angle_offset = rotation_degrees * PI / 180.0
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_offset + i * PI / 3
        vertices[i, 0] = center_x + side_length * np.cos(angle)
        vertices[i, 1] = center_y + side_length * np.sin(angle)
    return vertices

def create_hexagon_polygon_fast(center_x, center_y, side_length, rotation_degrees):
    """Create Shapely polygon representation of a hexagon."""
    vertices = hexagon_vertices_jit(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)

@jit(nopython=True)
def check_overlap_fast_jit(hex1, hex2):
    """Fast overlap check using numba."""
    # Convert to numpy arrays for faster access
    v1 = hexagon_vertices_jit(hex1[0], hex1[1], 1, hex1[2])
    v2 = hexagon_vertices_jit(hex2[0], hex2[1], 1, hex2[2])
    
    # Create simple polygon representations for checking
    poly1 = Polygon(v1)
    poly2 = Polygon(v2)
    return poly1.intersects(poly2)

def check_containment_fast(hex_data, outer_hex_radius):
    """Fast containment check using Shapely."""
    outer_poly = create_hexagon_polygon_fast(0, 0, outer_hex_radius, 0)
    
    for hex_params in hex_data:
        inner_poly = create_hexagon_polygon_fast(hex_params[0], hex_params[1], 1, hex_params[2])
        if not outer_poly.contains(inner_poly):
            return False
    return True

@jit(nopython=True)
def compute_outer_hexagon_radius_jit(hex_data, tolerance=1e-6):
    """Compute minimum outer hexagon radius efficiently."""
    # Collect all vertices
    all_vertices = []
    for hex_params in hex_data:
        vertices = hexagon_vertices_jit(hex_params[0], hex_params[1], 1, hex_params[2])
        for i in range(6):
            all_vertices.append((vertices[i, 0], vertices[i, 1]))
    
    if not all_vertices:
        return tolerance
    
    # Calculate center and max distance
    n = len(all_vertices)
    sum_x = 0.0
    sum_y = 0.0
    for x, y in all_vertices:
        sum_x += x
        sum_y += y
    
    center_x = sum_x / n
    center_y = sum_y / n
    
    max_dist = 0.0
    for x, y in all_vertices:
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        dist = np.sqrt(dist_sq)
        if dist > max_dist:
            max_dist = dist
    
    max_dist += tolerance
    return max_dist

def evaluate_fitness_fast(hex_data):
    """Fast fitness evaluation."""
    try:
        radius = compute_outer_hexagon_radius_jit(hex_data)
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
            if check_overlap_fast_jit(hex_data[i], hex_data[j]):
                return False
    return True

def generate_improved_initial_config():
    """Generate a better initial configuration for 11 hexagons."""
    # Based on mathematical hexagonal packing principles with improved spatial distribution
    # Using concentric rings with optimized distances to maximize density
    
    # Central hexagon
    positions = [[0.0, 0.0, 0.0]]
    
    # First ring - 6 hexagons in a tight hexagonal pattern
    # Spacing chosen to allow maximal density while providing room for optimization
    ring_radius = 1.732  # ~sqrt(3) for good packing density
    for i in range(6):
        angle = i * 60  # 60 degree increments
        rad = math.radians(angle)
        x = ring_radius * math.cos(rad)
        y = ring_radius * math.sin(rad)
        positions.append([x, y, 0.0])
    
    # Second ring - strategic placement to fill gaps
    # Place 4 hexagons in a way that maximizes packing efficiency
    ring2_radius = 3.0  # Further out to fill gaps
    ring2_angles = [30, 90, 150, 210]  # Offset to avoid alignment issues
    
    for angle in ring2_angles:
        rad = math.radians(angle)
        x = ring2_radius * math.cos(rad)
        y = ring2_radius * math.sin(rad)
        positions.append([x, y, 0.0])
    
    # Add additional strategic positions to reach 11 total
    # These are placed to provide good coverage and flexibility for optimization
    extra_positions = [
        [-2.5, 0, 0],       # Left
        [2.5, 0, 0],        # Right
        [0, 2.5, 0],        # Top
        [0, -2.5, 0],       # Bottom
    ]
    
    # Add remaining positions
    for i in range(11 - len(positions)):
        positions.append(extra_positions[i])
    
    # Trim to exactly 11 positions and add jitter
    positions = positions[:11]
    
    # Add small jitter to break symmetry
    for i in range(len(positions)):
        positions[i][0] += np.random.normal(0, 0.02)
        positions[i][1] += np.random.normal(0, 0.02)
    
    return np.array(positions)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    start_time = time.time()
    
    # Generate better initial configuration
    initial_positions = generate_improved_initial_config()
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

        # Evaluate fitness - we want to maximize 1/R (minimize R)
        try:
            score = evaluate_fitness_fast(positions_angles)
            return -score  # Negative because we minimize for maximization problem
        except:
            return 1e10  # Bad fitness if evaluation fails

    def evaluate_layout(inner_positions_angles):
        """Evaluate the layout quality"""
        try:
            # Compute outer hexagon radius
            outer_radius = compute_outer_hexagon_radius_jit(inner_positions_angles)
            
            # Validate constraints
            valid = check_containment_fast(inner_positions_angles, outer_radius)
            valid &= validate_individual_fast(inner_positions_angles)
            
            # Return negative because we want to maximize 1/R (minimize R)
            inv_radius = 1.0 / outer_radius if valid else 0.0
            
            return -inv_radius, outer_radius
        except:
            return -1e10, 100.0

    # Two-stage optimization approach
    best_score = float('inf')
    best_inner_data = inner_hex_data.copy()
    best_outer_side_length = 10.0

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
            disp=False
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
    if best_score < float('inf'):
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
                options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-8},
                callback=None
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
    try:
        outer_radius = compute_outer_hexagon_radius_jit(best_inner_data, 1e-6)
        
        # Final validation check
        if not check_containment_fast(best_inner_data, outer_radius) or not validate_individual_fast(best_inner_data):
            print("Final validation failed, using fallback...")
            # Fall back to initial configuration
            best_inner_data = initial_positions.copy()
            outer_radius = compute_outer_hexagon_radius_jit(best_inner_data, 1e-6)
    except:
        # If validation fails, use initial positions
        best_inner_data = initial_positions.copy()
        outer_radius = compute_outer_hexagon_radius_jit(best_inner_data, 1e-6)

    # Ensure we're returning the correct data format
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")

    # Return results
    return best_inner_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END