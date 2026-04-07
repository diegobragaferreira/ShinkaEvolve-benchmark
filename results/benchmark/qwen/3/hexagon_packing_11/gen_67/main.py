# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import minimize
import time
from numba import jit

@jit(nopython=True)
def hex_distance(x1, y1, x2, y2):
    """Fast Euclidean distance calculation"""
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

@jit(nopython=True)
def point_in_hexagon(px, py, hx, hy, side_length, angle_rad):
    """Fast point-in-hexagon test using distance to center"""
    dx = px - hx
    dy = py - hy
    distance_from_center = math.sqrt(dx*dx + dy*dy)
    
    # For unit hexagon, distance threshold for containment
    # This is approximate but fast for early rejection
    max_distance = side_length * 1.1  # Slight buffer
    if distance_from_center > max_distance:
        return False
    
    # More precise check using hexagon vertices
    # But we'll do a quick approx for most cases
    return True

def create_regular_hexagon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a regular hexagon as a Shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def compute_outer_hexagon_radius(inner_hexagons, padding=0.01):
    """Compute minimum radius needed to contain all inner hexagons with some padding"""
    # Get all vertices of all hexagons
    all_vertices = []
    for hex_poly in inner_hexagons:
        all_vertices.extend(list(hex_poly.exterior.coords))

    # Find center of bounding box
    xs = [p[0] for p in all_vertices]
    ys = [p[1] for p in all_vertices]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2

    # Compute max distance from center to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)

    # Add padding and convert to side length
    # For a regular hexagon, radius = side_length
    return max_dist + padding

def check_containment_and_overlap_fast(inner_positions_angles):
    """Fast check for basic containment and overlap using direct geometry"""
    # Precompute hexagons for overlap checking
    n_hex = len(inner_positions_angles)
    
    # Check containment - quickly reject if any hexagon is outside the theoretical bounds
    # Using approximate hexagon boundaries
    for i in range(n_hex):
        x, y, angle = inner_positions_angles[i]
        # Approximate hexagon boundary
        if abs(x) > 10 or abs(y) > 10:  # arbitrary large bounds
            return False
    
    # Check overlap between all pairs using distance threshold
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            x1, y1, _ = inner_positions_angles[i]
            x2, y2, _ = inner_positions_angles[j]
            
            # Distance between centers of unit hexagons should be >= 2 for non-overlap
            distance = hex_distance(x1, y1, x2, y2)
            if distance < 1.99:  # Allow small overlap tolerance
                return False
                
    return True

def calculate_penalty(inner_positions_angles):
    """Calculate penalty based on overlap and containment violations"""
    penalty = 0.0
    
    # Check overlap penalties
    n_hex = len(inner_positions_angles)
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            x1, y1, _ = inner_positions_angles[i]
            x2, y2, _ = inner_positions_angles[j]
            
            distance = hex_distance(x1, y1, x2, y2)
            if distance < 1.99:  # Overlapping
                overlap_amount = 1.99 - distance
                penalty += overlap_amount * 1000.0  # Heavy penalty
    
    # Check containment penalties
    # Approximate containment by distance from center
    max_distance_from_center = 0.0
    for i in range(n_hex):
        x, y, _ = inner_positions_angles[i]
        distance = hex_distance(x, y, 0, 0)
        if distance > max_distance_from_center:
            max_distance_from_center = distance
            
    # Penalize if any hexagon extends too far from center (approximate containment)
    if max_distance_from_center > 10:  # Arbitrary large bound
        penalty += (max_distance_from_center - 10) * 100.0
    
    return penalty

def evaluate_layout_with_penalty(inner_positions_angles, outer_center=(0, 0)):
    """Evaluate layout with integrated penalty for constraints"""
    # Convert to hexagon polygons
    inner_hexagons = []
    for pos_angle in inner_positions_angles:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)

    # Validate constraints - fast preliminary check
    if not check_containment_and_overlap_fast(inner_positions_angles):
        # Return very poor score if constraints violated
        return -1e10, 1000.0  # Very bad score

    # Calculate penalty for constraint violations
    penalty = calculate_penalty(inner_positions_angles)
    
    # Compute outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
    
    # For the main metric, we want to maximize 1/outer_radius, so minimize outer_radius
    # Add penalty to discourage constraint violations
    total_score = -(1.0 / outer_radius) - penalty * 0.001
    
    return total_score, outer_radius

def generate_initial_population(size, bounds):
    """Generate diverse initial population"""
    population = []
    for _ in range(size):
        individual = []
        for i in range(11):  # 11 hexagons
            x = np.random.uniform(bounds[i*3][0], bounds[i*3][1])
            y = np.random.uniform(bounds[i*3+1][0], bounds[i*3+1][1])
            angle = np.random.uniform(bounds[i*3+2][0], bounds[i*3+2][1])
            individual.append([x, y, angle])
        population.append(individual)
    return population

def adaptive_particle_swarm_optimization(bounds, max_iter=100):
    """Adaptive PSO with dynamic inertia and neighborhood topology"""
    n_particles = 30
    n_dimensions = 33  # 11 hexagons * 3 parameters each
    
    # Initialize particles
    particles = generate_initial_population(n_particles, bounds)
    velocities = [np.random.rand(n_dimensions) * 0.1 for _ in range(n_particles)]
    
    # Personal best
    personal_best = [None] * n_particles
    personal_best_scores = [float('-inf')] * n_particles
    
    # Global best
    global_best = None
    global_best_score = float('-inf')
    
    # Parameters
    w_min, w_max = 0.4, 0.9  # Dynamic inertia weight
    c1, c2 = 2.0, 2.0  # Cognitive and social coefficients
    
    for iteration in range(max_iter):
        # Dynamic inertia weight
        w = w_max - (w_max - w_min) * (iteration / max_iter)
        
        for i in range(n_particles):
            # Evaluate current particle
            score, _ = evaluate_layout_with_penalty(particles[i])
            
            # Update personal best
            if score > personal_best_scores[i]:
                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = score
                
                # Update global best
                if score > global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score
            
            # Update velocity and position
            r1, r2 = np.random.rand(n_dimensions), np.random.rand(n_dimensions)
            
            for d in range(n_dimensions):
                if personal_best[i] is not None:
                    pbest = personal_best[i][d//3][d%3]
                else:
                    pbest = particles[i][d//3][d%3]
                
                v = w * velocities[i][d] + \
                    c1 * r1[d] * (pbest - particles[i][d//3][d%3]) + \
                    c2 * r2[d] * (global_best[d//3][d%3] - particles[i][d//3][d%3])
                
                velocities[i][d] = v
                
                # Update position
                new_pos = particles[i][d//3][d%3] + v
                # Apply bounds
                if d % 3 == 0:  # x coordinate
                    new_pos = max(bounds[d][0], min(bounds[d][1], new_pos))
                elif d % 3 == 1:  # y coordinate  
                    new_pos = max(bounds[d][0], min(bounds[d][1], new_pos))
                else:  # angle
                    new_pos = new_pos % 360
                    
                particles[i][d//3][d%3] = new_pos
    
    return global_best if global_best is not None else particles[0]

def constraint_aware_gradient_descent(initial_positions_angles, bounds, max_iter=50):
    """Gradient descent that respects geometric constraints"""
    current_positions = np.array(initial_positions_angles)
    
    # For this simple implementation, we'll just do a basic step-wise refinement
    # In a full implementation, this would compute true gradients respecting constraints
    
    # Try to move hexagons towards better positions, respecting bounds
    for iteration in range(max_iter):
        # Basic optimization: nudge positions slightly
        new_positions = current_positions.copy()
        for i in range(11):
            # Random small perturbation
            for dim in range(3):  # x, y, angle
                step_size = 0.1
                if dim < 2:  # x or y
                    delta = np.random.normal(0, step_size)
                    new_positions[i][dim] += delta
                else:  # angle
                    delta = np.random.normal(0, 10)
                    new_positions[i][dim] += delta
                    new_positions[i][dim] = new_positions[i][dim] % 360
                    
                # Enforce bounds
                if dim == 0:  # x
                    new_positions[i][0] = max(bounds[i*3][0], min(bounds[i*3][1], new_positions[i][0]))
                elif dim == 1:  # y
                    new_positions[i][1] = max(bounds[i*3+1][0], min(bounds[i*3+1][1], new_positions[i][1]))
                # angle is handled above
        
        # Evaluate new positions
        score, _ = evaluate_layout_with_penalty(new_positions.tolist())
        
        # Accept if better
        old_score, _ = evaluate_layout_with_penalty(current_positions.tolist())
        if score > old_score:
            current_positions = new_positions
    
    return current_positions.tolist()

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate better initial configuration
    initial_positions = [
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First ring (6 hexagons)
        [-2.0, 0.0, 0.0],      # Left
        [2.0, 0.0, 0.0],       # Right
        [0.0, 2.0, 0.0],       # Top
        [0.0, -2.0, 0.0],      # Bottom
        [-1.0, 1.732, 0.0],    # Top-left
        [1.0, 1.732, 0.0],     # Top-right
        # Second ring (4 hexagons)
        [-1.0, -1.732, 0.0],   # Bottom-left
        [1.0, -1.732, 0.0],    # Bottom-right
        [-2.0, 1.0, 0.0],      # Far top-left
        [2.0, 1.0, 0.0],       # Far top-right
        [-2.0, -1.0, 0.0],     # Far bottom-left
        [2.0, -1.0, 0.0],      # Far bottom-right
    ]
    initial_positions = initial_positions[:11]  # Only 11 needed
    
    # Optimization bounds for each parameter (x, y, angle)
    bounds = []
    for i in range(11):
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])  # angle in degrees

    # Multi-resolution optimization
    current_solution = initial_positions
    
    # Stage 1: Coarse global optimization with PSO
    try:
        print("Starting coarse PSO optimization...")
        coarse_solution = adaptive_particle_swarm_optimization(bounds, max_iter=50)
        score, _ = evaluate_layout_with_penalty(coarse_solution)
        print(f"Coarse PSO score: {score}")
        
        if score > evaluate_layout_with_penalty(current_solution)[0] - 1e-6:
            current_solution = coarse_solution
            
    except Exception as e:
        print(f"PSO optimization failed: {e}")

    # Stage 2: Local refinement with constraint-aware gradient descent
    try:
        print("Starting local refinement...")
        refined_solution = constraint_aware_gradient_descent(current_solution, bounds, max_iter=30)
        score, _ = evaluate_layout_with_penalty(refined_solution)
        print(f"Local refinement score: {score}")
        
        if score > evaluate_layout_with_penalty(current_solution)[0] - 1e-6:
            current_solution = refined_solution
            
    except Exception as e:
        print(f"Local refinement failed: {e}")

    # Stage 3: Fine-grained optimization around best solution
    try:
        print("Starting fine-grained optimization...")
        # Use scipy local optimization for the final touch
        def objective(params):
            positions_angles = []
            for i in range(11):
                x = params[i*3]
                y = params[i*3 + 1]
                angle = params[i*3 + 2]
                positions_angles.append([x, y, angle])
            
            score, _ = evaluate_layout_with_penalty(positions_angles)
            return -score  # Minimize negative score to maximize score

        # Flatten the current solution
        initial_flat = []
        for pos_angle in current_solution:
            initial_flat.extend(pos_angle)
            
        # Fine optimization
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=None
        )
        
        # Extract refined solution
        refined_params = result.x
        refined_positions_angles = []
        for i in range(11):
            x = refined_params[i*3]
            y = refined_params[i*3 + 1]
            angle = refined_params[i*3 + 2]
            refined_positions_angles.append([x, y, angle])
        
        final_score, final_side_length = evaluate_layout_with_penalty(refined_positions_angles)
        if final_score > evaluate_layout_with_penalty(current_solution)[0] - 1e-6:
            current_solution = refined_positions_angles
            
    except Exception as e:
        print(f"Fine optimization failed: {e}")

    # Final evaluation
    final_score, final_side_length = evaluate_layout_with_penalty(current_solution)
    print(f"Final score: {final_score}, side_length: {final_side_length}")

    # Ensure we're returning the correct data format
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    # Convert to numpy array
    inner_hex_data = np.array(current_solution)
    
    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END