# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
import time
from math import sqrt

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_radius(inner_positions, inner_angles, initial_radius_estimate=5.0):
    """Compute minimum outer hexagon radius using binary search with adaptive precision"""
    # Binary search with adaptive precision based on convergence
    left = initial_radius_estimate
    right = 20.0
    best_radius = right
    
    # Track convergence to adjust precision dynamically
    prev_diff = float('inf')
    max_iterations = 50
    iterations = 0
    
    while iterations < max_iterations:
        current_diff = right - left
        # Adaptive precision: more precise as we converge
        if abs(current_diff - prev_diff) < 1e-3 and current_diff > 1e-4:
            precision_threshold = 1e-6
        else:
            precision_threshold = 1e-4
            
        if current_diff <= precision_threshold:
            break
            
        mid = (left + right) / 2.0
        outer_vertices = hexagon_vertices(0, 0, 0, mid)
        valid = True
        
        # Check all inner hexagons
        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
            if not check_containment(hex_vertices, outer_vertices):
                valid = False
                break
                
        if valid:
            best_radius = mid
            right = mid
        else:
            left = mid
            
        prev_diff = current_diff
        iterations += 1
        
    return best_radius

def evaluate_fitness_hexagon_config(positions, angles):
    """Evaluate fitness for a hexagon configuration"""
    # Check overlap constraint first (early rejection)
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            hex1_vertices = hexagon_vertices(positions[i][0], positions[i][1], angles[i])
            hex2_vertices = hexagon_vertices(positions[j][0], positions[j][1], angles[j])
            if check_overlap(hex1_vertices, hex2_vertices):
                return -1e10, 1e10  # Invalid configuration penalty
                
    # Compute outer radius
    outer_radius = compute_outer_hexagon_radius(positions, angles)
    
    # Check containment constraint
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        if not check_containment(hex_vertices, outer_vertices):
            return -1e10, 1e10  # Invalid containment penalty
            
    # Return 1/radius as fitness (higher is better)
    return 1.0 / outer_radius, outer_radius

def generate_geometric_initial_config():
    """Generate initial configuration using geometric principles for better packing"""
    # Hexagonal close packing arrangement optimized for 11 hexagons
    # This is based on mathematical analysis of optimal hexagon packings
    
    # Central hexagon
    positions = [[0.0, 0.0]]
    
    # First ring: 6 hexagons at unit distance
    for i in range(6):
        angle = i * np.pi/3
        positions.append([np.cos(angle), np.sin(angle)])
    
    # Second ring: 4 hexagons in triangular formation
    positions.append([-1.5, 0.0])  # Left
    positions.append([1.5, 0.0])   # Right
    positions.append([0.0, -1.5])  # Bottom
    positions.append([0.0, 1.5])   # Top
    
    # Convert to numpy array and add rotation (all initially 0)
    positions = np.array(positions[:11])  # Take only first 11
    angles = np.zeros(11)
    
    # Add small random perturbations to escape symmetry
    np.random.seed(42)
    positions += np.random.normal(0, 0.05, positions.shape)
    
    return positions, angles

def project_to_hexagonal_lattice(points, lattice_spacing=1.0):
    """Project points to nearest hexagonal lattice points for geometric constraints"""
    # Simplified projection to hexagonal lattice structure
    projected = []
    for pt in points:
        x, y = pt
        # Round to hexagonal lattice with proper spacing
        # Project to nearest hexagonal grid point
        # Using hexagonal coordinates system: q, r, s where q+r+s=0
        # This is a simplified approximation
        q = x * 2/3 - y * 1/3
        r = -x * 1/3 + y * 2/3
        s = -(q + r)
        
        # Round to nearest integers
        q_round = round(q)
        r_round = round(r)
        s_round = round(s)
        
        # Convert back to Cartesian
        # This is approximate but preserves hexagonal structure
        px = (2*q_round - r_round) * lattice_spacing / 3
        py = (r_round - q_round) * lattice_spacing * sqrt(3) / 3
        projected.append([px, py])
        
    return np.array(projected)

def constrained_gradient_optimization(initial_positions, initial_angles):
    """Use constrained optimization to refine the geometric configuration"""
    # Flatten initial data for optimization
    initial_vars = np.concatenate([initial_positions.flatten(), initial_angles])
    
    def objective_function(vars):
        # Reshape variables back to positions and angles
        pos_flat = vars[:-11]
        angles = vars[-11:]
        positions = pos_flat.reshape(-1, 2)
        
        # Apply geometric constraints
        # Ensure positions stay approximately in hexagonal pattern
        # Project to hexagonal lattice structure
        projected_pos = project_to_hexagonal_lattice(positions)
        
        # Evaluate fitness
        fitness, _ = evaluate_fitness_hexagon_config(projected_pos, angles)
        return -fitness  # Minimize negative fitness = maximize fitness
    
    # Constraints to maintain reasonable bounds
    bounds = []
    for i in range(len(initial_positions)):
        bounds.extend([(-10, 10), (-10, 10)])  # Position bounds
    for i in range(len(initial_angles)):
        bounds.extend([(0, 360)])  # Angle bounds
    
    # Perform optimization
    try:
        result = minimize(
            objective_function,
            initial_vars,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            final_vars = result.x
            pos_flat = final_vars[:-11]
            angles = final_vars[-11:]
            positions = pos_flat.reshape(-1, 2)
            return positions, angles
    except:
        pass
    
    # Return original if optimization fails
    return initial_positions, initial_angles

def geometric_hexagon_tiling_optimization():
    """Main geometric optimization approach focusing on tiling structure"""
    # Generate initial geometric configuration
    positions, angles = generate_geometric_initial_config()
    
    # Apply constrained optimization
    refined_positions, refined_angles = constrained_gradient_optimization(positions, angles)
    
    # Final fitness evaluation
    fitness, outer_radius = evaluate_fitness_hexagon_config(refined_positions, refined_angles)
    
    # If fitness is still poor, use fallback
    if fitness < 0.2:  # Threshold for acceptable solution
        positions, angles = generate_geometric_initial_config()
        refined_positions, refined_angles = constrained_gradient_optimization(positions, angles)
        fitness, outer_radius = evaluate_fitness_hexagon_config(refined_positions, refined_angles)
    
    return refined_positions, refined_angles, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use geometric tiling approach
    positions, angles, outer_radius = geometric_hexagon_tiling_optimization()
    
    # Format result properly
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Validate solution
    fitness, _ = evaluate_fitness_hexagon_config(positions, angles)
    if fitness < 0:
        # Fallback to safe configuration
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
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END