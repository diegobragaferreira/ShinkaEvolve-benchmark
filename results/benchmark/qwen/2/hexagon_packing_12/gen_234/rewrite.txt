# EVOLVE-BLOCK-START
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from shapely.geometry import Polygon
from numba import jit
import time
import random
from scipy.stats import norm

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given position, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using separating axis theorem"""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_vertices:
        if point_in_polygon(v, hex2_vertices):
            return True
    # Check if any vertex of hex2 is inside hex1  
    for v in hex2_vertices:
        if point_in_polygon(v, hex1_vertices):
            return True
    return False

def create_spatial_hash(hex_vertices_list, cell_size=2.0):
    """Create spatial hash grid for fast overlap checking"""
    hash_grid = {}
    for i, vertices in enumerate(hex_vertices_list):
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        # Add to all relevant cells
        start_col = int(min_x // cell_size)
        end_col = int(max_x // cell_size) + 1
        start_row = int(min_y // cell_size)
        end_row = int(max_y // cell_size) + 1

        for col in range(start_col, end_col + 1):
            for row in range(start_row, end_row + 1):
                cell_key = (col, row)
                if cell_key not in hash_grid:
                    hash_grid[cell_key] = []
                hash_grid[cell_key].append(i)
    return hash_grid

def get_overlapping_indices(hash_grid, hex_index, hex_vertices, cell_size=2.0):
    """Get indices of potentially overlapping hexagons using spatial hash"""
    overlapping = set()
    # Get bounding box of hexagon
    min_x = min(v[0] for v in hex_vertices)
    max_x = max(v[0] for v in hex_vertices)
    min_y = min(v[1] for v in hex_vertices)
    max_y = max(v[1] for v in hex_vertices)

    # Check all relevant cells
    start_col = int(min_x // cell_size)
    end_col = int(max_x // cell_size) + 1
    start_row = int(min_y // cell_size)
    end_row = int(max_y // cell_size) + 1

    for col in range(start_col, end_col + 1):
        for row in range(start_row, end_row + 1):
            cell_key = (col, row)
            if cell_key in hash_grid:
                for idx in hash_grid[cell_key]:
                    if idx != hex_index:
                        overlapping.add(idx)
    return overlapping

def create_initial_config():
    """Create an initial symmetric configuration based on group theory"""
    # Generate 12 hexagon positions in concentric rings
    positions = []

    # Center hexagon
    positions.append([0.0, 0.0, 0.0])

    # First ring around center (6 hexagons)
    for i in range(6):
        angle = i * 60
        radius = 1.732  # sqrt(3) - approximately optimal spacing
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    # Second ring (5 hexagons)
    for i in range(5):
        angle = i * 72
        radius = 3.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    return np.array(positions)

def get_outer_hexagon_radius(inner_hex_data):
    """Compute the minimum radius required to contain all hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, angle)
        for vx, vy in vertices:
            dist = np.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)
    return max_dist + 0.5  # Add small margin

def evaluate_packing(inner_hex_data):
    """Evaluate how well a packing satisfies constraints"""
    n = len(inner_hex_data)
    penalty = 0

    # Precompute vertices for all hexagons for efficient overlap checking
    hex_vertices_list = [hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2]) for i in range(n)]

    # Check containment (penalize if hexagons extend beyond boundary)
    outer_radius = get_outer_hexagon_radius(inner_hex_data)
    outer_hex_vertices = hexagon_vertices(0, 0, 0, outer_radius)

    for i in range(n):
        vertices = hex_vertices_list[i]

        # Check if all vertices are inside outer hexagon
        for vx, vy in vertices:
            point = np.array([vx, vy])
            if not point_in_polygon(point, outer_hex_vertices):
                # Calculate penalty based on how much it extends beyond
                dist = np.sqrt(vx*vx + vy*vy)
                penalty += (dist - outer_radius + 0.5)**2

    # Check overlaps between hexagons using spatial hashing for efficiency
    hash_grid = create_spatial_hash(hex_vertices_list)

    for i in range(n):
        # Get potentially overlapping hexagons
        overlapping_indices = get_overlapping_indices(hash_grid, i, hex_vertices_list[i])

        # Check actual overlaps
        for j in overlapping_indices:
            if i < j:  # Avoid double counting
                vertices1 = hex_vertices_list[i]
                vertices2 = hex_vertices_list[j]

                if check_hexagon_overlap(vertices1, vertices2):
                    penalty += 1000000  # Large penalty for overlaps
                    break

    return penalty

def bayesian_optimization_hexagon_packing():
    """
    Uses Bayesian Optimization with Gaussian Processes to find optimal hexagon packing.
    """
    # Define bounds for 12 hexagons (x, y, angle for each) and outer hexagon size
    # Each hexagon has 3 parameters: x, y, angle; outer hexagon size is last parameter
    bounds = []
    for _ in range(12):  # 12 hexagons
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)])  # x, y, angle
    bounds.append((1.0, 10.0))  # outer hexagon side length

    # Convert bounds to numpy array for easier handling
    bounds_array = np.array(bounds)
    
    # Define acquisition function - Expected Improvement (EI)
    def expected_improvement(X, X_sample, Y_sample, gp, xi=0.01):
        mu, sigma = gp.predict(X, return_std=True)
        mu_sample = gp.predict(X_sample)

        # Needed for avoiding division by zero
        sigma = sigma + 1e-8

        # Calculate EI
        mu_sample_opt = np.max(mu_sample)
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return -ei  # Negative because we're minimizing

    # Sample initial configurations
    sample_size = 10
    X_sample = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1], (sample_size, len(bounds_array)))
    Y_sample = []
    
    # Evaluate initial samples
    for i in range(sample_size):
        # Reshape sample to configuration format
        config = X_sample[i].reshape(-1, 3)
        # Evaluate penalty (we want to minimize this)
        penalty = evaluate_packing(config)
        # The objective is to minimize penalty and maximize 1/R, so we use -1/R + penalty
        outer_radius = get_outer_hexagon_radius(config)
        objective = penalty - 1.0/(outer_radius + 1e-6)
        Y_sample.append(objective)
    
    Y_sample = np.array(Y_sample)
    
    # Fit Gaussian Process
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
    gp.fit(X_sample, Y_sample)
    
    # Bayesian Optimization iterations
    n_iterations = 20
    for iteration in range(n_iterations):
        # Optimize acquisition function to find next sample point
        def objective_func(x):
            return expected_improvement(x.reshape(1, -1), X_sample, Y_sample, gp)
        
        # Random start points for optimization
        best_x = None
        best_value = np.inf
        for _ in range(5):
            start_point = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])
            result = minimize(objective_func, start_point, bounds=[(b[0], b[1]) for b in bounds_array], method='L-BFGS-B')
            if result.fun < best_value:
                best_value = result.fun
                best_x = result.x
        
        if best_x is not None:
            # Evaluate new point
            config = best_x.reshape(-1, 3)
            penalty = evaluate_packing(config)
            outer_radius = get_outer_hexagon_radius(config)
            objective = penalty - 1.0/(outer_radius + 1e-6)
            
            # Add new point to samples
            X_sample = np.vstack([X_sample, best_x])
            Y_sample = np.append(Y_sample, objective)
            
            # Refit GP with new data
            gp.fit(X_sample, Y_sample)
    
    # Return best solution found
    best_idx = np.argmin(Y_sample)
    best_config = X_sample[best_idx].reshape(-1, 3)
    return best_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Run Bayesian optimization
        inner_hex_data = bayesian_optimization_hexagon_packing()
    except Exception as e:
        # Fallback to the symmetric configuration if optimization fails
        print(f"Fallback due to optimization error: {e}")
        inner_hex_data = create_initial_config()
    
    # Compute final outer hexagon size
    outer_hex_side_length = get_outer_hexagon_radius(inner_hex_data)
    
    # Create outer hexagon data (centered, no rotation)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin, no rotation
    
    # Validate final configuration
    final_penalty = evaluate_packing(inner_hex_data)
    if final_penalty > 10000:  # If there are major overlaps or violations
        # Fallback to the original simpler solution
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
        outer_hex_side_length = 8  # large enough to contain all inner hexagons
    
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END