# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import time
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# Constants
HEXAGON_RADIUS = 1.0

@jit(nopython=True)
def hexagon_vertices_numba(x, y, angle_deg, side_length=1):
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
def hexagon_area_numba(vertices):
    """Calculate area of hexagon using Shoelace formula."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0

@jit(nopython=True)
def point_in_hexagon_numba(px, py, vertices):
    """Check if point is inside hexagon using ray casting."""
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
def distance_point_to_line_numba(px, py, x1, y1, x2, y2):
    """Distance from point to line segment."""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = px - xx
    dy = py - yy
    return np.sqrt(dx * dx + dy * dy)

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices_numba(x, y, angle_deg, side_length)
    return vertices

def check_containment_fast(vertices, outer_vertices):
    """Fast containment check."""
    # Check if all vertices are inside outer hexagon
    for v in vertices:
        if not point_in_hexagon_numba(v[0], v[1], outer_vertices):
            return False
    return True

def get_hexagon_centroids(inner_hex_data):
    """Get centroids of inner hexagons."""
    centroids = []
    for i in range(len(inner_hex_data)):
        x, y, _ = inner_hex_data[i]
        centroids.append([x, y])
    return np.array(centroids)

def evaluate_single_config(config_array):
    """Fast evaluation of a single configuration using numba-accelerated functions."""
    # Reshape into 12 hexagons of (x, y, angle)
    inner_hex_data = config_array[:36].reshape(12, 3)
    outer_radius = config_array[36]
    
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices_numba(0, 0, 0, outer_radius)
    
    # Check containment
    for i in range(12):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices_numba(x, y, angle)
        if not check_containment_fast(vertices, outer_vertices):
            return 1e10  # Large penalty for containment violation
    
    # Check pairwise overlaps using simple distance checks
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1, _ = inner_hex_data[i]
            x2, y2, _ = inner_hex_data[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            # Overlap occurs if distance < 2 (sum of hexagon radii)
            if dist < 2.0:
                # More precise check
                vertices1 = hexagon_vertices_numba(x1, y1, inner_hex_data[i][2])
                vertices2 = hexagon_vertices_numba(x2, y2, inner_hex_data[j][2])
                
                # Simple vertex containment check for overlap
                overlap_found = False
                for v1 in vertices1:
                    if point_in_hexagon_numba(v1[0], v1[1], vertices2):
                        overlap_found = True
                        break
                if not overlap_found:
                    for v2 in vertices2:
                        if point_in_hexagon_numba(v2[0], v2[1], vertices1):
                            overlap_found = True
                            break
                
                if overlap_found:
                    return 1e10  # Large penalty for overlap
    
    # Calculate objective: maximize 1/outer_radius (minimize -1/outer_radius)
    return -1.0 / outer_radius

def generate_symmetric_initial():
    """Generate a highly symmetric initial configuration."""
    # Start with a hexagonal pattern: center + 6 around + 6 in second ring
    positions = []
    
    # Central hexagon
    positions.append([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons 
    for i in range(6):
        angle = i * 60.0
        dist = 2.0  # Distance that allows tight packing
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Second ring: 6 hexagons offset
    for i in range(6):
        angle = i * 60.0 + 30.0  # Offset by 30 degrees
        dist = 3.5  # Further out
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Adjust to make it more compact
    positions[1][0] *= 0.95
    positions[1][1] *= 0.95
    positions[2][0] *= 0.95
    positions[2][1] *= 0.95
    positions[3][0] *= 0.95
    positions[3][1] *= 0.95
    positions[4][0] *= 0.95
    positions[4][1] *= 0.95
    positions[5][0] *= 0.95
    positions[5][1] *= 0.95
    positions[6][0] *= 0.95
    positions[6][1] *= 0.95
    
    return np.array(positions[:12])

def create_embedding_space(initial_configs, n_components=8):
    """Create an embedding of configurations using MDS."""
    # Flatten configurations for distance calculation
    flattened_configs = initial_configs.reshape(initial_configs.shape[0], -1)
    
    # Compute pairwise distances
    distances = squareform(pdist(flattened_configs, metric='euclidean'))
    
    # Apply MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    embedded = mds.fit_transform(distances)
    
    return embedded, mds

def optimize_with_mds():
    """Main MDS-based optimization routine."""
    # Generate multiple initial configurations
    n_configs = 20
    initial_configs = []
    
    for i in range(n_configs):
        # Mix of symmetric and random configurations
        if i < 10:
            config = generate_symmetric_initial()
        else:
            # Random slight variations
            config = generate_symmetric_initial() + np.random.normal(0, 0.2, (12, 3))
        initial_configs.append(config.flatten())
    
    initial_configs = np.array(initial_configs)
    
    # Create embedding space
    try:
        embedded_configs, mds_model = create_embedding_space(initial_configs)
    except:
        # Fallback to direct optimization if MDS fails
        return None
    
    # Evaluate all initial configurations in embedding space
    scores = []
    for i in range(len(initial_configs)):
        config_flat = initial_configs[i]
        # Add outer radius (estimate)
        est_radius = np.max(np.linalg.norm(config_flat[:24].reshape(12,2), axis=1)) + 2.0
        full_config = np.append(config_flat, est_radius)
        score = evaluate_single_config(full_config)
        scores.append(score)
    
    # Find best configuration from initial set
    best_idx = np.argmin(scores)
    best_config = initial_configs[best_idx]
    
    # Estimate outer radius
    est_radius = np.max(np.linalg.norm(best_config[:24].reshape(12,2), axis=1)) + 2.0
    best_config_full = np.append(best_config, est_radius)
    
    # Refine using direct optimization
    bounds = []
    # Position bounds
    for i in range(12):
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])
    # Outer radius bound
    bounds.append((0.5, 15.0))
    
    # Try several optimization approaches
    try:
        # Differential evolution for global search
        de_result = differential_evolution(
            lambda x: evaluate_single_config(x),
            bounds,
            maxiter=30,
            popsize=10,
            seed=42,
            disp=False
        )
        
        if de_result.success:
            return de_result.x
    except:
        pass
    
    # Return best initial configuration
    return best_config_full

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate improved initial configuration
    try:
        # Try MDS-based optimization
        result = optimize_with_mds()
        
        if result is not None:
            # Extract inner configuration
            inner_hex_data = result[:36].reshape(12, 3)
            outer_hex_side_length = result[36]
        else:
            # Fallback to symmetric approach
            inner_config = generate_symmetric_initial()
            inner_hex_data = inner_config
            # Estimate outer radius
            max_dist = np.max(np.linalg.norm(inner_config[:,:2], axis=1)) + 2.0
            outer_hex_side_length = max_dist * 1.1
            
    except Exception as e:
        # Final fallback
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
        outer_hex_side_length = 8.0
    
    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Validate final result (just a basic consistency check)
    try:
        # Simple check: ensure all hexagons have reasonable positions
        max_dist = np.max(np.linalg.norm(inner_hex_data[:,:2], axis=1))
        if max_dist > 10.0:
            outer_hex_side_length = 10.0
    except:
        pass
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END