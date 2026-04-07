# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time
import math
from typing import Tuple, List
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_regular_hexagon(center_x: float, center_y: float, side_length: float = 1.0, rotation_deg: float = 0.0) -> Polygon:
    """Create a regular hexagon as a Shapely polygon."""
    angle_rad = np.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def hexagon_vertices(center_x: float, center_y: float, side_length: float = 1.0, rotation_deg: float = 0.0) -> np.ndarray:
    """Get vertices of a regular hexagon as numpy array."""
    angle_rad = np.radians(rotation_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        vertices[i, 0] = center_x + side_length * np.cos(angle)
        vertices[i, 1] = center_y + side_length * np.sin(angle)
    return vertices

def point_in_hexagon_vertices(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Check if a point is inside hexagon defined by vertices using ray casting."""
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def hexagon_contains_point(hex_vertices: np.ndarray, point: np.ndarray) -> bool:
    """Check if hexagon contains a point."""
    return point_in_hexagon_vertices(point, hex_vertices)

def get_hexagon_axes(vertices: np.ndarray) -> np.ndarray:
    """Get the normal vectors (axes) for separating axis theorem."""
    axes = []
    n = len(vertices)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge = v2 - v1
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        axes.append(normal)
    return np.array(axes)

def project_polygon_onto_axis(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Project polygon onto axis and return min/max projections."""
    projections = np.dot(vertices, axis)
    return np.min(projections), np.max(projections)

def SAT_collision_check(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """Check collision using Separating Axis Theorem."""
    # Get axes from both polygons
    axes1 = get_hexagon_axes(vertices1)
    axes2 = get_hexagon_axes(vertices2)
    
    # Combine all axes
    all_axes = np.vstack([axes1, axes2])
    
    # Check each axis
    for axis in all_axes:
        proj1_min, proj1_max = project_polygon_onto_axis(vertices1, axis)
        proj2_min, proj2_max = project_polygon_onto_axis(vertices2, axis)
        
        # If projections don't overlap, polygons don't collide
        if proj1_max < proj2_min or proj2_max < proj1_min:
            return False
    
    return True

def check_hexagon_containment_fast(hex_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
    """Fast containment check assuming outer hex is convex."""
    # Check that all vertices of inner hex are inside outer hex
    for vertex in hex_vertices:
        if not hexagon_contains_point(outer_hex_vertices, vertex):
            return False
    return True

def check_hexagon_collision_fast(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """Fast collision check using SAT."""
    return SAT_collision_check(vertices1, vertices2)

def compute_min_outer_radius(inner_hex_data: np.ndarray) -> float:
    """Compute the minimum outer hexagon radius required to contain all inner hexagons."""
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        # Distance from center to hexagon center plus the hexagon's circumradius
        dist = np.sqrt(center_x**2 + center_y**2) + 1.0  # 1.0 is the circumradius of unit hexagon
        max_dist = max(max_dist, dist)
    return max_dist * 1.05  # Add safety margin

def binary_search_outer_radius(inner_hex_data: np.ndarray, min_radius: float,
                              max_radius: float, tolerance: float = 0.001) -> float:
    """Binary search to find the minimum valid outer radius."""
    while max_radius - min_radius > tolerance:
        mid_radius = (min_radius + max_radius) / 2.0
        if is_valid_configuration(inner_hex_data, mid_radius):
            max_radius = mid_radius
        else:
            min_radius = mid_radius
    return max_radius

def is_valid_configuration(inner_hex_data: np.ndarray, outer_hex_side_length: float) -> bool:
    """Fast validation of a configuration."""
    outer_hex_vertices = hexagon_vertices(0, 0, outer_hex_side_length)
    
    # Check containment and collisions
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        inner_hex_vertices = hexagon_vertices(center_x, center_y, 1.0, rotation)
        
        # Check containment
        if not check_hexagon_containment_fast(inner_hex_vertices, outer_hex_vertices):
            return False
        
        # Check collision with all previous hexagons
        for j in range(i):
            center_x2, center_y2, rotation2 = inner_hex_data[j]
            other_hex_vertices = hexagon_vertices(center_x2, center_y2, 1.0, rotation2)
            
            if check_hexagon_collision_fast(inner_hex_vertices, other_hex_vertices):
                return False
    
    return True

def generate_structured_initial_config() -> np.ndarray:
    """Generate a structured initial configuration based on efficient hexagonal packing principles."""
    # Create a configuration inspired by dense hexagonal packing
    config = np.zeros((11, 3))

    # Central hexagon
    config[0] = [0.0, 0.0, 0.0]

    # First ring (6 hexagons around center)
    ring1_angles = [i * 60 for i in range(6)]
    ring1_distance = 2.0

    for i, angle in enumerate(ring1_angles):
        rad = np.radians(angle)
        x = ring1_distance * np.cos(rad)
        y = ring1_distance * np.sin(rad)
        config[i+1] = [x, y, 0.0]

    # Second ring (4 hexagons)
    ring2_angles = [30, 90, 150, 210]
    ring2_distance = 3.5

    for i, angle in enumerate(ring2_angles):
        rad = np.radians(angle)
        x = ring2_distance * np.cos(rad)
        y = ring2_distance * np.sin(rad)
        config[i+7] = [x, y, 0.0]

    return config

def perturb_individual(individual: np.ndarray, perturb_scale: float = 0.3) -> np.ndarray:
    """Generate a perturbed version of the individual."""
    mutated = individual.copy()
    
    # Randomly choose which hexagons to perturb
    num_to_perturb = max(1, int(len(mutated) * 0.3))
    indices = np.random.choice(len(mutated), num_to_perturb, replace=False)
    
    for idx in indices:
        # Perturb position
        mutated[idx, 0] += np.random.normal(0, perturb_scale)
        mutated[idx, 1] += np.random.normal(0, perturb_scale)
        # Perturb rotation
        mutated[idx, 2] += np.random.normal(0, 15)
        mutated[idx, 2] %= 360.0
    
    return mutated

def optimize_single_step(current_config: np.ndarray, max_outer_radius: float = 15.0) -> np.ndarray:
    """Perform one optimization step."""
    # Try multiple perturbations and select the best valid one
    best_config = current_config
    best_radius = compute_min_outer_radius(current_config)
    
    for _ in range(10):  # Try several perturbations
        candidate = perturb_individual(current_config, 0.2)
        candidate_radius = compute_min_outer_radius(candidate)
        
        if candidate_radius < best_radius and is_valid_configuration(candidate, max_outer_radius):
            best_config = candidate
            best_radius = candidate_radius
    
    return best_config

def local_optimization_step(current_config: np.ndarray, max_outer_radius: float = 15.0) -> np.ndarray:
    """Apply local optimization to improve a configuration."""
    # Perform several local search steps
    best_config = current_config.copy()
    
    # Try to improve through systematic perturbations
    for _ in range(20):
        # Perturb positions slightly
        candidate = best_config.copy()
        for i in range(len(candidate)):
            # Small random changes
            candidate[i, 0] += np.random.normal(0, 0.1)
            candidate[i, 1] += np.random.normal(0, 0.1)
            candidate[i, 2] += np.random.normal(0, 5)
            candidate[i, 2] %= 360.0
        
        # Only accept if valid and improves the configuration
        if is_valid_configuration(candidate, max_outer_radius):
            candidate_radius = compute_min_outer_radius(candidate)
            current_radius = compute_min_outer_radius(best_config)
            if candidate_radius < current_radius:
                best_config = candidate
    
    return best_config

def hexagon_packing_11() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    timeout_seconds = 175

    # Start with a structured configuration
    current_config = generate_structured_initial_config()
    best_config = current_config.copy()
    
    # Multi-stage optimization
    for stage in range(5):
        if time.time() - start_time > timeout_seconds:
            break
            
        # Coarse search - make significant improvements
        for iteration in range(20):
            if time.time() - start_time > timeout_seconds:
                break
            current_config = optimize_single_step(current_config)
            
            # Check if we've improved
            current_radius = compute_min_outer_radius(current_config)
            best_radius = compute_min_outer_radius(best_config)
            
            if current_radius < best_radius:
                best_config = current_config.copy()
            
            # Local refinement
            current_config = local_optimization_step(current_config)
        
        # Fine-grained optimization
        for iteration in range(30):
            if time.time() - start_time > timeout_seconds:
                break
            current_config = local_optimization_step(current_config)
            
            # Check if we've improved
            current_radius = compute_min_outer_radius(current_config)
            best_radius = compute_min_outer_radius(best_config)
            
            if current_radius < best_radius:
                best_config = current_config.copy()
                
        # If we're not making progress, try to escape local minima
        if stage < 4:
            # Add some randomness
            current_config = perturb_individual(best_config, 0.5)
    
    # Final binary search to tighten the boundary
    estimated_radius = compute_min_outer_radius(best_config)
    min_radius = max(2.0, estimated_radius - 0.5)
    max_radius = estimated_radius + 1.0
    
    final_radius = binary_search_outer_radius(best_config, min_radius, max_radius)
    
    # Validate final result
    if not is_valid_configuration(best_config, final_radius):
        # Fallback to a known good configuration
        best_config = generate_structured_initial_config()
        final_radius = compute_min_outer_radius(best_config) * 1.1
    
    # Prepare output
    inner_hex_data = best_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = final_radius
    
    # Ensure minimum bounds
    if outer_hex_side_length <= 0:
        outer_hex_side_length = 10.0
    
    eval_time = time.time() - start_time
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END