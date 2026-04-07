# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
import math

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0  # Distance from center to corner for unit hexagon
UNIT_HEXAGON_WIDTH = 2.0  # Diameter of unit hexagon
MAX_EVAL_TIME = 180.0  # seconds

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def is_contained_in_outer_hexagon(hexagon_vertices_list, outer_center, outer_angle, outer_radius):
    """Check if hexagon is fully contained in outer hexagon using optimized approach"""
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Fast check: test if all vertices are inside outer polygon
    for vertex in hexagon_vertices_list:
        if not point_in_polygon(vertex, outer_polygon):
            return False
    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using spatial indexing and bounding box pruning"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)

        # Quick bounding box check first
        bbox1 = poly1.bounds
        bbox2 = poly2.bounds

        if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
            bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
            return False

        return poly1.intersects(poly2)
    except:
        # Fallback for degenerate cases
        return False

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution: check containment and non-overlap"""
    # Check containment first
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate outer radius based on this solution to check containment properly
        outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

        if not is_contained_in_outer_hexagon(vertices, outer_center, outer_angle, outer_radius):
            return False

    # Check overlaps using spatial indexing with grid-based approach
    # Create centroids for spatial indexing
    centroids = []
    hex_polygons = []
    
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))
        # Use centroid for spatial indexing
        centroid = np.mean(vertices, axis=0)
        centroids.append(centroid)

    if len(centroids) == 0:
        return False
    
    # Convert to numpy array for kdtree
    centroids = np.array(centroids)
    
    # Create spatial index (KDTree) for efficient neighbor search
    try:
        tree = cKDTree(centroids)
        
        # Find candidates for overlap checking using distance threshold
        # Only check pairs that are close enough to potentially overlap
        pairs = tree.query_pairs(r=UNIT_HEXAGON_WIDTH * 2.0, eps=0)
        
        # Check actual overlaps for candidate pairs
        for i, j in pairs:
            if hex_polygons[i].intersects(hex_polygons[j]):
                return False
    except Exception:
        # Fallback to brute force for small numbers or edge cases
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return False

    return True

def evaluate_fitness(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness (negative of outer hexagon radius for maximization)"""
    # Calculate minimum outer radius needed
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # If solution is invalid, penalize heavily
    if not validate_solution(inner_hex_data, outer_center, outer_angle):
        return -1e10  # Very poor fitness

    # Return negative radius (we want to minimize radius, so maximize negative value)
    return -outer_radius

def compute_greedy_score(position, angle, existing_hexes, outer_radius):
    """Compute a greedy score for placing a hexagon at given position and angle"""
    # Score components:
    # 1. Density factor: how much space is already occupied nearby
    # 2. Boundary proximity: avoid placing near boundary
    # 3. Available space: measure of free space around location
    
    # Create hexagon polygon for this candidate
    vertices = hexagon_vertices(position, np.radians(angle), UNIT_HEXAGON_RADIUS)
    candidate_poly = Polygon(vertices)
    
    # Calculate how many existing hexes are close (within 2 units)
    nearby_count = 0
    total_distance = 0
    for i, hex_data in enumerate(existing_hexes):
        center = hex_data[:2]
        hex_angle = np.radians(hex_data[2])
        hex_vertices = hexagon_vertices(center, hex_angle, UNIT_HEXAGON_RADIUS)
        hex_poly = Polygon(hex_vertices)
        
        # Check distance between centroids
        centroid1 = np.mean(vertices, axis=0)
        centroid2 = np.mean(hex_vertices, axis=0)
        distance = np.linalg.norm(centroid1 - centroid2)
        
        if distance < 2.0:  # Close enough to be considered
            nearby_count += 1
            total_distance += distance
    
    # How close is this candidate to the center?
    center_distance = np.linalg.norm(np.array(position))
    
    # How close is this candidate to the outer boundary?
    boundary_distance = outer_radius - center_distance
    
    # Compute scores
    proximity_score = -nearby_count * 0.1  # Prefer less crowded areas
    boundary_score = max(0, boundary_distance - 1.0) * 0.5  # Prefer center areas
    center_score = -center_distance * 0.01  # Prefer center areas
    
    # Final score - higher means better placement
    return proximity_score + boundary_score + center_score - (total_distance * 0.05)

def construct_greedy_solution(remaining_hexes, outer_radius, max_attempts=100):
    """Construct initial solution using greedy approach with randomization"""
    if len(remaining_hexes) == 0:
        return []
    
    solution = []
    
    # Start with center hexagon
    if len(remaining_hexes) > 0:
        center_hex = [0.0, 0.0, 0.0]  # center at origin, no rotation
        solution.append(center_hex)
        remaining_hexes = remaining_hexes[1:] if len(remaining_hexes) > 1 else []
    
    # Greedy construction phase
    attempts = 0
    while len(remaining_hexes) > 0 and attempts < max_attempts:
        best_position = None
        best_angle = None
        best_score = float('-inf')
        
        # Sample possible locations and evaluate them
        candidates = []
        
        # Add some predefined strategic locations
        strategic_locations = [
            # Around the perimeter
            (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (-2.0, 0.0, 0.0), (0.0, -2.0, 0.0),
            # Diagonal locations
            (1.5, 1.5, 0.0), (1.5, -1.5, 0.0), (-1.5, 1.5, 0.0), (-1.5, -1.5, 0.0),
            # Further away
            (3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (-3.0, 0.0, 0.0), (0.0, -3.0, 0.0),
        ]
        
        # Add random samples
        for _ in range(50):
            # Generate random position within reasonable bounds
            x = random.uniform(-outer_radius + 1.5, outer_radius - 1.5)
            y = random.uniform(-outer_radius + 1.5, outer_radius - 1.5)
            angle = random.uniform(0, 360)
            candidates.append((x, y, angle))
            
        # Add strategic locations
        for loc in strategic_locations:
            if random.random() < 0.7:  # Add some randomly
                candidates.append(loc)
        
        # Evaluate candidates
        for x, y, angle in candidates:
            # Check if this location is likely to fit
            test_pos = [x, y]
            score = compute_greedy_score(test_pos, angle, solution, outer_radius)
            
            # Add a bit of randomness to encourage exploration
            rand_factor = random.uniform(0, 0.1)
            adjusted_score = score + rand_factor
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_position = (x, y)
                best_angle = angle
        
        # Place the best candidate if found
        if best_position is not None and best_angle is not None:
            solution.append([best_position[0], best_position[1], best_angle])
            if len(remaining_hexes) > 0:
                remaining_hexes = remaining_hexes[1:] if len(remaining_hexes) > 1 else []
        else:
            # If we couldn't place anything, just take the first remaining
            if len(remaining_hexes) > 0:
                solution.append(remaining_hexes[0])
                remaining_hexes = remaining_hexes[1:] if len(remaining_hexes) > 1 else []
        
        attempts += 1
    
    # Fill up if not enough hexes were placed
    while len(solution) < NUM_INNER_HEXAGONS:
        # Fill with random placements
        x = random.uniform(-outer_radius + 1.5, outer_radius - 1.5)
        y = random.uniform(-outer_radius + 1.5, outer_radius - 1.5)
        angle = random.uniform(0, 360)
        solution.append([x, y, angle])
    
    return solution[:NUM_INNER_HEXAGONS]

def local_search_improve(solution, max_iterations=200):
    """Improve solution via local search"""
    current_solution = [list(h) for h in solution]
    current_fitness = evaluate_fitness(current_solution)
    
    for iteration in range(max_iterations):
        # Try to improve by perturbing one hexagon at a time
        best_improvement = 0
        best_move = None
        
        for i in range(len(current_solution)):
            original_pos = [current_solution[i][0], current_solution[i][1]]
            original_angle = current_solution[i][2]
            
            # Try several small moves
            for _ in range(10):
                # Small positional adjustments
                dx = random.uniform(-0.1, 0.1)
                dy = random.uniform(-0.1, 0.1)
                dangle = random.uniform(-2, 2)
                
                new_pos = [original_pos[0] + dx, original_pos[1] + dy]
                new_angle = (original_angle + dangle) % 360
                
                # Create a candidate solution
                candidate = [list(h) for h in current_solution]
                candidate[i] = [new_pos[0], new_pos[1], new_angle]
                
                # Evaluate
                fitness = evaluate_fitness(candidate)
                improvement = fitness - current_fitness
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (i, new_pos, new_angle)
        
        # If we found an improvement, apply it
        if best_move and best_improvement > 0:
            idx, pos, angle = best_move
            current_solution[idx] = [pos[0], pos[1], angle]
            current_fitness += best_improvement
        else:
            # No improvement found, stop early
            break
            
    return current_solution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses GRASP heuristic with greedy construction and local search improvement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Multi-start approach with GRASP
    best_overall_fitness = float('-inf')
    best_overall_individual = None

    # Run multiple independent optimizations
    num_starts = 8
    seeds = [42, 123, 456, 789, 999, 1001, 2023, 3033]
    
    for start_num in range(num_starts):
        if time.time() - start_time > MAX_EVAL_TIME - 1:  # Leave 1 second for final processing
            break
            
        seed = seeds[start_num] if start_num < len(seeds) else start_num * 100
        random.seed(seed)
        np.random.seed(seed)
        
        # Create initial set of hexagon data (positions + rotations)
        # Start with a base configuration for diversity
        base_positions = [
            [0, 0, 0],           # center
            [-2.5, 0, 0],       # left
            [2.5, 0, 0],        # right
            [-1.25, 2.17, 0],   # top-left
            [1.25, 2.17, 0],    # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],   # bottom-right
            [-3.75, 2.17, 0],   # far top-left
            [3.75, 2.17, 0],    # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],   # far bottom-right
        ]
        
        # Create diverse initial hexagon data
        initial_hex_data = []
        for pos in base_positions:
            x = pos[0] + random.uniform(-0.3, 0.3)
            y = pos[1] + random.uniform(-0.3, 0.3)
            angle = pos[2] + random.uniform(-10, 10)
            initial_hex_data.append([x, y, angle])
        
        # Use GRASP approach
        initial_solution = np.array(initial_hex_data)
        
        # Improve with greedy construction and local search
        # First initialize with a reasonable outer radius estimate
        est_outer_radius = calculate_outer_hexagon_radius(initial_solution)
        if est_outer_radius < 2.0:
            est_outer_radius = 3.0  # Minimum reasonable
            
        # Use our GRASP-like construction
        remaining_hexes = initial_solution[1:]  # Exclude center for now
        reconstructed_solution = construct_greedy_solution(remaining_hexes, est_outer_radius)
        
        # Add center back
        reconstructed_solution.insert(0, [0.0, 0.0, 0.0])
        
        # Local search improvement
        refined_solution = local_search_improve(reconstructed_solution, max_iterations=200)
        
        # Final fitness evaluation
        final_fitness = evaluate_fitness(refined_solution)
        
        if final_fitness > best_overall_fitness:
            best_overall_fitness = final_fitness
            best_overall_individual = np.array(refined_solution)

    # Validate final best solution
    if best_overall_individual is None:
        # Return fallback if we couldn't find anything good
        fallback = [
            [0, 0, 0],           # center
            [-2.5, 0, 0],       # left
            [2.5, 0, 0],        # right
            [-1.25, 2.17, 0],   # top-left
            [1.25, 2.17, 0],    # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],   # bottom-right
            [-3.75, 2.17, 0],   # far top-left
            [3.75, 2.17, 0],    # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],   # far bottom-right
        ]
        best_overall_individual = np.array(fallback)

    # Final validation and calculation of outer hexagon parameters
    outer_radius = -best_overall_fitness if best_overall_fitness != float('-inf') else 10.0  # fallback if not found

    # Return result
    inner_hex_data = best_overall_individual
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = outer_radius

    # Final validation to ensure correctness
    if not validate_solution(inner_hex_data):
        # Revert to a reasonable fallback
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
        outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END