# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import math

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Precomputed vertices of a unit regular hexagon centered at origin, oriented with one side horizontal
    def get_hexagon_vertices(center_x, center_y, angle_deg):
        angle_rad = math.radians(angle_deg)
        # Hexagon vertices in counterclockwise order
        vertices = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = center_x + math.cos(theta)
            y = center_y + math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    # Check if hexagon with given center and angle fits completely within the outer hexagon
    def is_contained(hex_center, hex_angle, outer_radius):
        vertices = get_hexagon_vertices(hex_center[0], hex_center[1], hex_angle)
        # Define outer hexagon vertices
        outer_vertices = []
        for i in range(6):
            theta = i * math.pi / 3
            x = outer_radius * math.cos(theta)
            y = outer_radius * math.sin(theta)
            outer_vertices.append((x, y))
        outer_poly = Polygon(outer_vertices)
        
        # Check that all vertices of inner hexagon are inside outer hexagon
        for vx, vy in vertices:
            point = Point(vx, vy)
            if not outer_poly.contains(point):
                return False
        return True
    
    # Check if two hexagons overlap
    def do_overlap(hex1_center, hex1_angle, hex2_center, hex2_angle):
        vertices1 = get_hexagon_vertices(hex1_center[0], hex1_center[1], hex1_angle)
        vertices2 = get_hexagon_vertices(hex2_center[0], hex2_center[1], hex2_angle)
        poly1 = Polygon(vertices1)
        poly2 = Polygon(vertices2)
        return poly1.intersects(poly2)
    
    # Find minimum radius that contains all hexagons (i.e. minimize outer_hex_side_length)
    def calculate_outer_radius(hex_data):
        max_dist = 0
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            # Calculate distance from origin to center
            dist = math.sqrt(center[0]**2 + center[1]**2)
            # Add radius of hexagon to get furthest point
            dist += 1.0  # Unit hexagon has circumradius 1
            max_dist = max(max_dist, dist)
        return max_dist
    
    # Optimized grid-based collision detection for faster evaluation
    class CollisionGrid:
        def __init__(self, cell_size=2.0):
            self.cell_size = cell_size
            self.grid = {}
            
        def add_hexagon(self, idx, center, angle):
            # Get bounding box of hexagon
            vertices = get_hexagon_vertices(center[0], center[1], angle)
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Determine which cells this hexagon covers
            start_col = int(min_x // self.cell_size)
            end_col = int(max_x // self.cell_size) + 1
            start_row = int(min_y // self.cell_size)
            end_row = int(max_y // self.cell_size) + 1
            
            for r in range(start_row, end_row + 1):
                for c in range(start_col, end_col + 1):
                    if (r, c) not in self.grid:
                        self.grid[(r, c)] = []
                    self.grid[(r, c)].append(idx)
                    
        def query_potential_collisions(self, idx, center, angle):
            # Find all hexagons in nearby cells
            vertices = get_hexagon_vertices(center[0], center[1], angle)
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            start_col = int(min_x // self.cell_size)
            end_col = int(max_x // self.cell_size) + 1
            start_row = int(min_y // self.cell_size)
            end_row = int(max_y // self.cell_size) + 1
            
            candidates = set()
            for r in range(start_row, end_row + 1):
                for c in range(start_col, end_col + 1):
                    if (r, c) in self.grid:
                        candidates.update(self.grid[(r, c)])
            return list(candidates)
            
        def clear(self):
            self.grid.clear()
    
    # Initialize with a good starting configuration inspired by known packings
    best_solution = None
    best_inv_radius = 0.0
    
    # Try multiple configurations to find good starting points
    configs_to_try = [
        # Configuration 1: Radial arrangement plus cluster
        {
            'centers': [
                (0, 0),
                (-2.0, 0),
                (2.0, 0),
                (0, 2.0),
                (0, -2.0),
                (-2.0, 2.0),
                (2.0, 2.0),
                (-2.0, -2.0),
                (2.0, -2.0),
                (1.0, 1.0),
                (-1.0, -1.0)
            ],
            'angles': [0]*11,
            'description': 'radial_cluster'
        },
        # Configuration 2: More compact arrangement
        {
            'centers': [
                (0, 0),
                (-2.5, 0),
                (2.5, 0),
                (0, 2.5),
                (0, -2.5),
                (-2.5, 2.5),
                (2.5, 2.5),
                (-2.5, -2.5),
                (2.5, -2.5),
                (0, 1.5),
                (0, -1.5)
            ],
            'angles': [0]*11,
            'description': 'compact'
        }
    ]
    
    # Try a few good starting configurations with local optimization
    for config in configs_to_try:
        centers = config['centers']
        angles = config['angles']
        
        # Create initial hex_data array
        hex_data = np.zeros((11, 3))
        for i, (center, angle) in enumerate(zip(centers, angles)):
            hex_data[i] = [center[0], center[1], angle]
        
        # Perform basic feasibility check
        valid = True
        grid = CollisionGrid()
        for i in range(11):
            center = (hex_data[i][0], hex_data[i][1])
            angle = hex_data[i][2]
            # First check if it's contained
            if not is_contained(center, angle, 10.0):  # Large initial radius
                valid = False
                break
            # Check overlaps with existing hexagons (using grid)
            grid.add_hexagon(i, center, angle)
        
        if not valid:
            continue
            
        # Local optimization loop
        for iteration in range(200):
            # Build grid for current configuration
            grid.clear()
            for i in range(11):
                center = (hex_data[i][0], hex_data[i][1])
                angle = hex_data[i][2]
                grid.add_hexagon(i, center, angle)
            
            # Perturb position of each hexagon
            new_hex_data = hex_data.copy()
            improved = False
            
            for i in range(11):
                current_center = (new_hex_data[i][0], new_hex_data[i][1])
                current_angle = new_hex_data[i][2]
                
                # Try small perturbations
                best_center = current_center
                best_angle = current_angle
                best_radius = calculate_outer_radius(new_hex_data)
                best_valid = True
                
                # Try moving in various directions
                steps = [(0.05, 0), (0, 0.05), (-0.05, 0), (0, -0.05)]
                for dx, dy in steps:
                    test_center = (current_center[0] + dx, current_center[1] + dy)
                    # Test if this move keeps the hexagon contained and doesn't cause overlaps
                    temp_data = new_hex_data.copy()
                    temp_data[i][0] = test_center[0]
                    temp_data[i][1] = test_center[1]
                    
                    # Check overlaps with all others
                    collision_free = True
                    candidates = grid.query_potential_collisions(i, test_center, current_angle)
                    for j in candidates:
                        if i != j and do_overlap(test_center, current_angle, 
                                               (temp_data[j][0], temp_data[j][1]), temp_data[j][2]):
                            collision_free = False
                            break
                    
                    if collision_free:
                        # Check if it's still contained
                        if is_contained(test_center, current_angle, 10.0):
                            new_radius = calculate_outer_radius(temp_data)
                            if new_radius < best_radius:
                                best_radius = new_radius
                                best_center = test_center
                                best_valid = True
                                improved = True
                        
                # Apply best change
                if best_valid and best_center != current_center:
                    new_hex_data[i][0] = best_center[0]
                    new_hex_data[i][1] = best_center[1]
            
            # If no improvement was made, stop
            if not improved:
                break
                
            hex_data = new_hex_data
            
        # Evaluate final solution
        final_radius = calculate_outer_radius(hex_data)
        inv_radius = 1.0 / final_radius
        
        if inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_solution = hex_data.copy()
    
    # Final refinement with more systematic search
    if best_solution is not None:
        # Try rotating some hexagons to see if we can improve further
        temp_solution = best_solution.copy()
        grid = CollisionGrid()
        for i in range(11):
            center = (temp_solution[i][0], temp_solution[i][1])
            angle = temp_solution[i][2]
            grid.add_hexagon(i, center, angle)
        
        # Try various rotations
        rotations_to_try = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        for i in range(11):
            best_angle = temp_solution[i][2]
            best_radius = calculate_outer_radius(temp_solution)
            
            for rot in rotations_to_try:
                temp_data = temp_solution.copy()
                temp_data[i][2] = rot
                
                # Check collisions and containment
                valid = True
                # Rebuild grid
                temp_grid = CollisionGrid()
                for j in range(11):
                    center = (temp_data[j][0], temp_data[j][1])
                    angle = temp_data[j][2]
                    temp_grid.add_hexagon(j, center, angle)
                
                # Check all pairs for overlap
                for j in range(11):
                    for k in range(j+1, 11):
                        if do_overlap((temp_data[j][0], temp_data[j][1]), temp_data[j][2],
                                    (temp_data[k][0], temp_data[k][1]), temp_data[k][2]):
                            valid = False
                            break
                    if not valid:
                        break
                
                # Check containment
                if valid:
                    for j in range(11):
                        if not is_contained((temp_data[j][0], temp_data[j][1]), temp_data[j][2], 10.0):
                            valid = False
                            break
                
                if valid:
                    new_radius = calculate_outer_radius(temp_data)
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_angle = rot
            
            temp_solution[i][2] = best_angle
            
        best_solution = temp_solution
    
    # Generate final result
    # We want to output the best solution found
    inner_hex_data = best_solution if best_solution is not None else np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0]
    ])
    
    # Calculate the actual outer hexagon size
    if best_solution is not None:
        outer_radius = calculate_outer_radius(best_solution)
        outer_hex_side_length = outer_radius
    else:
        outer_hex_side_length = 10.0
    
    # Return the centered outer hexagon
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
