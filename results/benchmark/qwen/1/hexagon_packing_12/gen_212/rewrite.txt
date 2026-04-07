# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import time
from math import sqrt
import heapq

class SpatialHexagonPacker:
    """
    A spatially-accelerated hexagon packing algorithm using hierarchical spatial indexing
    and targeted geometric optimization.
    """
    
    def __init__(self):
        self.unit_hex_radius = 1.0
        self.unit_hex_diameter = 2.0
        self.max_search_radius = 3.0  # Maximum distance to consider for overlap
        
    def create_hexagon_vertices(self, center, side_length, rotation_degrees):
        """Create vertices of a regular hexagon."""
        angle_rad = np.radians(rotation_degrees)
        angle_step = 2 * np.pi / 6
        vertices = []
        for i in range(6):
            angle = angle_step * i + angle_rad
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
    
    def get_hexagon_circumradius(self, side_length):
        """Get the circumradius of a regular hexagon."""
        return side_length
    
    def calculate_outer_hex_side_length(self, inner_hex_data, outer_hex_center=(0,0)):
        """Calculate the minimum required outer hexagon side length."""
        if len(inner_hex_data) == 0:
            return 100.0

        max_distance = 0.0
        for i in range(len(inner_hex_data)):
            cx, cy, _ = inner_hex_data[i]
            # Calculate distance from center of outer hexagon to center of inner hexagon
            distance = np.sqrt((cx - outer_hex_center[0])**2 + (cy - outer_hex_center[1])**2)
            # Add radius of inner hexagon
            distance_to_outer_edge = distance + self.get_hexagon_circumradius(1.0)
            max_distance = max(max_distance, distance_to_outer_edge)

        # For a regular hexagon, the side length equals the radius
        return max_distance * 2.0
    
    def check_containment_all_vertices(self, hex_vertices, outer_hex_center, outer_hex_side_length):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        outer_vertices = self.create_hexagon_vertices(outer_hex_center, outer_hex_side_length, 0)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hex_vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                return False
        return True
    
    def check_overlap_pair(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        hex1_polygon = Polygon(hex1_vertices)
        hex2_polygon = Polygon(hex2_vertices)
        return hex1_polygon.intersects(hex2_polygon)
    
    def compute_distance_between_centers(self, center1, center2):
        """Fast Euclidean distance calculation between two centers."""
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        return sqrt(dx*dx + dy*dy)
    
    def compute_hexagon_bounding_circle_radius(self, center, vertices):
        """Compute the radius of the bounding circle for a hexagon."""
        max_dist_sq = 0.0
        for vertex in vertices:
            dx = vertex[0] - center[0]
            dy = vertex[1] - center[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
        return sqrt(max_dist_sq)
    
    def build_bvh_structure(self, centers, max_depth=4, max_items_per_node=4):
        """Build a BVH-like structure for spatial indexing."""
        if len(centers) == 0:
            return None
            
        # Create a simple hierarchical structure
        # For this problem, we'll use a fixed-depth octree approach
        class BVHNode:
            def __init__(self, indices=None, bounds=None):
                self.indices = indices or []
                self.bounds = bounds
                self.children = []
                self.is_leaf = True
                
        # Simple spatial partitioning
        def create_partition(remaining_indices, depth):
            if depth >= max_depth or len(remaining_indices) <= max_items_per_node:
                node = BVHNode(remaining_indices)
                if remaining_indices:
                    # Compute bounds
                    coords = np.array([centers[i] for i in remaining_indices])
                    min_x, min_y = np.min(coords, axis=0)
                    max_x, max_y = np.max(coords, axis=0)
                    node.bounds = (min_x, min_y, max_x, max_y)
                return node
            
            # Partition by median split
            coords = np.array([centers[i] for i in remaining_indices])
            median_x = np.median(coords[:, 0])
            median_y = np.median(coords[:, 1])
            
            left_indices = [i for i in remaining_indices if centers[i][0] <= median_x]
            right_indices = [i for i in remaining_indices if centers[i][0] > median_x]
            
            if len(left_indices) == len(remaining_indices) or len(right_indices) == len(remaining_indices):
                # If splitting doesn't work well, try vertical split
                median_y_split = np.median(coords[:, 1])
                left_indices = [i for i in remaining_indices if centers[i][1] <= median_y_split]
                right_indices = [i for i in remaining_indices if centers[i][1] > median_y_split]
                
            node = BVHNode()
            node.is_leaf = False
            
            if left_indices:
                node.children.append(create_partition(left_indices, depth + 1))
            if right_indices:
                node.children.append(create_partition(right_indices, depth + 1))
                
            # Compute bounds for this node
            if remaining_indices:
                coords = np.array([centers[i] for i in remaining_indices])
                min_x, min_y = np.min(coords, axis=0)
                max_x, max_y = np.max(coords, axis=0)
                node.bounds = (min_x, min_y, max_x, max_y)
                
            return node
            
        return create_partition(list(range(len(centers))), 0)
    
    def query_potential_overlaps(self, bvh_root, centers):
        """Query potentially overlapping pairs using BVH."""
        if not bvh_root:
            return []
            
        pairs = set()
        
        def traverse(node):
            if not node:
                return
                
            if node.is_leaf:
                # Check all pairs among indices in this leaf
                indices = node.indices
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pairs.add((indices[i], indices[j]))
            else:
                # Check pairs across children
                if len(node.children) >= 2:
                    # Cross check between all children
                    for i in range(len(node.children)):
                        for j in range(i + 1, len(node.children)):
                            child1 = node.children[i]
                            child2 = node.children[j]
                            
                            if child1.bounds and child2.bounds:
                                # Check if bounds overlap
                                bx1, by1, bx2, by2 = child1.bounds
                                cx1, cy1, cx2, cy2 = child2.bounds
                                
                                if (bx2 >= cx1 and bx1 <= cx2 and 
                                    by2 >= cy1 and by1 <= cy2):
                                    # Cross-check with actual indices
                                    for ci in child1.indices:
                                        for cj in child2.indices:
                                            if ci < cj:  # Avoid duplicates
                                                pairs.add((ci, cj))
                
                # Recurse into children
                for child in node.children:
                    traverse(child)
                    
        traverse(bvh_root)
        return list(pairs)
    
    def estimate_best_inner_positions(self):
        """Generate an initial configuration based on mathematical insights."""
        # Create a hexagonal packing pattern inspired by optimal arrangements
        positions = []
        
        # Central hexagon
        positions.append([0, 0, 0])
        
        # First ring - 6 hexagons at distance 2
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions
        radius = 2.0
        
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append([x, y, 0])
        
        # Second ring - 6 hexagons at distance 3.2 
        angles2 = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions
        radius2 = 3.2
        
        for angle in angles2:
            x = radius2 * np.cos(angle)
            y = radius2 * np.sin(angle)
            positions.append([x, y, 0])
            
        # Trim to exactly 12
        positions = positions[:12]
        
        # Add small noise to break perfect symmetry
        np.random.seed(42)
        for i in range(12):
            positions[i][0] += np.random.normal(0, 0.05)
            positions[i][1] += np.random.normal(0, 0.05)
            
        return np.array(positions)
    
    def evaluate_configuration(self, inner_hex_data, outer_hex_center=(0,0)):
        """Fast evaluation using spatial acceleration."""
        if len(inner_hex_data) != 12:
            return 1e-10

        # Precompute all hexagon vertices
        hex_vertices_list = []
        for i in range(len(inner_hex_data)):
            cx, cy, angle = inner_hex_data[i]
            vertices = self.create_hexagon_vertices((cx, cy), 1.0, angle)
            hex_vertices_list.append(vertices)

        # Check containment: all hexagon vertices must be within outer hexagon
        outer_side_length = self.calculate_outer_hex_side_length(inner_hex_data, outer_hex_center)
        outer_vertices = self.create_hexagon_vertices(outer_hex_center, outer_side_length, 0)
        outer_polygon = Polygon(outer_vertices)

        # Check containment for all vertices
        for vertices in hex_vertices_list:
            for vertex in vertices:
                point = Point(vertex)
                if not outer_polygon.contains(point):
                    return 1e-10

        # Build spatial index for overlap checking
        centers = [(float(data[0]), float(data[1])) for data in inner_hex_data]
        
        # Build BVH structure
        bvh_root = self.build_bvh_structure(centers)
        
        # Query potentially overlapping pairs 
        potential_pairs = self.query_potential_overlaps(bvh_root, centers)
        
        # Check actual overlaps only for potentially overlapping pairs
        for i, j in potential_pairs:
            if i < j:
                if self.check_overlap_pair(hex_vertices_list[i], hex_vertices_list[j]):
                    return 1e-10

        # If we reach here, the configuration is valid
        return 1.0 / outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    start_time = time.time()
    
    # Create packer instance
    packer = SpatialHexagonPacker()
    
    # Generate initial promising configuration
    initial_guess = packer.estimate_best_inner_positions()
    
    # Simple local optimization approach: gradient-free local search
    best_config = initial_guess.copy()
    best_score = packer.evaluate_configuration(best_config)
    best_outer_side = 1.0 / best_score if best_score > 1e-5 else float('inf')
    
    # Local search with simulated annealing-inspired approach
    temperature = 1.0
    cooling_rate = 0.99
    max_iterations = 2000
    
    for iteration in range(max_iterations):
        # Create neighbor configuration
        new_config = best_config.copy()
        
        # Select random hexagon to modify
        hex_idx = np.random.randint(0, 12)
        
        # Perturb position slightly
        new_config[hex_idx, 0] += np.random.normal(0, 0.1)
        new_config[hex_idx, 1] += np.random.normal(0, 0.1)
        
        # Randomly change angle (only sometimes)
        if np.random.random() < 0.3:
            new_config[hex_idx, 2] = np.random.uniform(0, 360)
        
        # Evaluate new configuration
        score = packer.evaluate_configuration(new_config)
        
        if score > best_score and score > 1e-5:
            # Accept better solution
            best_score = score
            best_config = new_config.copy()
            best_outer_side = 1.0 / best_score
            
            # Early termination if we're approaching SOTA
            if best_score > 0.2535:  # Close to target
                break
        
        # Cool down temperature
        temperature *= cooling_rate
        
        # Check time limit
        if time.time() - start_time > 175:  # Leave some margin
            break
    
    # Final validation
    final_score = packer.evaluate_configuration(best_config)
    
    if final_score > 1e-5:
        outer_side_length = 1.0 / final_score
        outer_hex_data = np.array([0, 0, 0])  # Centered at origin
        return best_config, outer_hex_data, outer_side_length
    
    # Fallback to a known good configuration
    inner_hex_data = np.array([
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom
        [1.732, 1, 0],  # top right
        [-1.732, 1, 0], # top left
        [1.732, -1, 0], # bottom right
        [-1.732, -1, 0],# bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0], # top far left
        [1.732, -3, 0], # bottom far right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 6.928  # approximated value
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END