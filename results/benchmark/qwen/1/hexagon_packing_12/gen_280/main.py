# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import time
from numba import jit, njit
import math
from collections import defaultdict

@njit
def create_hexagon_vertices_numba(center_x, center_y, side_length, rotation_degrees):
    """Create vertices of a regular hexagon using numba for speed."""
    angle_rad = np.radians(rotation_degrees)
    angle_step = 2 * np.pi / 6
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_step * i + angle_rad
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices[i] = (x, y)
    return vertices

@njit
def get_hexagon_circumradius(side_length):
    """Get the circumradius of a regular hexagon."""
    return side_length

@njit
def fast_distance_point_to_point(x1, y1, x2, y2):
    """Fast Euclidean distance calculation."""
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)

@njit
def point_in_hexagon_fast(point_x, point_y, hex_center_x, hex_center_y, rotation, side_length):
    """Fast check if a point is inside a regular hexagon."""
    # Transform point to hexagon's coordinate system
    cos_rot = np.cos(np.radians(rotation))
    sin_rot = np.sin(np.radians(rotation))
    dx = point_x - hex_center_x
    dy = point_y - hex_center_y
    rot_x = dx * cos_rot + dy * sin_rot
    rot_y = -dx * sin_rot + dy * cos_rot

    # Distance from center to edge in x and y directions
    edge_distance_x = side_length * np.sqrt(3) / 2
    edge_distance_y = side_length * 0.5

    # Check if point is within bounds
    return (abs(rot_x) <= edge_distance_x and 
            abs(rot_y) <= edge_distance_y and 
            abs(rot_x) + abs(rot_y) <= side_length * np.sqrt(3))

class QuadTreeNode:
    """Quadtree node for spatial acceleration."""
    def __init__(self, bounds, capacity=4):
        self.bounds = bounds  # [min_x, min_y, max_x, max_y]
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.children = [None, None, None, None]  # NW, NE, SW, SE
        
    def subdivide(self):
        """Divide the node into four quadrants."""
        min_x, min_y, max_x, max_y = self.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        # Create children bounds
        self.children[0] = QuadTreeNode([min_x, mid_y, mid_x, max_y], self.capacity)  # NW
        self.children[1] = QuadTreeNode([mid_x, mid_y, max_x, max_y], self.capacity)  # NE
        self.children[2] = QuadTreeNode([min_x, min_y, mid_x, mid_y], self.capacity)  # SW
        self.children[3] = QuadTreeNode([mid_x, min_y, max_x, mid_y], self.capacity)  # SE
        
        self.divided = True
        
        # Redistribute points
        for point in self.points:
            self.insert_point(point)
        self.points = []
        
    def insert_point(self, point):
        """Insert a point into the quadtree."""
        if not self.in_bounds(point):
            return False
            
        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True
            
        if not self.divided:
            self.subdivide()
            
        for child in self.children:
            if child.insert_point(point):
                return True
        return False
        
    def in_bounds(self, point):
        """Check if point is within node bounds."""
        x, y = point
        min_x, min_y, max_x, max_y = self.bounds
        return min_x <= x <= max_x and min_y <= y <= max_y
        
    def query_range(self, range_bounds):
        """Query points within a range."""
        result = []
        min_x, min_y, max_x, max_y = range_bounds
        
        if not self.intersects_range(range_bounds):
            return result
            
        for point in self.points:
            x, y = point
            if min_x <= x <= max_x and min_y <= y <= max_y:
                result.append(point)
                
        if self.divided:
            for child in self.children:
                result.extend(child.query_range(range_bounds))
                
        return result
        
    def intersects_range(self, range_bounds):
        """Check if node bounds intersect with query range."""
        min_x, min_y, max_x, max_y = self.bounds
        r_min_x, r_min_y, r_max_x, r_max_y = range_bounds
        return not (r_max_x < min_x or r_min_x > max_x or r_max_y < min_y or r_min_y > max_y)
        
    def query_neighbors(self, point, radius):
        """Query all points within radius of a given point."""
        # Create a query range around the point
        q_min_x, q_min_y = point[0] - radius, point[1] - radius
        q_max_x, q_max_y = point[0] + radius, point[1] + radius
        points_in_range = self.query_range([q_min_x, q_min_y, q_max_x, q_max_y])
        
        # Filter by actual distance
        neighbors = []
        for p in points_in_range:
            if fast_distance_point_to_point(point[0], point[1], p[0], p[1]) <= radius:
                neighbors.append(p)
        return neighbors

class HexagonPacker:
    """Handles the core packing logic with advanced spatial acceleration."""
    
    def __init__(self):
        self.hexagons = []
        self.outer_hex = None
        self.quadtree = None
        
    def add_hexagon(self, center_x, center_y, rotation_deg, side_length=1):
        """Add a hexagon to the packer."""
        self.hexagons.append({
            'center_x': center_x,
            'center_y': center_y,
            'rotation_deg': rotation_deg,
            'side_length': side_length,
            'vertices': create_hexagon_vertices_numba(center_x, center_y, side_length, rotation_deg)
        })
        
    def set_outer_hexagon(self, center_x, center_y, side_length):
        """Set the outer hexagon constraints."""
        self.outer_hex = {
            'center_x': center_x,
            'center_y': center_y,
            'side_length': side_length,
            'vertices': create_hexagon_vertices_numba(center_x, center_y, side_length, 0)
        }
        
    def update_quadtree(self):
        """Update the spatial acceleration structure."""
        if not self.hexagons:
            return
            
        # Determine bounds for quadtree
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        centers = []
        
        for hex_data in self.hexagons:
            centers.append((hex_data['center_x'], hex_data['center_y']))
            min_x = min(min_x, hex_data['center_x'])
            max_x = max(max_x, hex_data['center_x'])
            min_y = min(min_y, hex_data['center_y'])
            max_y = max(max_y, hex_data['center_y'])
            
        # Add some padding
        padding = 1.0
        bounds = [min_x - padding, min_y - padding, max_x + padding, max_y + padding]
        
        self.quadtree = QuadTreeNode(bounds, capacity=4)
        for center in centers:
            self.quadtree.insert_point(center)
    
    def validate_containment(self):
        """Check if all inner hexagons are fully contained within outer hexagon."""
        if not self.outer_hex:
            return False
            
        outer_polygon = Polygon(self.outer_hex['vertices'])
        
        for hex_data in self.hexagons:
            # Check all vertices of inner hexagon
            for vertex in hex_data['vertices']:
                point = Point(vertex)
                if not outer_polygon.contains(point):
                    return False
        return True
    
    def get_overlapping_pairs(self):
        """Find overlapping pairs using quadtree acceleration."""
        if not self.hexagons:
            return []
            
        overlapping_pairs = []
        # Use quadtree to quickly find nearby hexagons
        self.update_quadtree()
        
        # Check each hexagon against nearby candidates
        for i, hex1_data in enumerate(self.hexagons):
            # Get nearby hexagons within 2 unit lengths (potential overlap distance)
            nearby_centers = self.quadtree.query_neighbors(
                (hex1_data['center_x'], hex1_data['center_y']), 
                2.0
            )
            
            for j, (nearby_x, nearby_y) in enumerate(nearby_centers):
                if i >= j:  # Avoid double counting and self-checking
                    continue
                    
                hex2_data = self.hexagons[j]
                # Fast preliminary check
                dist_centers = fast_distance_point_to_point(
                    hex1_data['center_x'], hex1_data['center_y'],
                    hex2_data['center_x'], hex2_data['center_y']
                )
                
                if dist_centers < 2.0:  # Close enough to potentially overlap
                    # Full polygon intersection check
                    hex1_poly = Polygon(hex1_data['vertices'])
                    hex2_poly = Polygon(hex2_data['vertices'])
                    
                    if hex1_poly.intersects(hex2_poly):
                        overlapping_pairs.append((i, j))
                        
        return overlapping_pairs
    
    def get_outer_hex_side_length(self):
        """Estimate the minimum required outer hexagon side length."""
        if not self.hexagons:
            return 100.0
            
        max_dist = 0.0
        for hex_data in self.hexagons:
            center_x, center_y = hex_data['center_x'], hex_data['center_y']
            dist = np.sqrt(center_x**2 + center_y**2)
            # Add the circumradius of inner hexagon (1 for unit hexagon)
            dist_to_edge = dist + 1.0
            max_dist = max(max_dist, dist_to_edge)
        
        return max_dist * 2.0  # Diameter gives us the side length for a hexagon
    
    def get_all_vertices(self):
        """Get all vertices for plotting/analysis."""
        all_vertices = []
        for hex_data in self.hexagons:
            all_vertices.extend(hex_data['vertices'].tolist())
        return all_vertices

class EvolutionaryOptimizer:
    """Advanced evolutionary optimizer with multi-objective fitness."""
    
    def __init__(self):
        self.population_size = 30
        self.max_generations = 50
        self.mutation_rate = 0.1
        
    def evaluate_fitness(self, individual):
        """
        Multi-objective fitness evaluation.
        Returns: (combined_score, detailed_metrics)
        """
        # Decode individual
        positions_angles = individual.reshape(-1, 3)
        
        # Create packer instance
        packer = HexagonPacker()
        
        # Add inner hexagons
        for i in range(12):
            x, y, angle = positions_angles[i]
            packer.add_hexagon(x, y, angle, 1.0)
        
        # Estimate outer hexagon size
        estimated_side_length = packer.get_outer_hex_side_length() 
        packer.set_outer_hexagon(0, 0, estimated_side_length)
        
        # Check constraints
        containment_ok = packer.validate_containment()
        overlap_pairs = packer.get_overlapping_pairs()
        
        # Calculate fitness components
        if not containment_ok:
            return 1e10, {"containment": False, "overlap_count": len(overlap_pairs)}
            
        if len(overlap_pairs) > 0:
            return 1e10, {"containment": True, "overlap_count": len(overlap_pairs)}
            
        # If valid, return negative inverse side length (better = higher fitness)
        inv_side_length = 1.0 / estimated_side_length
        return -inv_side_length, {"containment": True, "overlap_count": 0, "side_length": estimated_side_length}
    
    def create_individual(self):
        """Create a random valid individual."""
        # Start with symmetric configuration
        positions_angles = []
        
        # Central hexagon
        positions_angles.append([0.0, 0.0, 0.0])
        
        # First ring (6 hexagons)
        for i in range(6):
            angle = i * np.pi/3
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions_angles.append([x, y, 0.0])
        
        # Second ring (6 hexagons)
        for i in range(6):
            angle = i * np.pi/3 + np.pi/6  # Offset by π/6
            x = 3.0 * np.cos(angle)
            y = 3.0 * np.sin(angle)
            positions_angles.append([x, y, 0.0])
            
        # Add randomness for variation
        individual = np.array(positions_angles)
        np.random.seed(int(time.time()) % 10000)
        individual[:, 0] += np.random.normal(0, 0.2, 12)
        individual[:, 1] += np.random.normal(0, 0.2, 12)
        
        return individual.flatten()
    
    def mutate(self, individual, generation):
        """Smart mutation that adapts based on generation."""
        # Decrease mutation rate over time
        adapted_mutation_rate = self.mutation_rate * (1.0 - generation / self.max_generations)
        
        mutated = individual.copy()
        
        # Mutate positions and rotations
        for i in range(0, len(mutated), 3):  # For each hexagon
            if np.random.random() < adapted_mutation_rate:
                mutated[i] += np.random.normal(0, 0.3)  # x position
            if np.random.random() < adapted_mutation_rate:
                mutated[i+1] += np.random.normal(0, 0.3)  # y position
            if np.random.random() < adapted_mutation_rate:
                mutated[i+2] = (mutated[i+2] + np.random.normal(0, 15)) % 360  # angle
                
        # Keep positions within reasonable bounds
        mutated[::3] = np.clip(mutated[::3], -10, 10)
        mutated[1::3] = np.clip(mutated[1::3], -10, 10)
        
        return mutated
    
    def crossover(self, parent1, parent2):
        """Uniform crossover with preference for good genes."""
        child = parent1.copy()
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child[i] = parent2[i]
        return child
    
    def optimize(self):
        """Run the evolutionary optimization."""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_fitness_history = []
        
        # Evolutionary loop
        for generation in range(self.max_generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, metrics = self.evaluate_fitness(individual)
                fitness_scores.append((fitness, individual, metrics))
            
            # Sort by fitness (ascending, since we're minimizing negative values)
            fitness_scores.sort(key=lambda x: x[0])
            
            # Track best
            best_fitness = fitness_scores[0][0]
            best_fitness_history.append(best_fitness)
            
            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {-best_fitness}")
            
            # Early termination if we're converging
            if len(best_fitness_history) > 5:
                recent_avg = np.mean(best_fitness_history[-5:])
                if abs(recent_avg - best_fitness) < 1e-6:
                    break
            
            # Selection: keep top 30% (elitism)
            elite_count = int(self.population_size * 0.3)
            elites = [ind for _, ind, _ in fitness_scores[:elite_count]]
            
            # Create new population
            new_population = elites.copy()
            
            # Fill rest with offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self.tournament_selection(fitness_scores)
                parent2 = self.tournament_selection(fitness_scores)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child, generation)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best solution
        final_fitness, best_individual, metrics = fitness_scores[0]
        return best_individual, -final_fitness, metrics
    
    def tournament_selection(self, fitness_scores, tournament_size=3):
        """Select an individual using tournament selection."""
        tournament = np.random.choice(fitness_scores, tournament_size)
        winner = min(tournament, key=lambda x: x[0])  # Minimize fitness
        return winner[1]  # Return individual

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses an evolutionary optimization with advanced spatial acceleration and multi-objective fitness.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Initialize optimizer
    optimizer = EvolutionaryOptimizer()
    
    try:
        # Run optimization
        best_individual, side_length, metrics = optimizer.optimize()
        
        # Extract hexagon data
        positions_angles = best_individual.reshape(-1, 3)
        inner_hex_data = positions_angles.copy()
        
        # Outer hex is centered at origin
        outer_hex_data = np.array([0, 0, 0])
        
        return inner_hex_data, outer_hex_data, side_length
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        pass
    
    # Fallback to a reasonably good configuration
    inner_hex_data = np.array([
        [0, 0, 0],  # center
        [0, 2, 0],  # top
        [0, -2, 0],  # bottom
        [1.732, 1, 0],  # top right
        [-1.732, 1, 0],  # top left
        [1.732, -1, 0],  # bottom right
        [-1.732, -1, 0],  # bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0],  # far left
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0],  # top far left
        [1.732, -3, 0],  # bottom far right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 6.928  # approximated value

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
