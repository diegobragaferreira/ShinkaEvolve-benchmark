# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import time
from math import sqrt
from joblib import Parallel, delayed
from collections import defaultdict
import random

class HexagonPackagingEngine:
    def __init__(self):
        self.hex_side_length = 1.0
        self.circumradius = self.hex_side_length
        self.inradius = self.hex_side_length * sqrt(3) / 2
        self.max_search_radius = 10.0
        
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
    
    def get_outer_hex_side_from_config(self, inner_hex_data, center=(0,0)):
        """Compute the minimum required outer hexagon side length from current configuration."""
        if len(inner_hex_data) == 0:
            return 100

        max_dist = 0
        for i in range(len(inner_hex_data)):
            cx, cy, _ = inner_hex_data[i]
            dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            # Add the circumradius of inner hexagon (1 for unit hexagon)
            dist_to_edge = dist + self.circumradius
            max_dist = max(max_dist, dist_to_edge)

        return max_dist * 2.0  # Diameter gives us the side length for a hexagon

    def check_containment_all_vertices(self, hex_vertices, outer_hex_center, outer_hex_side_length):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        outer_vertices = self.create_hexagon_vertices(outer_hex_center, outer_hex_side_length, 0)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hex_vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                return False
        return True

    def fast_check_overlap_pair(self, hex1_vertices, hex2_vertices):
        """Fast overlap check with approximate bounding circle test first."""
        # Quick bounding circle test
        hex1_center = np.mean(hex1_vertices, axis=0)
        hex2_center = np.mean(hex2_vertices, axis=0)

        # Get approximate distances from centers
        dist_centers = np.linalg.norm(hex1_center - hex2_center)

        # Circumradii of unit hexagons
        circumradius = self.circumradius

        # If centers are too far apart, no overlap
        if dist_centers > 2 * circumradius:
            return False

        # Full polygon intersection test
        hex1_polygon = Polygon(hex1_vertices)
        hex2_polygon = Polygon(hex2_vertices)
        return hex1_polygon.intersects(hex2_polygon)

    def evaluate_individual(self, individual, outer_hex_center=(0,0)):
        """Evaluate a single individual configuration."""
        # Decode individual to hexagon positions and rotations
        # Format: [x1, y1, angle1, x2, y2, angle2, ..., x12, y12, angle12]
        hex_data = individual.reshape(-1, 3)
        
        if len(hex_data) != 12:
            return 1e-10

        # Create all hexagon polygons
        hex_polygons = []
        for i in range(len(hex_data)):
            cx, cy, angle = hex_data[i]
            vertices = self.create_hexagon_vertices((cx, cy), self.hex_side_length, angle)
            hex_polygons.append(Polygon(vertices))

        # Check containment: all hexagon vertices must be within outer hexagon
        outer_side_length = self.get_outer_hex_side_from_config(hex_data, outer_hex_center)
        outer_vertices = self.create_hexagon_vertices(outer_hex_center, outer_side_length, 0)
        outer_polygon = Polygon(outer_vertices)

        # Check containment for all vertices
        for i in range(len(hex_data)):
            vertices = hex_polygons[i].exterior.coords[:-1]  # Exclude last point (duplicate of first)
            for vertex in vertices:
                point = Point(vertex)
                if not outer_polygon.contains(point):
                    return 1e-10

        # Check overlaps between all pairs of hexagons
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return 1e-10

        # If we reach here, the configuration is valid
        return 1.0 / outer_side_length

    def initialize_population(self, pop_size, symmetry_preserving=True):
        """Initialize a diverse population with symmetry awareness."""
        population = []
        
        # Generate base symmetric configurations
        for _ in range(pop_size // 3):
            # Fully symmetric configuration
            positions = []
            positions.append([0, 0, 0])  # Center
            
            # First ring - 6 hexagons
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            radius = 2.0
            
            for angle in angles:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append([x, y, 0])
            
            # Second ring - 5 hexagons
            angles2 = np.linspace(0, 2*np.pi, 6)[:-1]
            radius2 = 3.5
            
            for angle in angles2:
                x = radius2 * np.cos(angle)
                y = radius2 * np.sin(angle)
                positions.append([x, y, 0])
                
            # Adjust to exactly 12
            positions = positions[:12]
            
            # Convert to array and add small perturbations
            config = np.array(positions)
            if symmetry_preserving:
                config[:, 0] += np.random.normal(0, 0.1, 12)
                config[:, 1] += np.random.normal(0, 0.1, 12)
            
            population.append(config.flatten())
            
        # Generate random configurations for diversity
        for _ in range(pop_size - len(population)):
            config = np.zeros(36)  # 12 hexagons * 3 params each
            # Position bounds
            config[::3] = np.random.uniform(-self.max_search_radius, self.max_search_radius, 12)  # x coords
            config[1::3] = np.random.uniform(-self.max_search_radius, self.max_search_radius, 12)  # y coords
            config[2::3] = np.random.uniform(0, 360, 12)  # angles
            population.append(config)
            
        return population

    def mutate(self, individual, mutation_rate=0.1):
        """Apply mutation to an individual, preserving structural constraints."""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                if i % 3 == 0:  # x coordinate
                    mutated[i] += np.random.normal(0, 0.3)
                    mutated[i] = np.clip(mutated[i], -self.max_search_radius, self.max_search_radius)
                elif i % 3 == 1:  # y coordinate
                    mutated[i] += np.random.normal(0, 0.3)
                    mutated[i] = np.clip(mutated[i], -self.max_search_radius, self.max_search_radius)
                else:  # angle
                    mutated[i] += np.random.normal(0, 30)
                    mutated[i] = mutated[i] % 360
                    
        return mutated

    def crossover(self, parent1, parent2, crossover_rate=0.8):
        """Perform crossover between two parents."""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
            
        # Simple uniform crossover
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover for chromosome segments
        crossover_point = np.random.randint(0, len(parent1))
        child1[crossover_point:] = parent2[crossover_point:]
        child2[crossover_point:] = parent1[crossover_point:]
        
        return child1, child2

class OctreeSpatialIndex:
    """Hierarchical spatial acceleration structure."""
    
    def __init__(self, max_depth=5, max_points_per_node=4):
        self.max_depth = max_depth
        self.max_points_per_node = max_points_per_node
        self.root = None
        
    def build(self, hex_centers):
        """Build octree structure from hex centers."""
        if len(hex_centers) == 0:
            return
            
        # Determine bounding box
        mins = np.min(hex_centers, axis=0)
        maxs = np.max(hex_centers, axis=0)
        self.root = self._build_recursive(hex_centers, mins, maxs, 0)
        
    def _build_recursive(self, points, min_coords, max_coords, depth):
        """Recursively build octree node."""
        node = {
            'points': points,
            'min': min_coords,
            'max': max_coords,
            'depth': depth,
            'children': [],
            'is_leaf': True
        }
        
        if len(points) <= self.max_points_per_node or depth >= self.max_depth:
            return node
            
        # Split into 8 octants
        mid = (min_coords + max_coords) / 2
        children_points = [[] for _ in range(8)]
        
        for point in points:
            # Determine which octant this point belongs to
            octant = 0
            if point[0] > mid[0]: octant |= 1
            if point[1] > mid[1]: octant |= 2
            if point[2] > mid[2]: octant |= 4
            children_points[octant].append(point)
            
        # Build children recursively
        node['is_leaf'] = False
        for i, pts in enumerate(children_points):
            if len(pts) > 0:
                # Determine child bounds
                child_min = min_coords.copy()
                child_max = max_coords.copy()
                
                if i & 1: child_min[0] = mid[0]
                else: child_max[0] = mid[0]
                if i & 2: child_min[1] = mid[1]
                else: child_max[1] = mid[1]
                if i & 4: child_min[2] = mid[2]
                else: child_max[2] = mid[2]
                
                child_node = self._build_recursive(np.array(pts), child_min, child_max, depth + 1)
                node['children'].append(child_node)
                
        return node

class EvolutionaryHexPackager:
    def __init__(self):
        self.engine = HexagonPackagingEngine()
        self.pop_size = 50
        self.num_generations = 100
        self.elitism_rate = 0.1
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.max_evaluations = 5000
        
    def optimize(self, max_time_seconds=170):
        """Main evolutionary optimization loop."""
        start_time = time.time()
        
        # Initialize population
        population = self.engine.initialize_population(self.pop_size)
        fitness_history = []
        
        evaluation_count = 0
        
        for generation in range(self.num_generations):
            if time.time() - start_time > max_time_seconds:
                break
                
            # Evaluate population
            fitness_scores = []
            for i, individual in enumerate(population):
                score = self.engine.evaluate_individual(individual)
                fitness_scores.append(score)
                evaluation_count += 1
                
                # Early stopping if we've found a good solution
                if score > 0.2535:  # Close to target
                    return self.reconstruct_solution(individual), score
                
            # Track best
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            fitness_history.append(best_fitness)
            
            # Elitism
            elite_count = int(self.elitism_rate * self.pop_size)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_individuals = [population[i] for i in sorted_indices[:elite_count]]
            
            # Create new population
            new_population = elite_individuals.copy()
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                if time.time() - start_time > max_time_seconds:
                    break
                    
                # Tournament selection
                parent1_idx = self.tournament_selection(population, fitness_scores, 3)
                parent2_idx = self.tournament_selection(population, fitness_scores, 3)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                child1, child2 = self.engine.crossover(parent1, parent2, self.crossover_rate)
                child1 = self.engine.mutate(child1, self.mutation_rate)
                child2 = self.engine.mutate(child2, self.mutation_rate)
                
                new_population.extend([child1, child2])
                
            population = new_population[:self.pop_size]
            
            # Local refinement for best individual occasionally
            if generation % 10 == 0:
                best_individual = population[best_idx]
                refined = self.local_refinement(best_individual)
                refined_score = self.engine.evaluate_individual(refined)
                if refined_score > best_fitness:
                    population[best_idx] = refined
                    
        # Return best solution found
        final_scores = [self.engine.evaluate_individual(ind) for ind in population]
        best_idx = np.argmax(final_scores)
        return self.reconstruct_solution(population[best_idx]), final_scores[best_idx]
        
    def tournament_selection(self, population, fitness_scores, tournament_size):
        """Tournament selection for choosing parents."""
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx
    
    def local_refinement(self, individual):
        """Use local optimization on a given solution."""
        # Convert individual to flat vector for scipy optimization
        def objective(x):
            # Reshape back to hex data
            hex_data = x.reshape(-1, 3)
            # Return negative of the score (since we're minimizing)
            score = self.engine.evaluate_individual(x)
            return -score if score > 1e-5 else 1e10
            
        # Use L-BFGS-B with bounds
        bounds = []
        for _ in range(12):
            bounds.extend([(-self.engine.max_search_radius, self.engine.max_search_radius),
                          (-self.engine.max_search_radius, self.engine.max_search_radius)])
        for _ in range(12):
            bounds.append((0, 360))
            
        try:
            result = minimize(objective, individual, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 50})
            if result.success:
                return result.x
        except:
            pass
        return individual
    
    def reconstruct_solution(self, individual):
        """Reconstruct a clean solution from individual."""
        return individual.reshape(-1, 3)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    try:
        packager = EvolutionaryHexPackager()
        inner_hex_data, best_score = packager.optimize()
        
        if best_score > 1e-5:
            outer_side_length = 1.0 / best_score
            outer_hex_data = np.array([0, 0, 0])  # centered at origin
            return inner_hex_data, outer_hex_data, outer_side_length
            
    except Exception as e:
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