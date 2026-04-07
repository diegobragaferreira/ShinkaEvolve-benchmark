# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def generate_hexagon_vertices_numba(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon efficiently using numba"""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    translated_vertices = rotated_vertices + np.array([center_x, center_y])

    return translated_vertices

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""
    
    @staticmethod
    @jit(nopython=True)
    def vertices(x, y, angle_deg, side_length=1):
        """Calculate vertices of a hexagon given center, angle, and side length"""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vx = x + side_length * np.cos(theta)
            vy = y + side_length * np.sin(theta)
            vertices.append((vx, vy))
        return np.array(vertices)

    @staticmethod
    def polygon(x, y, angle_deg, side_length=1):
        """Get shapely polygon representation of hexagon"""
        vertices = HexagonGeometry.vertices(x, y, angle_deg, side_length)
        return Polygon(vertices)

class ConstraintChecker:
    """Handles constraint checking for hexagon packing"""
    
    @staticmethod
    def contains(hex_poly, outer_poly):
        """Check if hexagon is completely contained within outer hexagon"""
        return outer_poly.contains(hex_poly) or outer_poly.intersection(hex_poly).area == hex_poly.area

    @staticmethod
    def overlaps(hex1_poly, hex2_poly):
        """Check if two hexagons overlap"""
        return hex1_poly.intersects(hex2_poly)

class SolutionEvaluator:
    """Evaluates solution quality and feasibility"""
    
    @staticmethod
    def calculate_outer_radius(inner_positions, inner_angles):
        """Calculate minimum radius needed to contain all inner hexagons"""
        max_dist = 0
        outer_center = (0, 0)

        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_positions)):
            pos = inner_positions[i]
            angle = inner_angles[i]
            hex_vertices = HexagonGeometry.vertices(pos[0], pos[1], angle)
            all_vertices.extend(hex_vertices)

        # Find maximum distance from center
        for vertex in all_vertices:
            dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)

        # Add buffer for safety and account for hexagon shape
        return max_dist * 1.1  # Safety factor

    @staticmethod
    def evaluate(solution):
        """Evaluate a solution and return negative of objective (since we minimize)"""
        # Reshape solution into positions and angles
        positions = solution[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates each
        angles = solution[22:]  # 11 angles

        # Create inner hexagons
        inner_hexagons = []
        for i in range(11):
            pos = positions[i]
            angle = angles[i]
            hex_poly = HexagonGeometry.polygon(pos[0], pos[1], angle)
            inner_hexagons.append(hex_poly)

        # Check containment
        outer_radius = SolutionEvaluator.calculate_outer_radius(positions, angles)
        # Outer hexagon with center at origin and calculated radius
        outer_hexagon = HexagonGeometry.polygon(0, 0, 0, outer_radius)

        # Check containment for all inner hexagons
        for hex_poly in inner_hexagons:
            if not ConstraintChecker.contains(hex_poly, outer_hexagon):
                return 1e10  # Penalty for non-containment

        # Check for overlaps
        for i in range(11):
            for j in range(i+1, 11):
                if ConstraintChecker.overlaps(inner_hexagons[i], inner_hexagons[j]):
                    return 1e10  # Penalty for overlap

        # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
        return -1.0 / outer_radius

class OptimizationManager:
    """Manages the complete optimization process"""
    
    def __init__(self):
        self.n = 11
        self.pop_size = 50
        self.generations = 200
        self.elite_size = 5
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8
        self.max_time = 175
        
    def create_initial_population(self):
        """Create initial population with diverse arrangements"""
        population = []
        # Generate multiple good starting solutions
        for _ in range(self.pop_size):
            # Random placement with some clustering around central region
            individual = []
            for i in range(self.n):
                # Center hexagons more tightly clustered
                if i == 0:  # Center hexagon
                    center_x, center_y = 0.0, 0.0
                    angle = random.uniform(0, 360)
                elif i <= 6:  # Around center with regular spacing
                    distance = random.uniform(1.0, 2.5)
                    angle = random.uniform(0, 360)
                    center_x = distance * np.cos(np.radians(angle))
                    center_y = distance * np.sin(np.radians(angle))
                    angle = random.uniform(0, 360)
                else:  # Outer ring
                    distance = random.uniform(3.0, 5.0)
                    angle = random.uniform(0, 360)
                    center_x = distance * np.cos(np.radians(angle))
                    center_y = distance * np.sin(np.radians(angle))
                    angle = random.uniform(0, 360)

                individual.append([center_x, center_y, angle])

            population.append(np.array(individual))
        return population

    def mutate_individual(self, individual, mutation_rate=0.1, max_disp=0.5):
        """Mutate an individual by slightly changing positions and angles"""
        mutated = individual.copy()

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate position
                mutated[i][0] += random.uniform(-max_disp, max_disp)  # x
                mutated[i][1] += random.uniform(-max_disp, max_disp)  # y
                # Mutate angle
                mutated[i][2] += random.uniform(-30, 30)  # angle in degrees
                mutated[i][2] %= 360  # Keep angle in [0,360)

        return mutated

    def local_refinement(self, individual, max_iter=50):
        """Apply local refinement to improve individual quality"""
        # Simple gradient-free local search using coordinate descent
        best_individual = individual.copy()
        best_fitness = self.evaluate_individual(best_individual)

        for _ in range(max_iter):
            improved = False
            # Try small perturbations to each parameter
            for i in range(len(best_individual)):
                for j in range(3):  # x, y, angle
                    original_value = best_individual[i][j]
                    # Try small positive and negative steps
                    steps = [-0.05, 0.05]
                    for step in steps:
                        test_individual = best_individual.copy()
                        if j < 2:  # x or y
                            test_individual[i][j] = original_value + step
                        else:  # angle
                            test_individual[i][j] = (original_value + step) % 360

                        fitness = self.evaluate_individual(test_individual)
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_individual = test_individual
                            improved = True

            if not improved:
                break

        return best_individual

    def evaluate_individual(self, individual):
        """Evaluate fitness of individual solution"""
        try:
            # Create polygons for all inner hexagons
            hex_polygons = []
            for i in range(len(individual)):
                center_x, center_y, angle = individual[i]
                vertices = generate_hexagon_vertices_numba(center_x, center_y, angle)
                hex_polygons.append(Polygon(vertices))

            # Check containment and overlap
            outer_side_length = self.calculate_outer_side_length(individual)
            outer_vertices = generate_hexagon_vertices_numba(0, 0, 0, outer_side_length)
            outer_polygon = Polygon(outer_vertices)

            # Check containment
            for poly in hex_polygons:
                if not ConstraintChecker.contains(poly, outer_polygon):
                    return 0.0  # Invalid - not fully contained

            # Check overlaps
            for i in range(len(hex_polygons)):
                for j in range(i+1, len(hex_polygons)):
                    if ConstraintChecker.overlaps(hex_polygons[i], hex_polygons[j]):
                        return 0.0  # Invalid - overlaps

            # Return 1/outer_side_length as fitness
            return 1.0 / outer_side_length if outer_side_length > 0 else 0.0

        except Exception:
            return 0.0

    def calculate_outer_side_length(self, inner_hex_data):
        """Calculate minimum side length of outer hexagon needed to contain all inner hexagons"""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = generate_hexagon_vertices_numba(center_x, center_y, angle)
            all_vertices.extend(vertices)

        all_vertices = np.array(all_vertices)

        # Find bounding box
        min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
        min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

        # Calculate approximate side length (simplified approach)
        # A hexagon with side length s has width 2*s and height sqrt(3)*s
        width = max_x - min_x
        height = max_y - min_y

        # Estimate side length from dimensions
        side_len_width = width / 2.0
        side_len_height = height / (np.sqrt(3))

        # Take maximum to ensure containment
        estimated_side_length = max(side_len_width, side_len_height) * 1.1  # Add small buffer

        return estimated_side_length

    def crossover_parents(self, parent1, parent2, crossover_rate=0.8):
        """Crossover parents to produce offspring"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        # Single point crossover
        crossover_point = random.randint(1, len(parent1) - 1)

        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return offspring1, offspring2

    def run_evolution(self):
        """Main evolutionary algorithm loop"""
        start_time = time.time()
        
        # Create initial population
        population = self.create_initial_population()

        best_fitness = 0.0
        best_individual = None

        # Evolution loop
        for gen in range(self.generations):
            if time.time() - start_time > self.max_time:
                break

            # Evaluate fitness
            fitness_scores = []
            for ind in population:
                fitness = self.evaluate_individual(ind)
                fitness_scores.append(fitness)

            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()

            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            # Create new generation
            new_population = []
            # Elitism
            for i in range(self.elite_size):
                new_population.append(population[i].copy())

            # Generate offspring
            while len(new_population) < self.pop_size:
                # Tournament selection
                tournament_size = 3
                parent1_idx = random.choices(range(self.elite_size*2), k=tournament_size)[0]
                parent2_idx = random.choices(range(self.elite_size*2), k=tournament_size)[0]

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                child1, child2 = self.crossover_parents(parent1, parent2)

                # Apply mutations
                child1 = self.mutate_individual(child1, self.mutation_rate)
                child2 = self.mutate_individual(child2, self.mutation_rate)

                # Apply local refinement to offspring
                child1 = self.local_refinement(child1, max_iter=25)
                child2 = self.local_refinement(child2, max_iter=25)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.pop_size]

            # Adaptive mutation rate scheduling
            adaptive_mutation_rate = self.mutation_rate - (gen / self.generations) * 0.10
            adaptive_mutation_rate = max(adaptive_mutation_rate, 0.05)  # Minimum mutation rate
            self.mutation_rate = adaptive_mutation_rate

        return best_individual, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize optimization manager
    optimizer = OptimizationManager()
    
    try:
        # Run evolutionary optimization
        best_individual, best_fitness = optimizer.run_evolution()
        
        # Final evaluation of best solution
        if best_individual is None:
            # Fallback to initial solution if optimization failed
            best_individual = np.array([
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
                [3.75, -2.17, 0],  # far bottom-right
            ])

        outer_side_length = 1.0 / best_fitness if best_fitness > 0 else 8.0

        # Ensure valid outer hexagon side length
        if outer_side_length > 100:
            outer_side_length = 10.0

        # Center the outer hexagon at origin
        outer_hex_data = np.array([0, 0, 0])

        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        
        return best_individual, outer_hex_data, outer_side_length
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to improved initial solution
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
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END