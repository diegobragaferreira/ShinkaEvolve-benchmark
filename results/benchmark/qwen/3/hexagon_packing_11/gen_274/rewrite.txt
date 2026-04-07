# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
import math
from numba import jit
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class Hexagon:
    """Represents a regular hexagon with center, rotation and side length"""
    
    def __init__(self, center_x: float, center_y: float, angle_degrees: float, side_length: float = 1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.angle_degrees = angle_degrees
        self.side_length = side_length
    
    @staticmethod
    @jit(nopython=True)
    def _generate_base_vertices(side_length: float) -> np.ndarray:
        """Generate base vertices of a unit hexagon centered at origin"""
        sqrt3 = np.sqrt(3)
        return np.array([
            [side_length, 0.0],
            [side_length/2.0, sqrt3/2.0 * side_length],
            [-side_length/2.0, sqrt3/2.0 * side_length],
            [-side_length, 0.0],
            [-side_length/2.0, -sqrt3/2.0 * side_length],
            [side_length/2.0, -sqrt3/2.0 * side_length]
        ], dtype=np.float64)
    
    def get_vertices(self) -> np.ndarray:
        """Get vertices of the hexagon with current transformation"""
        # Get base vertices
        base_vertices = self._generate_base_vertices(self.side_length)
        
        # Apply rotation
        angle_rad = np.radians(self.angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        
        rotated_vertices = base_vertices @ rotation_matrix.T
        
        # Apply translation
        return rotated_vertices + np.array([self.center_x, self.center_y], dtype=np.float64)
    
    def to_polygon(self) -> Polygon:
        """Convert hexagon to shapely polygon"""
        return Polygon(self.get_vertices())

class SpatialHexagonPacker:
    """Main optimization class using spatial-guided approach"""
    
    def __init__(self, n_inner_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length
        
    def generate_spatial_initial_config(self) -> np.ndarray:
        """Generate initial configuration based on spatial hexagonal tiling"""
        # Create a structured layout that naturally avoids overlaps
        # We arrange in concentric rings around center
        
        config = []
        
        # Central hexagon
        config.append([0.0, 0.0, 0.0])
        
        # First ring (6 hexagons) - radius = 2 (spacing = 2)
        for i in range(6):
            angle = i * 60  # 60 degree increments
            x = 2.0 * np.cos(np.radians(angle))
            y = 2.0 * np.sin(np.radians(angle))
            config.append([x, y, 0.0])
        
        # Second ring (12 hexagons) - radius = 4
        for i in range(12):
            angle = i * 30  # 30 degree increments
            distance = 4.0
            x = distance * np.cos(np.radians(angle))
            y = distance * np.sin(np.radians(angle))
            config.append([x, y, 0.0])
        
        # Trim to exactly 11 hexagons
        if len(config) > 11:
            # Keep central + first ring (7 total) + add some from second ring
            config = [config[0]] + config[1:7] + config[7:11]  # Take all from ring 1, and 4 from ring 2
            config = config[:11]
        
        # Add small random perturbations to escape local optima
        result = np.array(config)
        for i in range(len(result)):
            # Add perturbations with decreasing magnitude to avoid large disruptions
            perturbation_magnitude = 0.5 if i < 7 else 0.2  # More freedom for outer hexagons
            result[i][0] += np.random.normal(0, perturbation_magnitude)
            result[i][1] += np.random.normal(0, perturbation_magnitude)
            # Angle perturbation with higher variance for outer hexagons
            angle_perturbation = 30 if i < 7 else 15
            result[i][2] += np.random.normal(0, angle_perturbation)
            result[i][2] = result[i][2] % 360
            
        return result
    
    def create_hexagons_from_array(self, hex_data: np.ndarray) -> list[Hexagon]:
        """Convert array data to list of Hexagon objects"""
        return [Hexagon(row[0], row[1], row[2], self.hex_side_length) for row in hex_data]
    
    def check_containment(self, hexagons: list[Hexagon], outer_radius: float) -> bool:
        """Check if all hexagons are contained within outer hexagon of given radius"""
        # Create outer hexagon centered at origin
        outer_hex = Hexagon(0.0, 0.0, 0.0, outer_radius)
        outer_polygon = outer_hex.to_polygon()
        
        for hexagon in hexagons:
            hex_polygon = hexagon.to_polygon()
            if not outer_polygon.contains(hex_polygon):
                return False
        return True
    
    def check_overlap(self, hexagons: list[Hexagon]) -> bool:
        """Check if any hexagons overlap"""
        polygons = [h.to_polygon() for h in hexagons]
        
        # Check pairwise overlaps with early termination
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return True
        return False
    
    def compute_min_radius(self, hexagons: list[Hexagon]) -> float:
        """Compute minimum outer radius that contains all hexagons"""
        # Get all vertices from all hexagons
        all_vertices = []
        for hexagon in hexagons:
            vertices = hexagon.get_vertices()
            all_vertices.extend(vertices)
        
        # Find maximum distance from center to any vertex
        max_dist = 0
        for x, y in all_vertices:
            dist = math.sqrt(x*x + y*y)
            max_dist = max(max_dist, dist)
        
        # Add small buffer to ensure containment
        return max_dist + 0.01
    
    def evaluate_fitness(self, hexagons: list[Hexagon]) -> float:
        """Evaluate fitness based on geometric constraints and packing density"""
        # Check constraints
        if not self.check_containment(hexagons, 10.0):  # High bound for initial check
            return -np.inf  # Invalid - penalty
        
        if self.check_overlap(hexagons):
            return -np.inf  # Invalid - penalty
            
        # Compute actual radius needed
        outer_radius = self.compute_min_radius(hexagons)
        
        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius
    
    def mutate_individual(self, individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply mutation with spatial awareness and progressive refinement"""
        mutated = individual.copy()
        
        # Dynamic mutation rate based on generation
        mutation_rate = max(0.05, 0.2 * (1 - generation/max_generations))
        
        # Calculate spatial relationships to guide mutation
        positions = individual[:, :2]
        avg_distance = np.mean([
            np.linalg.norm(pos1 - pos2) 
            for i, pos1 in enumerate(positions) 
            for j, pos2 in enumerate(positions) if i != j
        ])
        
        # Apply mutation with spatial awareness
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Base perturbation magnitude depends on generation and spatial context
                base_magnitude = 0.3 if generation < max_generations//2 else 0.1
                spatial_magnitude = base_magnitude * (1 + avg_distance/10.0)
                
                # Mutate position with spatial guidance
                mutated[i][0] += np.random.normal(0, spatial_magnitude)
                mutated[i][1] += np.random.normal(0, spatial_magnitude)
                
                # Mutate angle with direction awareness
                mutated[i][2] += np.random.normal(0, 15)  # Less aggressive angle changes
                mutated[i][2] = mutated[i][2] % 360
                
        return mutated
    
    def local_optimize_individual(self, individual: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply local optimization with progressive refinement"""
        def objective(params):
            # Reshape params back to hexagon data
            new_data = individual.copy()
            for i in range(len(new_data)):
                new_data[i][0] = params[i*3]
                new_data[i][1] = params[i*3+1]
                new_data[i][2] = params[i*3+2]
            
            # Convert to hexagon objects for evaluation
            hexagons = self.create_hexagons_from_array(new_data)
            
            # Evaluate fitness
            fitness = self.evaluate_fitness(hexagons)
            return -fitness  # minimize negative fitness
        
        # Flatten the data for optimization
        initial_params = []
        for i in range(len(individual)):
            initial_params.extend([individual[i][0], individual[i][1], individual[i][2]])
        
        # Multi-stage local optimization
        try:
            # Stage 1: Coarse optimization
            result1 = minimize(objective, initial_params, method='L-BFGS-B', 
                             bounds=[(-10, 10), (-10, 10), (0, 360)] * len(individual),
                             options={'maxiter': max_iter//2})
            
            if result1.success and not np.isnan(result1.x).any():
                # Stage 2: Finer optimization
                refined_params = result1.x.copy()
                result2 = minimize(objective, refined_params, method='L-BFGS-B',
                                 bounds=[(-5, 5), (-5, 5), (0, 360)] * len(individual),
                                 options={'maxiter': max_iter//2})
                
                if result2.success and not np.isnan(result2.x).any():
                    # Reshape optimized result back
                    refined_data = individual.copy()
                    for i in range(len(refined_data)):
                        refined_data[i][0] = result2.x[i*3]
                        refined_data[i][1] = result2.x[i*3+1]
                        refined_data[i][2] = result2.x[i*3+2]
                    return refined_data
                    
        except Exception:
            pass
        
        return individual
    
    def spatial_guided_evolution(self, max_generations: int = 50) -> tuple[np.ndarray, float]:
        """Main optimization loop with spatial guidance"""
        # Start with spatially informed initialization
        population = [self.generate_spatial_initial_config()]
        
        # Add diverse random starting points
        for _ in range(9):  # Total 10 individuals
            individual = np.zeros((self.n_inner_hexagons, 3))
            for i in range(self.n_inner_hexagons):
                # Random positions within reasonable bounds
                x = np.random.uniform(-8, 8)
                y = np.random.uniform(-8, 8)
                angle = np.random.uniform(0, 360)
                individual[i] = [x, y, angle]
            population.append(individual)
        
        best_fitness = -np.inf
        best_config = None
        best_radius = 10.0
        
        for gen in range(max_generations):
            # Evaluate fitness with early filtering
            fitness_scores = []
            valid_individuals = []
            
            for individual in population:
                hexagons = self.create_hexagons_from_array(individual)
                
                # Quick overlap check for filtering
                if self.check_overlap(hexagons):
                    fitness_scores.append(-np.inf)
                    continue
                    
                # Compute fitness
                fitness = self.evaluate_fitness(hexagons)
                fitness_scores.append(fitness)
                valid_individuals.append(individual)
            
            # Update best solution
            if len(fitness_scores) > 0:
                max_idx = np.argmax(fitness_scores)
                if fitness_scores[max_idx] > best_fitness:
                    best_fitness = fitness_scores[max_idx]
                    best_config = population[max_idx].copy()
                    
                    # Recalculate precise radius
                    hexagons = self.create_hexagons_from_array(best_config)
                    best_radius = self.compute_min_radius(hexagons)
            
            # Selection and reproduction with spatial awareness
            if len(fitness_scores) > 0 and len(valid_individuals) > 0:
                # Select top performers
                sorted_indices = np.argsort(fitness_scores)[::-1][:len(valid_individuals)//2]
                selected = [valid_individuals[i] for i in sorted_indices if fitness_scores[i] > -np.inf]
                
                # Create new population
                new_population = selected.copy()
                
                # Add some diversity with mutations
                for _ in range(10 - len(selected)):
                    if len(selected) > 0:
                        # Select parent with probability proportional to fitness
                        parent_idx = np.random.choice(len(selected), p=np.array(fitness_scores)[sorted_indices]/np.sum(np.array(fitness_scores)[sorted_indices]))
                        parent = selected[parent_idx]
                        mutated = self.mutate_individual(parent, gen, max_generations)
                        new_population.append(mutated)
                    else:
                        # If no valid individuals, create new random one
                        individual = np.zeros((self.n_inner_hexagons, 3))
                        for i in range(self.n_inner_hexagons):
                            x = np.random.uniform(-8, 8)
                            y = np.random.uniform(-8, 8)
                            angle = np.random.uniform(0, 360)
                            individual[i] = [x, y, angle]
                        new_population.append(individual)
                
                population = new_population
            else:
                # If no valid individuals, continue with current population
                pass
        
        return best_config, best_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize packer
    packer = SpatialHexagonPacker(n_inner_hexagons=11, hex_side_length=1.0)
    
    # Apply spatial-guided evolutionary optimization
    best_config, best_radius = packer.spatial_guided_evolution(max_generations=50)
    
    # Final local optimization on best solution
    if best_config is not None:
        # Perform final local optimization
        final_config = packer.local_optimize_individual(best_config, max_iter=150)
        final_hexagons = packer.create_hexagons_from_array(final_config)
        final_radius = packer.compute_min_radius(final_hexagons)
        
        # Check if final optimization improved fitness
        final_fitness = packer.evaluate_fitness(final_hexagons)
        initial_fitness = packer.evaluate_fitness(packer.create_hexagons_from_array(best_config))
        
        if final_fitness > initial_fitness:
            best_config = final_config
            best_radius = final_radius
    
    # Prepare output
    inner_hex_data = best_config if best_config is not None else np.zeros((11, 3))
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = best_radius if 'best_radius' in locals() else 8.0
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END