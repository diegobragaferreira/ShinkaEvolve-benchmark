# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
import random
from typing import Tuple, List, Optional
import warnings
from deap import base, creator, tools, algorithms
import copy
import multiprocessing as mp
from functools import partial

# Custom DEAP fitness class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class HexagonGeometry:
    """Handles all geometric operations for hexagon computations."""
    
    @staticmethod
    def create_hexagon(center: Tuple[float, float], side_length: float, rotation_degrees: float) -> Polygon:
        """Create a regular hexagon as a shapely polygon."""
        angle_rad = np.radians(rotation_degrees)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return Polygon(vertices)
    
    @staticmethod
    def compute_hexagon_vertices(center: Tuple[float, float], side_length: float, rotation_degrees: float) -> List[Tuple[float, float]]:
        """Compute vertices of a regular hexagon."""
        angle_rad = np.radians(rotation_degrees)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return vertices
    
    @staticmethod
    def check_containment(inner_hex: Polygon, outer_hex: Polygon) -> bool:
        """Check if inner hexagon is fully contained within outer hexagon."""
        return outer_hex.contains(inner_hex)
    
    @staticmethod
    def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap."""
        return hex1.intersects(hex2)
    
    @staticmethod
    def calculate_outer_hex_side_length(hexagons: List[Polygon]) -> float:
        """Calculate minimum outer hexagon side length that contains all inner hexagons."""
        if not hexagons:
            return 10.0
        
        # Collect all vertices from all hexagons
        all_points = []
        for hexagon in hexagons:
            all_points.extend(list(hexagon.exterior.coords))
        
        if not all_points:
            return 10.0
        
        # Calculate distances from center (0,0) to all points
        distances = []
        for point in all_points:
            dist = np.sqrt(point[0]**2 + point[1]**2)
            distances.append(dist)
        
        if not distances:
            return 10.0
            
        # Maximum distance determines the required outer radius
        max_dist = max(distances)
        
        # For regular hexagon, side length = max_dist / sqrt(3)
        return max_dist / np.sqrt(3) * 2

class ConstraintValidator:
    """Validates geometric constraints for hexagon packing."""
    
    @classmethod
    def validate_packing(cls, hexagon_data: np.ndarray, outer_center: Tuple[float, float] = (0, 0)) -> Tuple[bool, float]:
        """
        Validate if all constraints are satisfied for the packing.
        
        Returns:
            tuple: (is_valid, inverse_outer_side_length)
        """
        try:
            # Create individual hexagon polygons
            hexagons = []
            for i in range(12):
                center = (hexagon_data[i][0], hexagon_data[i][1])
                angle = hexagon_data[i][2]
                hexagon = HexagonGeometry.create_hexagon(center, 1.0, angle)
                hexagons.append(hexagon)
            
            # Check for overlaps between any pair of hexagons
            for i in range(12):
                for j in range(i+1, 12):
                    if HexagonGeometry.check_overlap(hexagons[i], hexagons[j]):
                        return False, 0.0
            
            # Create outer hexagon
            outer_side_length = HexagonGeometry.calculate_outer_hex_side_length(hexagons)
            outer_hexagon = HexagonGeometry.create_hexagon(outer_center, outer_side_length, 0.0)
            
            # Check containment of all hexagon vertices
            for hexagon in hexagons:
                for vertex in hexagon.exterior.coords:
                    point = Point(vertex[0], vertex[1])
                    if not outer_hexagon.contains(point):
                        return False, 0.0
            
            # Return inverse of outer side length as objective
            return True, 1.0 / outer_side_length
            
        except Exception as e:
            warnings.warn(f"Constraint validation error: {e}")
            return False, 0.0

class EvolutionaryOptimizer:
    """Evolutionary optimization engine for hexagon packing."""
    
    def __init__(self, timeout: float = 170):
        self.toolbox = base.Toolbox()
        self.timeout = timeout
        self.start_time = time.time()
        self.init_toolbox()
        
    def init_toolbox(self):
        """Initialize the DEAP toolbox with custom operators."""
        # Define the structure of an individual (12 hexagons with (x,y,angle))
        IND_SIZE = 36  # 12 * 3 parameters
        
        def create_individual():
            # Create random individual within bounds
            individual = []
            for i in range(12):
                # x coordinate (-10 to 10)
                individual.append(random.uniform(-10, 10))
                # y coordinate (-10 to 10)
                individual.append(random.uniform(-10, 10))
                # angle (0 to 360 degrees)
                individual.append(random.uniform(0, 360))
            return creator.Individual(individual)
        
        def mutate_individual(individual):
            # Adaptive mutation with decreasing rate
            mutation_rate = 0.1
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    if i % 3 == 0:  # x coordinate
                        individual[i] += random.gauss(0, 0.5)  # Small random step
                        individual[i] = max(-10, min(10, individual[i]))
                    elif i % 3 == 1:  # y coordinate
                        individual[i] += random.gauss(0, 0.5)
                        individual[i] = max(-10, min(10, individual[i]))
                    else:  # angle
                        individual[i] += random.gauss(0, 10)
                        individual[i] = individual[i] % 360
            return individual,
        
        def crossover_individuals(ind1, ind2):
            # Uniform crossover for hexagon positions and rotations
            for i in range(len(ind1)):
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2
        
        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", crossover_individuals)
        self.toolbox.register("mutate", mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual."""
        # Convert individual to hexagon data format
        hexagon_data = np.array(individual).reshape(-1, 3)
        
        # Validate the packing
        is_valid, objective_value = ConstraintValidator.validate_packing(hexagon_data)
        
        if not is_valid:
            # Penalize invalid configurations heavily
            return -1e6
        
        return objective_value
    
    def get_current_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def optimize_with_evolution(self, initial_pop_size=50, generations=100):
        """Run evolutionary optimization with timeout control."""
        pop = self.toolbox.population(n=initial_pop_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.evaluate_individual, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,)
        
        # Evolutionary loop
        for gen in range(generations):
            # Check timeout
            if self.get_current_time() > self.timeout * 0.9:
                break
                
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.evaluate_individual, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)
            
            # Replace population
            pop[:] = offspring
            
            # Track best individual
            best_ind = tools.selBest(pop, 1)[0]
            best_fitness = best_ind.fitness.values[0]
            
            # Print progress
            if gen % 20 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}")
        
        # Return the best individual
        best_ind = tools.selBest(pop, 1)[0]
        return np.array(best_ind).reshape(-1, 3)

class ConfigurationGenerator:
    """Generates various initial configurations for optimization."""
    
    @staticmethod
    def get_standard_configurations() -> List[np.ndarray]:
        """Generate standard initial configurations."""
        configs = []
        
        # Configuration 1: Traditional 2-ring arrangement with symmetry
        config1 = np.array([
            [0.0, 0.0, 0],           # Center
            [0.0, 2.0, 0],           # Top
            [1.732050808, 1.0, 0],   # Top right
            [1.732050808, -1.0, 0],  # Bottom right
            [0.0, -2.0, 0],          # Bottom
            [-1.732050808, -1.0, 0], # Bottom left
            [-1.732050808, 1.0, 0],  # Top left
            [3.464101616, 2.0, 0],   # Far top right
            [3.464101616, -2.0, 0],  # Far bottom right
            [-3.464101616, -2.0, 0], # Far bottom left
            [-3.464101616, 2.0, 0],  # Far top left
            [0.0, -4.0, 0],          # Far bottom
        ], dtype=float)
        configs.append(config1)
        
        # Configuration 2: Honeycomb-like arrangement with different spacing
        config2 = np.array([
            [0.0, 0.0, 0],
            [2.0, 0.0, 0],
            [1.0, 1.732050808, 0],
            [-1.0, 1.732050808, 0],
            [-2.0, 0.0, 0],
            [-1.0, -1.732050808, 0],
            [1.0, -1.732050808, 0],
            [3.0, 1.732050808, 0],
            [3.0, -1.732050808, 0],
            [-3.0, -1.732050808, 0],
            [-3.0, 1.732050808, 0],
            [0.0, -3.464101616, 0],
        ], dtype=float)
        configs.append(config2)
        
        # Configuration 3: Starburst-pattern arrangement
        config3 = np.array([
            [0.0, 0.0, 0],
            [0.0, 2.5, 0],
            [2.165, 1.25, 0],
            [2.165, -1.25, 0],
            [0.0, -2.5, 0],
            [-2.165, -1.25, 0],
            [-2.165, 1.25, 0],
            [0.0, 4.0, 0],
            [3.464, 2.0, 0],
            [3.464, -2.0, 0],
            [0.0, -4.0, 0],
            [-3.464, -2.0, 0],
        ], dtype=float)
        configs.append(config3)
        
        # Configuration 4: Triangular lattice pattern
        config4 = np.array([
            [0.0, 0.0, 0],
            [0.0, 2.0, 0],
            [1.732050808, 1.0, 0],
            [1.732050808, -1.0, 0],
            [0.0, -2.0, 0],
            [-1.732050808, -1.0, 0],
            [-1.732050808, 1.0, 0],
            [3.464101616, 2.0, 0],
            [3.464101616, -2.0, 0],
            [-3.464101616, -2.0, 0],
            [-3.464101616, 2.0, 0],
            [0.0, -4.0, 0],
        ], dtype=float)
        configs.append(config4)
        
        return configs
    
    @staticmethod
    def generate_random_perturbed_config(base_config: np.ndarray, perturbation_magnitude: float = 0.1) -> np.ndarray:
        """Generate a random perturbed version of a base configuration."""
        perturbed = base_config.copy()
        for i in range(len(perturbed)):
            # Perturb positions slightly
            perturbed[i][0] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
            perturbed[i][1] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
        return perturbed

class HexagonPackingOptimizer:
    """Main optimizer coordinating the complete packing process."""
    
    def __init__(self, timeout: float = 180):
        self.timeout = timeout
        self.start_time = time.time()
        self.best_score = 0.0
        self.best_config = None
        
    def get_current_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
        
    def initialize_configurations(self) -> List[np.ndarray]:
        """Generate diverse initial configurations."""
        generator = ConfigurationGenerator()
        configs = generator.get_standard_configurations()
        
        # Add some random variants to ensure exploration
        for config in configs[:2]:  # Only perturb first two
            perturbed = generator.generate_random_perturbed_config(config, 0.05)
            configs.append(perturbed)
        
        return configs
    
    def stage_one_coarse_optimization(self, config: np.ndarray) -> Tuple[np.ndarray, float]:
        """Stage 1: Coarse optimization with fixed rotations."""
        print("Stage 1: Coarse position optimization...")
        # Fix rotations for this stage
        fixed_config = config.copy()
        for i in range(len(fixed_config)):
            fixed_config[i][2] = 0  # Set all rotations to 0 for coarse optimization
        
        # Use scipy optimization for this stage
        def objective_func(params):
            positions = params.reshape(-1, 2)
            temp_data = fixed_config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            outer_radius = self.calculate_outer_radius(temp_data)
            return outer_radius

        def constraint_func(params):
            positions = params.reshape(-1, 2)
            temp_data = fixed_config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]

            # Create hexagon polygons
            hexagons = []
            for i in range(12):
                center = (positions[i][0], positions[i][1])
                rotation = fixed_config[i][2]
                hexagon = HexagonGeometry.create_hexagon(center, 1.0, rotation)
                hexagons.append(hexagon)

            penalty = 0
            for i in range(12):
                for j in range(i+1, 12):
                    if HexagonGeometry.check_overlap(hexagons[i], hexagons[j]):
                        penalty += 1000
            return penalty

        try:
            # Flatten the initial positions for optimization
            initial_positions = np.column_stack((fixed_config[:, 0], fixed_config[:, 1])).flatten()

            result = minimize(objective_func, initial_positions, method='L-BFGS-B',
                            bounds=[(-5, 5) for _ in range(24)],
                            constraints={'type': 'ineq', 'fun': constraint_func},
                            options={'maxiter': 200})

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                fixed_config[:, 0] = final_positions[:, 0]
                fixed_config[:, 1] = final_positions[:, 1]
        except:
            pass  # Fall back to original config if optimization fails

        return fixed_config, self.evaluate_configuration(fixed_config)
    
    def stage_two_refinement(self, config: np.ndarray) -> Tuple[np.ndarray, float]:
        """Stage 2: Fine-grained refinement with rotation awareness."""
        print("Stage 2: Fine-grained refinement...")
        # Allow rotations to vary but keep positions relatively close to evolved ones
        refined_config = config.copy()
        # Add small random rotations to improve packing
        for i in range(len(refined_config)):
            # Perturb rotations slightly
            refined_config[i][2] += random.uniform(-5, 5) if random.random() < 0.4 else 0

        # Use scipy optimization for this stage
        def objective_func(params):
            positions = params.reshape(-1, 2)
            temp_data = refined_config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            outer_radius = self.calculate_outer_radius(temp_data)
            return outer_radius

        def constraint_func(params):
            positions = params.reshape(-1, 2)
            temp_data = refined_config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]

            # Create hexagon polygons
            hexagons = []
            for i in range(12):
                center = (positions[i][0], positions[i][1])
                rotation = refined_config[i][2]
                hexagon = HexagonGeometry.create_hexagon(center, 1.0, rotation)
                hexagons.append(hexagon)

            penalty = 0
            for i in range(12):
                for j in range(i+1, 12):
                    if HexagonGeometry.check_overlap(hexagons[i], hexagons[j]):
                        penalty += 1000
            return penalty

        try:
            # Flatten the initial positions for optimization
            initial_positions = np.column_stack((refined_config[:, 0], refined_config[:, 1])).flatten()

            result = minimize(objective_func, initial_positions, method='L-BFGS-B',
                            bounds=[(-5, 5) for _ in range(24)],
                            constraints={'type': 'ineq', 'fun': constraint_func},
                            options={'maxiter': 200})

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                refined_config[:, 0] = final_positions[:, 0]
                refined_config[:, 1] = final_positions[:, 1]
        except:
            pass  # Fall back to previous config if optimization fails

        return refined_config, self.evaluate_configuration(refined_config)
    
    def stage_three_evolution(self, config: np.ndarray) -> Tuple[np.ndarray, float]:
        """Stage 3: Evolutionary optimization for final refinement."""
        print("Stage 3: Evolutionary optimization...")
        # Initialize and run evolutionary optimization
        optimizer = EvolutionaryOptimizer(timeout=self.timeout * 0.8)
        evolved_config = optimizer.optimize_with_evolution(initial_pop_size=30, generations=50)
        
        # Final evaluation
        score = self.evaluate_configuration(evolved_config)
        return evolved_config, score
    
    def calculate_outer_radius(self, hex_data: np.ndarray) -> float:
        """Calculate the outer radius of the configuration."""
        all_vertices = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = HexagonGeometry.create_hexagon(center, 1.0, rotation)
            all_vertices.extend(list(hexagon.exterior.coords))
        
        max_distance = 0
        for vertex in all_vertices:
            distance = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_distance = max(max_distance, distance)
        return max_distance + 0.1
    
    def evaluate_configuration(self, hex_data: np.ndarray) -> float:
        """Evaluate a configuration and return the inverse radius."""
        outer_radius = self.calculate_outer_radius(hex_data)

        # Create hexagon polygons
        hexagons = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = HexagonGeometry.create_hexagon(center, 1.0, rotation)
            hexagons.append(hexagon)

        # Compute penalties
        penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                if HexagonGeometry.check_overlap(hexagons[i], hexagons[j]):
                    penalty += 1000

        total_penalty = penalty

        # If valid configuration, return inverse of outer radius; otherwise return a very small value
        if total_penalty == 0:
            return 1.0 / outer_radius
        else:
            # Invalid configuration gets penalized heavily
            return 1e-10

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete multi-stage optimization pipeline."""
        # Get multiple symmetric configurations
        configs = self.initialize_configurations()

        # Try multiple configurations and find the best starting point
        best_initial_score = 0
        best_initial_config = None

        for config in configs:
            score = self.evaluate_configuration(config)
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()

        # Store the best configuration found so far
        self.best_score = best_initial_score
        self.best_config = best_initial_config.copy()

        # Stage 1: Coarse-grained position optimization (fixed rotations)
        evolved_config, evolved_score = self.stage_one_coarse_optimization(best_initial_config)

        # Stage 2: Fine-grained refinement with rotation awareness
        refined_config, refined_score = self.stage_two_refinement(evolved_config)

        # Stage 3: Evolutionary optimization for final refinement
        final_config, final_score = self.stage_three_evolution(refined_config)

        # Final evaluation
        final_score = self.evaluate_configuration(final_config)
        final_outer_radius = self.calculate_outer_radius(final_config)
        outer_hex_side_length = final_outer_radius + 0.2  # Add margin

        # Return result
        outer_hex_data = np.array([0, 0, 0])  # centered at origin

        return final_config, outer_hex_data, outer_hex_side_length

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
        optimizer = HexagonPackingOptimizer(timeout=180)
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimizer.run_full_optimization()
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        warnings.warn(f"Main optimization failed: {e}")
        # Fallback to a reliable configuration
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
            [0, -4, 0]
        ])
        
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END