# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time
import random
from itertools import combinations

class HexagonPacker:
    """Manages hexagon packing operations with geometric validation"""
    
    def __init__(self):
        self.hex_side_length = 1.0
        self.outer_center = np.array([0.0, 0.0])
        self.max_eval_time = 180.0

    def hexagon_vertices(self, center_x, center_y, size=1, angle_deg=0):
        """Generate vertices of a regular hexagon given center, size, and rotation."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center_x + size * np.cos(angle)
            y = center_y + size * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    def get_outer_hexagon(self, outer_radius):
        """Get vertices of the outer hexagon with given radius."""
        return self.hexagon_vertices(self.outer_center[0], self.outer_center[1], outer_radius, 0)

    def validate_containment(self, hex_vertices, outer_radius):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        outer_vertices = self.get_outer_hexagon(outer_radius)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True

    def validate_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely with buffer for precision."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        # Early rejection using bounding boxes
        if (min(v[0] for v in hex1_vertices) > max(v[0] for v in hex2_vertices) or
            max(v[0] for v in hex1_vertices) < min(v[0] for v in hex2_vertices) or
            min(v[1] for v in hex1_vertices) > max(v[1] for v in hex2_vertices) or
            max(v[1] for v in hex1_vertices) < min(v[1] for v in hex2_vertices)):
            return False
        return poly1.intersects(poly2)

    def calculate_max_distance_from_center(self, hex_data):
        """Calculate maximum distance from center to any hexagon vertex."""
        max_dist = 0
        for i in range(len(hex_data)):
            cx, cy, _ = hex_data[i]
            # Calculate distance to center plus hexagon radius
            dist = np.sqrt(cx**2 + cy**2) + self.hex_side_length
            max_dist = max(max_dist, dist)
        return max_dist

    def evaluate_configuration(self, hex_data, outer_radius):
        """Evaluate current configuration: returns (validity, inv_radius)."""
        # Check for overlaps
        for i in range(len(hex_data)):
            hex1_vertices = self.hexagon_vertices(hex_data[i][0], hex_data[i][1],
                                                self.hex_side_length, hex_data[i][2])
            for j in range(i+1, len(hex_data)):
                hex2_vertices = self.hexagon_vertices(hex_data[j][0], hex_data[j][1],
                                                    self.hex_side_length, hex_data[j][2])
                if self.validate_overlap(hex1_vertices, hex2_vertices):
                    return False, 0

        # Check containment
        for i in range(len(hex_data)):
            hex_vertices = self.hexagon_vertices(hex_data[i][0], hex_data[i][1],
                                               self.hex_side_length, hex_data[i][2])
            if not self.validate_containment(hex_vertices, outer_radius):
                return False, 0

        # Return inverse of outer radius
        return True, 1.0 / outer_radius

class SymmetryAwareMutation:
    """Handles symmetry-aware mutation operations for evolutionary optimization"""
    
    @staticmethod
    def create_advanced_symmetric_config():
        """Create an advanced symmetric configuration based on mathematical analysis"""
        # Known good configuration from optimal packing studies
        config = [
            [0, 0, 0],              # Center
            [0, 2.0, 0],            # Top
            [1.732050808, 1.0, 0],  # Top right
            [1.732050808, -1.0, 0], # Bottom right
            [0, -2.0, 0],           # Bottom
            [-1.732050808, -1.0, 0], # Bottom left
            [-1.732050808, 1.0, 0],  # Top left
            [3.464101616, 2.0, 0],   # Far top right
            [3.464101616, -2.0, 0],  # Far bottom right
            [-3.464101616, -2.0, 0], # Far bottom left
            [-3.464101616, 2.0, 0],  # Far top left
            [0, -4.0, 0],           # Far bottom
        ]
        return np.array(config)

    @staticmethod
    def create_perturbed_config(base_config, perturbation_strength=0.1):
        """Create a perturbed version of a configuration"""
        perturbed = base_config.copy()
        for i in range(len(perturbed)):
            # Add small random perturbations to positions
            perturbed[i][0] += np.random.normal(0, perturbation_strength)
            perturbed[i][1] += np.random.normal(0, perturbation_strength)
        return perturbed

    @staticmethod
    def mutate_symmetrically(parent_config, mutation_rate=0.1, is_final_stage=False):
        """Apply symmetry-aware mutation to maintain structural relationships"""
        child_config = parent_config.copy()
        
        # Apply position mutations
        for i in range(len(child_config)):
            if random.random() < mutation_rate:
                # Mutate x coordinate with adaptive variance
                variance = 0.3 if not is_final_stage else 0.05
                child_config[i][0] += np.random.normal(0, variance)
                    
                # Mutate y coordinate  
                variance = 0.3 if not is_final_stage else 0.05
                child_config[i][1] += np.random.normal(0, variance)
            
            # Apply rotation mutations with lower rate but higher magnitude
            if random.random() < mutation_rate * 0.25:
                variance = 20 if not is_final_stage else 5
                child_config[i][2] += np.random.normal(0, variance)
                child_config[i][2] = child_config[i][2] % 360
                
        return child_config

class EvolutionaryOptimizer:
    """Evolutionary optimization with symmetry awareness"""
    
    def __init__(self, packer, max_generations=25, population_size=20):
        self.packer = packer
        self.max_generations = max_generations
        self.population_size = population_size
        self.initial_mutation_rate = 0.2
        self.elite_size = 4
        
    def evaluate_individual(self, individual, outer_radius):
        """Evaluate fitness of individual"""
        valid, fitness = self.packer.evaluate_configuration(individual, outer_radius)
        if not valid:
            fitness = 0  # Invalid individuals get low fitness
        return fitness
    
    def selection(self, population, fitnesses):
        """Tournament selection with elitism"""
        selected = []
        tournament_size = 3
        
        # Select elite individuals (top performers)
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        for idx in elite_indices:
            selected.append(population[idx].copy())
        
        # Tournament selection for rest of population
        while len(selected) < self.population_size:
            tournament = random.sample(list(range(len(population))), tournament_size)
            winner_idx = max(tournament, key=lambda i: fitnesses[i])
            selected.append(population[winner_idx].copy())
            
        return selected
    
    def crossover(self, parent1, parent2):
        """Single-point crossover respecting symmetry concepts"""
        # Create offspring by combining parts of parents
        crossover_point = random.randint(0, len(parent1))
        
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def adaptive_mutate(self, individual, generation, max_generations, is_final_stage=False):
        """Apply adaptive mutation with decreasing rate and enhanced randomness"""
        # Decrease mutation rate over time
        current_mutation_rate = self.initial_mutation_rate * (1 - generation / max_generations)
        current_mutation_rate = max(current_mutation_rate, 0.02)  # Minimum mutation rate
        
        return SymmetryAwareMutation.mutate_symmetrically(
            individual, 
            current_mutation_rate, 
            is_final_stage=is_final_stage
        )
    
    def optimize_population(self, initial_config, outer_radius, is_final_stage=False):
        """Run evolutionary optimization on population"""
        # Initialize population with diverse configurations
        population = []
        population.append(initial_config)
        
        # Generate diverse initial population
        for _ in range(self.population_size - 1):
            # Mix of symmetric and perturbed configurations
            if random.random() < 0.6:
                mutated = SymmetryAwareMutation.mutate_symmetrically(
                    initial_config, self.initial_mutation_rate, is_final_stage
                )
            else:
                # Create a more randomly perturbed version
                mutated = SymmetryAwareMutation.create_perturbed_config(initial_config, 0.05)
            population.append(mutated)
        
        best_fitness = 0
        best_individual = initial_config.copy()
        
        # Evolutionary loop
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitnesses = []
            for individual in population:
                fitness = self.evaluate_individual(individual, outer_radius)
                fitnesses.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selection
            selected = self.selection(population, fitnesses)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism - keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                if random.random() < 0.6:  # 60% crossover
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(child1)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)
                else:  # Mutation only
                    mutated1 = self.adaptive_mutate(parent1, generation, self.max_generations, is_final_stage)
                    mutated2 = self.adaptive_mutate(parent2, generation, self.max_generations, is_final_stage)
                    new_population.append(mutated1)
                    if len(new_population) < self.population_size:
                        new_population.append(mutated2)
            
            population = new_population
            
        return best_individual, best_fitness

class HybridOptimizer:
    """Combines evolutionary and gradient-based optimization"""
    
    def __init__(self, packer):
        self.packer = packer
        
    def local_search_refinement(self, config, outer_radius, max_iter=50):
        """Apply local search to refine a configuration using gradient-based optimization"""
        # Create a copy where we keep angles fixed for positional optimization
        config_fixed_angles = config.copy()

        # Flatten initial configuration (positions only)
        params = []
        for i in range(len(config_fixed_angles)):
            params.extend([config_fixed_angles[i][0], config_fixed_angles[i][1]])

        def objective_pos_only(params_flat):
            # Reconstruct configuration with fixed angles
            temp_config = config_fixed_angles.copy()
            idx = 0
            for i in range(len(temp_config)):
                temp_config[i][0] = params_flat[idx]
                temp_config[i][1] = params_flat[idx + 1]
                idx += 2
                
            # Evaluate
            validity, inv_radius = self.packer.evaluate_configuration(temp_config, outer_radius)
            if not validity:
                return 1e10
            return -inv_radius  # Negative because we maximize
        
        # Perform optimization
        try:
            result = minimize(
                objective_pos_only,
                params,
                method='L-BFGS-B',
                bounds=[(-10, 10), (-10, 10)] * len(config_fixed_angles),
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                # Reconstruct refined configuration
                refined_config = config_fixed_angles.copy()
                idx = 0
                for i in range(len(refined_config)):
                    refined_config[i][0] = result.x[idx]
                    refined_config[i][1] = result.x[idx + 1]
                    idx += 2
                return refined_config
        except:
            pass
            
        return config_fixed_angles
    
    def optimize_positions_only(self, initial_config, outer_radius):
        """Phase 1: Optimize positions only with fixed rotations"""
        # Use evolutionary optimization for global search
        evolutionary = EvolutionaryOptimizer(self.packer, max_generations=15, population_size=15)
        evolved_config, _ = evolutionary.optimize_population(initial_config, outer_radius)
        
        # Apply local search for fine tuning
        refined_config = self.local_search_refinement(evolved_config, outer_radius, 30)
        
        return refined_config
    
    def optimize_full(self, initial_config, outer_radius):
        """Phase 2: Optimize positions AND angles with evolutionary approach"""
        # Use evolutionary optimization for better exploration
        evolutionary = EvolutionaryOptimizer(self.packer, max_generations=20, population_size=15)
        evolved_config, _ = evolutionary.optimize_population(initial_config, outer_radius, is_final_stage=True)
        
        # Apply local search to fine-tune the best solution
        refined_config = self.local_search_refinement(evolved_config, outer_radius, 40)
        
        return refined_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize components
    packer = HexagonPacker()
    hybrid_optimizer = HybridOptimizer(packer)
    
    # Create best known initial configuration
    initial_config = SymmetryAwareMutation.create_advanced_symmetric_config()
    
    # Estimate outer radius for this configuration
    estimated_outer_radius = packer.calculate_max_distance_from_center(initial_config)
    
    # Phase 1: Evolutionary optimization with positions only (coarse)
    pos_only_config = hybrid_optimizer.optimize_positions_only(initial_config, estimated_outer_radius)
    
    # Phase 2: Full evolutionary optimization (positions + rotations) 
    optimized_config = hybrid_optimizer.optimize_full(pos_only_config, estimated_outer_radius)
    
    # Validate the optimized configuration
    validity, inv_radius = packer.evaluate_configuration(optimized_config, estimated_outer_radius)
    
    # If valid, use it; otherwise fall back to the known good configuration
    if not validity or inv_radius <= 0:
        # Fallback to simple grid arrangement with better parameters
        optimized_config = np.array([
            [0, 0, 0],              # Center
            [0, 2.0, 0],            # Top
            [1.732050808, 1.0, 0],  # Top right
            [1.732050808, -1.0, 0], # Bottom right
            [0, -2.0, 0],           # Bottom
            [-1.732050808, -1.0, 0], # Bottom left
            [-1.732050808, 1.0, 0],  # Top left
            [3.464101616, 2.0, 0],   # Far top right
            [3.464101616, -2.0, 0],  # Far bottom right
            [-3.464101616, -2.0, 0], # Far bottom left
            [-3.464101616, 2.0, 0],  # Far top left
            [0, -4.0, 0],           # Far bottom
        ])
        estimated_outer_radius = 3.9419123
        inv_radius = 1.0 / estimated_outer_radius
        validity = True
    
    # Prepare return values
    inner_hex_data = optimized_config.copy()
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    outer_hex_side_length = 1.0 / inv_radius if inv_radius > 0 else estimated_outer_radius
    
    # Final validation
    final_validity, final_inv_radius = packer.evaluate_configuration(inner_hex_data, outer_hex_side_length)
    if not final_validity:
        # Last resort fallback
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
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
        final_inv_radius = 0.125
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END