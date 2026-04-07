# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
import random
from scipy.spatial.distance import cdist

# Constants for unit hexagon geometry
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_VERTEX_ANGLE = np.pi / 3
SQRT_3 = np.sqrt(3)

def create_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Create vertices of a unit regular hexagon centered at center with given rotation."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEX_VERTEX_ANGLE
        x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
        y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def compute_outer_hexagon_vertices(center=(0,0), side_length=1.0, rotation=0):
    """Create vertices of the outer hexagon."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEX_VERTEX_ANGLE
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_hexagon_containment(inner_vertices, outer_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    inner_polygon = Polygon(inner_vertices)
    outer_polygon = Polygon(outer_vertices)
    return outer_polygon.contains(inner_polygon)

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def evaluate_configuration_fast(config):
    """Fast evaluation of configuration with early rejection of invalid states."""
    # Config: 36 position/angle values + 1 outer side length = 37 total
    n = 12
    positions = config[:3*n].reshape(n, 3)
    outer_side_length = config[3*n]
    
    # Very small outer hexagon is invalid
    if outer_side_length < 1.0:
        return False, 0.0
    
    # Compute outer hexagon vertices once
    outer_hex_vertices = compute_outer_hexagon_vertices((0,0), outer_side_length)
    
    # Check containment and overlap in single pass
    for i in range(n):
        center_x, center_y, angle = positions[i]
        hex_vertices = create_unit_hexagon_vertices((center_x, center_y), np.radians(angle))
        
        # Check containment first (early rejection)
        if not check_hexagon_containment(hex_vertices, outer_hex_vertices):
            return False, 0.0
            
        # Check overlap with all previous hexagons (symmetry-aware)
        for j in range(i):
            prev_center_x, prev_center_y, prev_angle = positions[j]
            prev_hex_vertices = create_unit_hexagon_vertices((prev_center_x, prev_center_y), np.radians(prev_angle))
            
            if check_hexagon_overlap(hex_vertices, prev_hex_vertices):
                return False, 0.0
                
    # If we get here, all checks passed
    return True, outer_side_length

def objective_function(config):
    """Objective function to minimize (negative inverse of outer hexagon side length)."""
    valid, outer_side_length = evaluate_configuration_fast(config)
    
    if not valid:
        # Heavy penalty for invalid configurations
        return 1e10
    
    # Return negative inverse (since we want to maximize 1/R)
    return -1.0 / outer_side_length

def generate_initial_symmetric_config():
    """Generate a highly symmetric initial configuration based on hexagonal lattice."""
    config = []
    
    # Central hexagon
    config.extend([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons around center
    for i in range(6):
        angle = i * UNIT_HEX_VERTEX_ANGLE
        x = SQRT_3 * np.cos(angle)  # Circumradius for adjacent hexagons
        y = SQRT_3 * np.sin(angle)
        config.extend([x, y, 0.0])
    
    # Second ring: 6 hexagons forming a larger ring
    for i in range(6):
        angle = i * UNIT_HEX_VERTEX_ANGLE + UNIT_HEX_VERTEX_ANGLE / 2
        x = 2 * SQRT_3 * np.cos(angle)
        y = 2 * SQRT_3 * np.sin(angle)
        config.extend([x, y, 0.0])
    
    # Add one extra hexagon to make 12 total
    config.extend([0.0, 2 * SQRT_3, 0.0])
    
    # Add outer side length parameter (start with larger value to allow optimization)
    config.append(6.0)
    
    return np.array(config)

def generate_alternative_config():
    """Generate an alternative high-quality initial configuration."""
    config = []
    
    # Layer 1: Center
    config.extend([0.0, 0.0, 0.0])
    
    # Layer 2: Ring of 6 hexagons
    for i in range(6):
        angle = i * UNIT_HEX_VERTEX_ANGLE
        x = 2.0 * UNIT_HEX_RADIUS * np.cos(angle)
        y = 2.0 * UNIT_HEX_RADIUS * np.sin(angle)
        config.extend([x, y, 0.0])
    
    # Layer 3: Ring of 5 hexagons
    for i in range(5):
        angle = i * UNIT_HEX_VERTEX_ANGLE + UNIT_HEX_VERTEX_ANGLE / 2
        x = 3.0 * UNIT_HEX_RADIUS * np.cos(angle)
        y = 3.0 * UNIT_HEX_RADIUS * np.sin(angle)
        config.extend([x, y, 0.0])
    
    # Add one extra hexagon
    config.extend([0.0, -3.0 * UNIT_HEX_RADIUS, 0.0])
    
    # Outer side length
    config.append(7.0)
    
    return np.array(config)

def generate_kagome_config():
    """Generate a configuration based on Kagome lattice pattern."""
    config = []
    
    # Central hexagon
    config.extend([0.0, 0.0, 0.0])
    
    # Surrounding hexagons in a Kagome-like pattern
    angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    for i, angle in enumerate(angles):
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        config.extend([x, y, 0.0])
    
    # Additional layer using hexagonal close packing arrangement
    angles2 = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6]
    for i, angle in enumerate(angles2):
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        config.extend([x, y, 0.0])
        
    # Extra hexagon in center of outer ring
    config.extend([0.0, 3.0, 0.0])
    
    # Outer side length
    config.append(8.0)
    
    return np.array(config)

def generate_random_config():
    """Generate a random but plausible initial configuration."""
    config = []
    
    # Generate random positions within a reasonable range
    for i in range(12):
        # Position with some clustering around center
        angle = random.uniform(0, 2*np.pi)
        radius = random.uniform(0, 3.0)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        angle_deg = random.uniform(0, 360)
        config.extend([x, y, angle_deg])
    
    # Add outer side length parameter
    config.append(random.uniform(5.0, 10.0))
    
    return np.array(config)

def generate_fallback_config():
    """Generate a fallback configuration when optimization fails."""
    inner_hex_data = np.array([
        [0, 0, 0],          # center
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
        [0, -4, 0],         # far bottom-center
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

class EvolutionaryOptimizer:
    """Advanced evolutionary optimizer with adaptive parameters and hybrid techniques."""
    
    def __init__(self):
        self.population_size = 20
        self.elite_count = 3
        self.max_generations = 60
        self.initial_mutation_strength = 0.8
        self.mutation_decay_factor = 0.97
        self.temperature_start = 0.8
        self.temperature_end = 0.01
        self.cooling_rate = (self.temperature_end / self.temperature_start) ** (1.0 / self.max_generations)
        self.convergence_threshold = 1e-6
        self.convergence_window = 10
        self.best_fitness_history = []
        self.timeout = 170  # Leave 10 seconds for cleanup

    def adaptive_mutation_strength(self, generation):
        """Calculate current mutation strength based on generation."""
        return self.initial_mutation_strength * (self.mutation_decay_factor ** generation)

    def generate_population(self, base_configs):
        """Generate initial population around base configurations."""
        population = []
        for base_config in base_configs:
            population.append(base_config.copy())
            # Add variants with small random perturbations
            for _ in range(int(self.population_size/len(base_configs)) - 1):
                variant = base_config.copy()
                # Apply small Gaussian perturbations
                for i in range(len(variant)):
                    if i < 36:  # Position and angle parameters
                        if random.random() < 0.4:  # 40% chance to perturb
                            variant[i] += np.random.normal(0, 0.1)
                            if i % 3 == 2:  # angle parameter
                                variant[i] = variant[i] % 360
                    else:  # outer side length parameter
                        if random.random() < 0.3:  # 30% chance to perturb
                            variant[i] = max(1.0, variant[i] + np.random.normal(0, 0.1))
                population.append(variant)
        return population[:self.population_size]

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_index]

    def hybrid_evolutionary_search(self, base_configs, start_time):
        """Perform hybrid evolutionary-search with simulated annealing."""
        # Initialize population
        population = self.generate_population(base_configs)
        
        # Track best solution
        best_individual = None
        best_fitness = float('inf')
        
        # Convergence tracking
        recent_improvements = []
        
        for generation in range(self.max_generations):
            if time.time() - start_time > self.timeout:
                break
                
            # Evaluate fitness of entire population
            fitness_scores = []
            for individual in population:
                fitness = objective_function(individual)
                fitness_scores.append(fitness)
            
            # Sort population by fitness (lower is better)
            sorted_indices = np.argsort(fitness_scores)
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Update best individual
            if fitness_scores[0] < best_fitness:
                best_individual = population[0].copy()
                best_fitness = fitness_scores[0]
            
            # Track improvements for convergence detection
            recent_improvements.append(fitness_scores[0])
            if len(recent_improvements) > self.convergence_window:
                recent_improvements.pop(0)
            
            # Check for convergence
            if len(recent_improvements) >= self.convergence_window:
                improvement = abs(recent_improvements[-1] - recent_improvements[0])
                if improvement < self.convergence_threshold:
                    # Reduce mutation strength if converged
                    self.initial_mutation_strength *= 0.8
            
            # Create new population
            new_population = population[:self.elite_count]  # Elitism
            
            # Generate offspring through tournament selection and mutation
            while len(new_population) < self.population_size:
                # Select parent via tournament
                parent = self.tournament_selection(population, fitness_scores)
                
                # Apply adaptive mutation
                current_mutation_strength = self.adaptive_mutation_strength(generation)
                mutated = parent.copy()
                
                for i in range(len(mutated)):
                    if random.random() < 0.3:  # 30% mutation rate
                        if i < 36:  # Position and angle parameters
                            delta = np.random.normal(0, current_mutation_strength * 0.5)
                            mutated[i] += delta
                            if i % 3 == 2:  # angle parameter
                                mutated[i] = mutated[i] % 360
                        else:  # outer side length parameter
                            delta = np.random.normal(0, current_mutation_strength)
                            mutated[i] = max(1.0, mutated[i] + delta)
                
                # Accept with probability based on temperature and fitness difference
                current_fitness = objective_function(mutated)
                if current_fitness < fitness_scores[0]:  # Always accept better solutions
                    new_population.append(mutated)
                else:  # Accept worse solutions with prob based on SA
                    fitness_diff = current_fitness - fitness_scores[0]
                    current_temp = self.temperature_start * (self.cooling_rate ** generation)
                    acceptance_prob = np.exp(-fitness_diff / max(current_temp, 1e-10))
                    if random.random() < acceptance_prob:
                        new_population.append(mutated)
            
            population = new_population
            
            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.8f}")

        return best_individual, best_fitness

    def optimize_with_stages(self, base_configs, start_time):
        """Perform multi-stage optimization with hybrid approach."""
        # Stage 1: Hybrid evolutionary search with symmetry preservation
        print("Stage 1: Hybrid evolutionary search...")
        stage1_result, _ = self.hybrid_evolutionary_search(base_configs, start_time)
        
        # Stage 2: Fine-tune with local optimization
        print("Stage 2: Local optimization with L-BFGS...")
        try:
            bounds_final = [(None, None)] * 36
            bounds_final.extend([(1.0, 15.0)])
            
            result_final = minimize(
                objective_function,
                stage1_result,
                method='L-BFGSB',
                bounds=bounds_final,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-8
            )
            
            if result_final.success:
                stage2_result = result_final.x
            else:
                stage2_result = stage1_result
                
        except Exception as e:
            print(f"L-BFGS optimization failed: {e}")
            stage2_result = stage1_result
            
        return stage2_result

def optimize_with_evolution():
    """Perform evolutionary optimization with multiple starting configurations."""
    start_time = time.time()
    optimizer = EvolutionaryOptimizer()
    
    # Generate diversified starting configurations
    base_configs = [
        generate_initial_symmetric_config(),
        generate_alternative_config(),
        generate_kagome_config(),
        generate_random_config(),
        generate_random_config()
    ]
    
    # Run hybrid optimization
    final_result = optimizer.optimize_with_stages(base_configs, start_time)
    
    return final_result

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
        final_config = optimize_with_evolution()

        # Extract the final configuration
        final_positions = final_config[:-1].reshape(12, 3)
        final_side_length = final_config[-1]

        # Return in the required format
        inner_hex_data = final_positions.copy()
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered

        # Validate final result
        valid, _ = evaluate_configuration_fast(final_config)
        if not valid:
            print("Warning: Final configuration may be invalid")
            # Fallback to a better-known configuration
            inner_hex_data, outer_hex_data, final_side_length = generate_fallback_config()

    except Exception as e:
        # Fallback to old method if anything goes wrong
        print(f"Optimization error: {e}")
        inner_hex_data, outer_hex_data, final_side_length = generate_fallback_config()

    end_time = time.time()

    # Calculate performance metrics
    inv_outer_hex_side_length = 1.0 / final_side_length if final_side_length > 0 else 0.0
    benchmark_ratio = inv_outer_hex_side_length / 0.2537

    print(f"Optimized result: inverse_side_length={inv_outer_hex_side_length:.6f}, "
          f"benchmark_ratio={benchmark_ratio:.6f}, eval_time={(end_time-start_time):.3f}s")

    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END