# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
import random
from scipy.spatial.distance import cdist
import math

# Constants for unit hexagon geometry
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3) / 2
UNIT_HEX_VERTEX_ANGLE = np.pi / 3
SQRT_3 = np.sqrt(3)

class LatticeHexagonOptimizer:
    """Implements lattice-based optimization for hexagon packing using geometric constraints and symmetry preservation"""
    
    def __init__(self):
        self.best_score = 0.0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds
        
    @staticmethod
    def create_unit_hexagon_vertices(center=(0,0), rotation=0):
        """Create vertices of a unit regular hexagon centered at center with given rotation."""
        vertices = []
        for i in range(6):
            angle = rotation + i * UNIT_HEX_VERTEX_ANGLE
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    @staticmethod
    def compute_outer_hexagon_vertices(center=(0,0), side_length=1.0, rotation=0):
        """Create vertices of the outer hexagon."""
        vertices = []
        for i in range(6):
            angle = rotation + i * UNIT_HEX_VERTEX_ANGLE
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    @staticmethod
    def check_hexagon_containment(inner_vertices, outer_vertices):
        """Check if all vertices of inner hexagon are within outer hexagon using vectorized operations."""
        inner_polygon = Polygon(inner_vertices)
        outer_polygon = Polygon(outer_vertices)
        return outer_polygon.contains(inner_polygon)

    @staticmethod
    def check_hexagon_overlap(hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)

    @staticmethod
    def compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y):
        """
        Calculate minimum radius needed for outer hexagon to contain all inner hexagons
        by checking maximum distance from center to any vertex of any hexagon
        """
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            hex_vertices = LatticeHexagonOptimizer.create_unit_hexagon_vertices((center_x, center_y), np.radians(angle))
            for vertex in hex_vertices:
                dist = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
                max_dist = max(max_dist, dist)

        # Add buffer for numerical precision
        return max_dist * 1.01

    @staticmethod
    def fast_evaluate_config(inner_hex_data, outer_side_length):
        """Fast evaluation with early rejection and minimal overhead"""
        # Quick validation
        if len(inner_hex_data) != 12:
            return False, 0.0
            
        # Early distance check for feasibility
        positions = np.array([[h[0], h[1]] for h in inner_hex_data])
        distances = cdist(positions, positions)
        min_dist = np.min(distances[distances > 0])
        if min_dist < 1.8:  # Conservative overlap check
            return False, 0.0
            
        # Compute outer hexagon vertices once
        outer_hex_vertices = LatticeHexagonOptimizer.compute_outer_hexagon_vertices((0,0), outer_side_length)
        
        # Check all constraints in one pass
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            hex_vertices = LatticeHexagonOptimizer.create_unit_hexagon_vertices((center_x, center_y), np.radians(angle))
            
            # Early containment check
            if not LatticeHexagonOptimizer.check_hexagon_containment(hex_vertices, outer_hex_vertices):
                return False, 0.0
                
            # Overlap check with previous hexagons (symmetry-aware)
            for j in range(i):
                prev_center_x, prev_center_y, prev_angle = inner_hex_data[j]
                prev_hex_vertices = LatticeHexagonOptimizer.create_unit_hexagon_vertices((prev_center_x, prev_center_y), np.radians(prev_angle))
                
                if LatticeHexagonOptimizer.check_hexagon_overlap(hex_vertices, prev_hex_vertices):
                    return False, 0.0
                    
        return True, outer_side_length

    @staticmethod
    def generate_lattice_configurations():
        """Generate mathematical lattice-based starting configurations"""
        configs = []
        
        # Kagome lattice pattern (dense packing)
        kagome_config = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom
            [SQRT_3, 1.0, 0],    # top-right
            [-SQRT_3, 1.0, 0],   # top-left
            [SQRT_3, -1.0, 0],   # bottom-right
            [-SQRT_3, -1.0, 0],  # bottom-left
            [2*SQRT_3, 0, 0],    # far right
            [-2*SQRT_3, 0, 0],   # far left
            [SQRT_3, 3.0, 0],    # upper right corner
            [-SQRT_3, 3.0, 0],   # upper left corner
            [SQRT_3, -3.0, 0],   # lower right corner
            [-SQRT_3, -3.0, 0],  # lower left corner
        ])
        configs.append(kagome_config[:12])
        
        # HCP-like pattern (hexagonal close-packed)
        hcp_config = np.array([
            [0, 0, 0],           # center
            [0, 1.8, 0],         # top
            [0, -1.8, 0],        # bottom
            [SQRT_3 * 0.9, 0.9, 0],    # top-right
            [-SQRT_3 * 0.9, 0.9, 0],   # top-left
            [SQRT_3 * 0.9, -0.9, 0],   # bottom-right
            [-SQRT_3 * 0.9, -0.9, 0],  # bottom-left
            [SQRT_3 * 1.8, 0, 0],      # far right
            [-SQRT_3 * 1.8, 0, 0],     # far left
            [SQRT_3 * 0.9, 2.7, 0],    # upper right corner
            [-SQRT_3 * 0.9, 2.7, 0],   # upper left corner
            [SQRT_3 * 0.9, -2.7, 0],   # lower right corner
            [-SQRT_3 * 0.9, -2.7, 0],  # lower left corner
        ])
        configs.append(hcp_config[:12])
        
        # Hexagonal cluster pattern
        cluster_config = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom
            [SQRT_3, 1.0, 0],    # top-right
            [-SQRT_3, 1.0, 0],   # top-left
            [SQRT_3, -1.0, 0],   # bottom-right
            [-SQRT_3, -1.0, 0],  # bottom-left
            [2*SQRT_3, 0, 0],    # far right
            [-2*SQRT_3, 0, 0],   # far left
            [SQRT_3, 3.0, 0],    # upper right corner
            [-SQRT_3, 3.0, 0],   # upper left corner
            [SQRT_3, -3.0, 0],   # lower right corner
            [-SQRT_3, -3.0, 0],  # lower left corner
        ])
        configs.append(cluster_config[:12])
        
        return configs

    def adaptive_mutate(self, individual, mutation_strength=0.2, stage=1):
        """Mutation that respects hexagonal symmetries and lattice structure"""
        mutated = individual.copy()
        
        # Stage-dependent mutation parameters
        if stage == 1:  # Exploratory phase
            pos_mutation = mutation_strength * 2.0
            rot_mutation = 30
        elif stage == 2:  # Exploitative phase  
            pos_mutation = mutation_strength * 1.0
            rot_mutation = 15
        else:  # Refinement phase
            pos_mutation = mutation_strength * 0.5
            rot_mutation = 5
            
        # Apply mutations respecting symmetry groups
        # Central hexagon
        mutated[0][0] += random.uniform(-pos_mutation, pos_mutation)
        mutated[0][1] += random.uniform(-pos_mutation, pos_mutation)
        
        # First ring (6 hexagons) - maintain 6-fold symmetry 
        for i in range(1, 7):
            mutated[i][0] += random.uniform(-pos_mutation, pos_mutation)
            mutated[i][1] += random.uniform(-pos_mutation, pos_mutation)
            if random.random() < 0.3:
                mutated[i][2] += random.uniform(-rot_mutation, rot_mutation)
                
        # Second ring (5 hexagons) - maintain approximate symmetry
        for i in range(7, 12):
            mutated[i][0] += random.uniform(-pos_mutation, pos_mutation)
            mutated[i][1] += random.uniform(-pos_mutation, pos_mutation)
            if random.random() < 0.4:
                mutated[i][2] += random.uniform(-rot_mutation, rot_mutation)
                
        # Normalize rotations
        for i in range(12):
            mutated[i][2] %= 360
            
        return mutated

    def optimize_stage(self, initial_config, stage=1):
        """Optimize a single stage using adaptive evolutionary approach"""
        # Stage settings
        if stage == 1:
            pop_size = 25
            gens = 20
            mut_strength = 0.3
        elif stage == 2:
            pop_size = 20
            gens = 25
            mut_strength = 0.15
        else:
            pop_size = 15
            gens = 30
            mut_strength = 0.05
            
        # Initialize population
        population = [initial_config.copy()]
        for _ in range(pop_size - 1):
            variant = initial_config.copy()
            # Add systematic perturbations
            for i in range(12):
                variant[i][0] += random.gauss(0, 0.05)
                variant[i][1] += random.gauss(0, 0.05)
            population.append(variant)
            
        # Evolution loop
        for gen in range(gens):
            if time.time() - self.start_time > self.timeout * 0.8:
                break
                
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                # Convert to tuple format for compatibility
                individual_tuple = [tuple(row) for row in individual]
                valid, side_length = self.fast_evaluate_config(individual_tuple, 10.0)
                if valid:
                    fitness = 1.0 / side_length
                else:
                    fitness = -1e10
                fitness_scores.append(fitness)
                
            # Selection & elitism
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = max(1, pop_size // 3)
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]
            
            # Generate new population
            new_population = elite.copy()
            while len(new_population) < pop_size:
                parent = random.choice(elite)
                mutated = self.adaptive_mutate(parent, mut_strength, stage)
                new_population.append(mutated)
                
            population = new_population
            
            # Update best
            for individual in population:
                individual_tuple = [tuple(row) for row in individual]
                valid, side_length = self.fast_evaluate_config(individual_tuple, 10.0)
                if valid:
                    fitness = 1.0 / side_length
                    if fitness > self.best_score:
                        self.best_score = fitness
                        self.best_config = individual.copy()
                        
        return self.best_config, self.best_score

    def refine_with_gradient(self, config):
        """Use gradient-based optimization for final refinement"""
        def objective_func(params):
            positions = params.reshape(-1, 2)
            temp_config = config.copy()
            temp_config[:, 0] = positions[:, 0]
            temp_config[:, 1] = positions[:, 1]
            
            # Fast evaluation
            valid, side_length = self.fast_evaluate_config([tuple(row) for row in temp_config], 10.0)
            if valid:
                return side_length
            else:
                return 1e10
                
        try:
            # Flatten positions
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()
            
            # Optimize with bounds
            result = minimize(
                objective_func,
                initial_positions,
                method='L-BFGS-B',
                bounds=[(-5, 5) for _ in range(24)],
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]
                
        except:
            pass
            
        return config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize optimizer
    optimizer = LatticeHexagonOptimizer()
    
    # Generate starting configurations from mathematical lattices
    configs = LatticeHexagonOptimizer.generate_lattice_configurations()
    
    # Find best starting configuration
    best_initial_score = 0.0
    best_initial_config = None
    
    for config in configs:
        # Test configuration
        config_tuple = [tuple(row) for row in config]
        valid, side_length = LatticeHexagonOptimizer.fast_evaluate_config(config_tuple, 10.0)
        if valid:
            score = 1.0 / side_length
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()
    
    # Set up optimization tracking
    optimizer.best_score = best_initial_score
    optimizer.best_config = best_initial_config.copy()
    
    # Phase 1: Global lattice optimization (position only)
    print("Phase 1: Global lattice optimization...")
    coarse_config = best_initial_config.copy()
    for i in range(12):
        coarse_config[i][2] = 0  # Fix rotations initially
        
    evolved_config, _ = optimizer.optimize_stage(coarse_config, stage=1)
    
    # Phase 2: Local refinement with rotation awareness
    print("Phase 2: Local refinement with rotation...")
    refined_config = evolved_config.copy()
    # Add small random rotations to enhance packing
    for i in range(12):
        if random.random() < 0.3:
            refined_config[i][2] += random.uniform(-10, 10)
            
    rotated_config, _ = optimizer.optimize_stage(refined_config, stage=2)
    
    # Phase 3: Final gradient refinement
    print("Phase 3: Final gradient refinement...")  
    final_config = optimizer.refine_with_gradient(rotated_config)
    
    # Final evaluation
    final_config_tuple = [tuple(row) for row in final_config]
    valid, side_length = LatticeHexagonOptimizer.fast_evaluate_config(final_config_tuple, 10.0)
    if not valid:
        # Fallback to initial configuration if final failed
        final_config = best_initial_config
        final_config_tuple = [tuple(row) for row in final_config]
        valid, side_length = LatticeHexagonOptimizer.fast_evaluate_config(final_config_tuple, 10.0)
        
    # Calculate metrics
    inv_outer_hex_side_length = 1.0 / side_length if side_length > 0 else 0.0
    benchmark_ratio = inv_outer_hex_side_length / 0.2537
    
    end_time = time.time()
    
    print(f"Optimized result: inverse_side_length={inv_outer_hex_side_length:.6f}, "
          f"benchmark_ratio={benchmark_ratio:.6f}, eval_time={(end_time-start_time):.3f}s")
          
    # Return in required format
    inner_hex_data = final_config.copy()
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered
    
    return inner_hex_data, outer_hex_data, side_length

# EVOLVE-BLOCK-END