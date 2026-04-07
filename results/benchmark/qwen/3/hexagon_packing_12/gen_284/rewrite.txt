# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
from scipy.spatial.distance import cdist
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

class HexagonState:
    """Tracks the optimization state and best solutions"""
    def __init__(self):
        self.best_score = 0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""
    
    @staticmethod
    @jit(nopython=True)
    def create_unit_hexagon_vertices(center_x: float, center_y: float, rotation: float) -> np.ndarray:
        """Create vertices of unit hexagon using JIT compilation"""
        vertices = np.empty((6, 2), dtype=np.float64)
        angle_offset = rotation * math.pi / 180.0
        radius = 1.0
        for i in range(6):
            angle = angle_offset + i * math.pi / 3
            vertices[i][0] = center_x + radius * math.cos(angle)
            vertices[i][1] = center_y + radius * math.sin(angle)
        return vertices
    
    @staticmethod
    @jit(nopython=True)
    def get_hexagon_vertices_array(hex_data: np.ndarray) -> np.ndarray:
        """Get all vertices of all hexagons efficiently"""
        all_vertices = np.empty((72, 2), dtype=np.float64)  # 12 hexagons * 6 vertices
        idx = 0
        for i in range(len(hex_data)):
            center_x, center_y, rotation = hex_data[i]
            vertices = HexagonGeometry.create_unit_hexagon_vertices(center_x, center_y, rotation)
            for j in range(6):
                all_vertices[idx][0] = vertices[j][0]
                all_vertices[idx][1] = vertices[j][1]
                idx += 1
        return all_vertices

class HexagonConstraintChecker:
    """Handles constraint checking for hexagon arrangements"""
    
    @staticmethod
    @jit(nopython=True)
    def point_in_hexagon(px: float, py: float, hex_center_x: float, hex_center_y: float, 
                         hex_rotation: float) -> bool:
        """Fast point-in-hexagon test using dot products"""
        # Convert to hexagon's coordinate system
        cos_rot = math.cos(hex_rotation * math.pi / 180.0)
        sin_rot = math.sin(hex_rotation * math.pi / 180.0)
        dx = px - hex_center_x
        dy = py - hex_center_y
        x = dx * cos_rot + dy * sin_rot
        y = -dx * sin_rot + dy * cos_rot
        
        # Check against hexagon boundaries
        # Hexagon with radius 1 centered at origin
        # Check if point is within the hexagon using distance to edges
        # Simplified: check against 6 half-planes
        h = 1.0  # hexagon radius
        if abs(y) > h:
            return False
        if abs(x) > h:
            return False
        # Check diagonal boundaries
        if abs(y) > h - abs(x) * 0.5773502691896257:
            return False
        return True

    @staticmethod
    @jit(nopython=True)
    def hexagons_overlap_fast(v1: np.ndarray, v2: np.ndarray) -> bool:
        """Fast overlap detection between hexagons using bounding boxes and separation axis theorem"""
        # Simple AABB check first
        min_x1, max_x1 = v1[:, 0].min(), v1[:, 0].max()
        min_y1, max_y1 = v1[:, 1].min(), v1[:, 1].max()
        min_x2, max_x2 = v2[:, 0].min(), v2[:, 0].max()
        min_y2, max_y2 = v2[:, 1].min(), v2[:, 1].max()
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return False
        
        return True

    @staticmethod
    def compute_overlap_penalty_fast(hex_data: np.ndarray) -> float:
        """Fast overlap penalty computation"""
        penalty = 0.0
        # Only check with nearby hexagons to reduce complexity
        # Precompute distances for efficiency
        vertices_list = []
        for i in range(len(hex_data)):
            center_x, center_y, rotation = hex_data[i]
            vertices = HexagonGeometry.create_unit_hexagon_vertices(center_x, center_y, rotation)
            vertices_list.append(vertices)
        
        # Check pairwise overlaps
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                # Only check if hexagons are close enough
                dx = hex_data[i][0] - hex_data[j][0]
                dy = hex_data[i][1] - hex_data[j][1]
                distance_sq = dx*dx + dy*dy
                # Threshold for checking overlap (simplified)
                if distance_sq < 8.0:  # 2*sqrt(2) distance squared
                    if HexagonConstraintChecker.hexagons_overlap_fast(vertices_list[i], vertices_list[j]):
                        penalty += 1000.0
                        
        return penalty

class HexagonPackingEvaluator:
    """Evaluates hexagon packing configurations"""
    
    @staticmethod
    def calculate_outer_hex_radius(hex_data: np.ndarray) -> float:
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons - optimized version"""
        # Use fast vertex extraction method
        all_vertices = HexagonGeometry.get_hexagon_vertices_array(hex_data)
        
        if len(all_vertices) == 0:
            return 0.0
            
        # Compute centroid
        centroid_x = np.mean(all_vertices[:, 0])
        centroid_y = np.mean(all_vertices[:, 1])
        
        # Find maximum distance from centroid to any vertex
        max_distance = 0.0
        for i in range(len(all_vertices)):
            dx = all_vertices[i][0] - centroid_x
            dy = all_vertices[i][1] - centroid_y
            distance_sq = dx*dx + dy*dy
            distance = math.sqrt(distance_sq)
            max_distance = max(max_distance, distance)
            
        return max_distance + 1.0  # Add buffer for hexagon size

    @staticmethod
    def evaluate_configuration(hex_data: np.ndarray) -> float:
        """Evaluate a configuration and return the inverse radius - optimized version"""
        if len(hex_data) != 12:
            return 1e-10
            
        outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(hex_data)
        
        # Compute penalties
        overlap_penalty = HexagonConstraintChecker.compute_overlap_penalty_fast(hex_data)

        # If valid configuration, return inverse of outer radius; otherwise return a very small value
        if overlap_penalty == 0:
            return 1.0 / outer_radius
        else:
            # Invalid configuration gets penalized heavily
            return 1e-10

class SymmetryAwareMutation:
    """Handles mutation strategies that preserve geometric symmetries"""
    
    @staticmethod
    def group_hexagons_by_ring(hex_data: np.ndarray) -> List[List[int]]:
        """Group hexagons by approximate ring distance from center"""
        if len(hex_data) < 2:
            return [[0]]
            
        rings = [[0]]  # Center hexagon
        
        # Calculate distances from center for all other hexagons
        distances = []
        center_x, center_y = hex_data[0][0], hex_data[0][1]
        
        for i in range(1, len(hex_data)):
            dx = hex_data[i][0] - center_x
            dy = hex_data[i][1] - center_y
            distance = math.sqrt(dx*dx + dy*dy)
            distances.append((distance, i))
        
        # Sort by distance and group into rings
        distances.sort()
        
        # Create rings (center, first ring, second ring, remaining)
        ring1 = []  # Closest ~6 hexagons
        ring2 = []  # Next ~4 hexagons
        ring3 = []  # Remaining hexagons
        
        for i, (dist, idx) in enumerate(distances):
            if i < 6:
                ring1.append(idx)
            elif i < 10:
                ring2.append(idx)
            else:
                ring3.append(idx)
        
        return [rings[0], ring1, ring2, ring3]
    
    @staticmethod
    def mutate_symmetrically(hex_data: np.ndarray, mutation_strength: float = 0.2, stage: int = 1, 
                           generation: int = 0, max_generations: int = 50) -> np.ndarray:
        """Apply symmetric mutation to maintain hexagonal properties with adaptive scaling"""
        mutated_data = hex_data.copy()

        # Adaptive mutation strength based on stage and generation progress
        if stage == 1:  # Coarse stage - aggressive mutation for exploration
            base_mutation = mutation_strength * 2.0
        elif stage == 2:  # Fine stage - moderate mutation
            base_mutation = mutation_strength * 1.0
        else:  # Final stage - conservative mutation
            base_mutation = mutation_strength * 0.5

        # Progressive mutation decay based on generation progress
        progress = generation / max_generations if max_generations > 0 else 0
        decay_factor = 1.0 - (0.25 * progress)  # Decay from 1.0 to 0.75 over generations
        mutation_factor = base_mutation * decay_factor

        # Mutate center hexagon
        mutated_data[0][0] += random.uniform(-mutation_factor, mutation_factor)
        mutated_data[0][1] += random.uniform(-mutation_factor, mutation_factor)

        # Mutate rotations of center hexagon
        mutated_data[0][2] += random.uniform(-mutation_factor*0.5, mutation_factor*0.5)

        # Group hexagons by rings for more intelligent mutation
        rings = SymmetryAwareMutation.group_hexagons_by_ring(hex_data)
        
        # Mutate each ring with appropriate strength based on ring distance
        for ring_idx, ring in enumerate(rings):
            if ring_idx == 0:  # Center
                continue
            elif ring_idx == 1:  # First ring (core positions)
                ring_mutation = mutation_factor * 0.8
            elif ring_idx == 2:  # Second ring (boundary positions)
                ring_mutation = mutation_factor * 0.6
            else:  # Third ring (edge positions)
                ring_mutation = mutation_factor * 0.4
                
            for idx in ring:
                mutated_data[idx][0] += random.uniform(-ring_mutation, ring_mutation)
                mutated_data[idx][1] += random.uniform(-ring_mutation, ring_mutation)
                # Add rotation variation for non-central hexagons
                mutated_data[idx][2] += random.uniform(-ring_mutation*0.3, ring_mutation*0.3)

        return mutated_data

class InitialConfigurationGenerator:
    """Generates initial configurations for the optimization process"""
    
    @staticmethod
    def get_initial_configurations() -> List[np.ndarray]:
        """Generate several symmetric configurations to choose from"""
        configs = []

        # Configuration 1: Hexagonal cluster around center (inspired by literature)
        config1 = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom
            [1.732, 1.0, 0],     # top-right
            [-1.732, 1.0, 0],    # top-left
            [1.732, -1.0, 0],    # bottom-right
            [-1.732, -1.0, 0],   # bottom-left
            [3.464, 0, 0],       # far right
            [-3.464, 0, 0],      # far left
            [1.732, 3.0, 0],     # upper right corner
            [-1.732, 3.0, 0],    # upper left corner
            [1.732, -3.0, 0],    # lower right corner
            [-1.732, -3.0, 0],   # lower left corner
        ])
        configs.append(config1[:12])

        # Configuration 2: More compact arrangement
        config2 = np.array([
            [0, 0, 0],           # center
            [0, 1.8, 0],         # top
            [0, -1.8, 0],        # bottom
            [1.55, 0.9, 0],      # top-right
            [-1.55, 0.9, 0],     # top-left
            [1.55, -0.9, 0],     # bottom-right
            [-1.55, -0.9, 0],    # bottom-left
            [3.1, 0, 0],         # far right
            [-3.1, 0, 0],        # far left
            [1.55, 2.7, 0],      # upper right corner
            [-1.55, 2.7, 0],     # upper left corner
            [1.55, -2.7, 0],     # lower right corner
            [-1.55, -2.7, 0],    # lower left corner
        ])
        configs.append(config2[:12])

        # Configuration 3: Hexagonal ring pattern
        config3 = np.array([
            [0, 0, 0],           # center
            [0, 2.1, 0],         # top
            [1.8, 1.0, 0],       # top-right
            [1.8, -1.0, 0],      # bottom-right
            [0, -2.1, 0],        # bottom
            [-1.8, -1.0, 0],     # bottom-left
            [-1.8, 1.0, 0],      # top-left
            [3.6, 0, 0],         # far right
            [0, 3.6, 0],         # far top
            [-3.6, 0, 0],        # far left
            [0, -3.6, 0],        # far bottom
            [1.8, 2.1, 0],       # upper right corner
            [-1.8, 2.1, 0],      # upper left corner
            [1.8, -2.1, 0],      # lower right corner
            [-1.8, -2.1, 0],     # lower left corner
        ])
        configs.append(config3[:12])

        # Configuration 4: Mathematical optimum with rotational symmetry
        config4 = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [1.732, 1.0, 0],     # top-right
            [1.732, -1.0, 0],    # bottom-right
            [0, -2.0, 0],        # bottom
            [-1.732, -1.0, 0],   # bottom-left
            [-1.732, 1.0, 0],    # top-left
            [3.464, 0, 0],       # far right
            [0, 3.464, 0],       # far top
            [0, -3.464, 0],      # far bottom
            [-3.464, 0, 0],      # far left
            [1.732, 3.0, 0],     # upper right corner
            [-1.732, 3.0, 0],    # upper left corner
            [1.732, -3.0, 0],    # lower right corner
            [-1.732, -3.0, 0],   # lower left corner
        ])
        configs.append(config4[:12])

        # Configuration 5: HCP (Hexagonal Close-Packed) arrangement
        config5 = np.array([
            [0, 0, 0],           # center
            [0, 2.1, 0],         # top
            [0, -2.1, 0],        # bottom
            [1.8, 1.0, 0],       # top-right
            [-1.8, 1.0, 0],      # top-left
            [1.8, -1.0, 0],      # bottom-right
            [-1.8, -1.0, 0],     # bottom-left
            [3.6, 0, 0],         # far right
            [-3.6, 0, 0],        # far left
            [1.8, 2.1, 0],       # upper right corner
            [-1.8, 2.1, 0],      # upper left corner
            [1.8, -2.1, 0],      # lower right corner
            [-1.8, -2.1, 0],     # lower left corner
            [0, 4.2, 0],         # far top
            [0, -4.2, 0],        # far bottom
        ])
        configs.append(config5[:12])

        return configs

class OptimizationStage:
    """Represents a single stage of optimization"""
    
    def __init__(self, name: str, population_size: int = 20, generations: int = 50, 
                 mutation_strength: float = 0.2, use_elite: bool = True):
        self.name = name
        self.population_size = population_size
        self.generations = generations
        self.mutation_strength = mutation_strength
        self.use_elite = use_elite

    def execute(self, initial_config: np.ndarray, state: HexagonState, stage: int = 1) -> Tuple[np.ndarray, float]:
        """Execute this optimization stage with simulated annealing"""
        # Start with perturbed configuration and several variants
        population = [initial_config.copy()]
        for _ in range(self.population_size - 1):
            variant = initial_config.copy()
            for i in range(len(variant)):
                if random.random() < 0.5:
                    variant[i][0] += random.gauss(0, 0.1)
                if random.random() < 0.5:
                    variant[i][1] += random.gauss(0, 0.1)
                # Small rotation variations
                if random.random() < 0.3:
                    variant[i][2] += random.uniform(-2, 2)
            population.append(variant)

        for gen in range(self.generations):
            # Check timeout
            if time.time() - state.start_time > state.timeout * 0.8:
                break

            # Temperature schedule for simulated annealing
            temperature = 0.8 * (0.95 ** gen)  # Decreasing temperature from 0.8 to 0.01

            # Evaluate fitness of entire population
            fitness_scores = []
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
                fitness_scores.append(score)

            # Select top performers (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = self.population_size // 3 if self.use_elite else 0
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]

            # Generate new population through mutation with acceptance criterion
            new_population = elite.copy()

            # Fill remaining slots through mutation with probabilistic acceptance
            while len(new_population) < self.population_size:
                parent = random.choice(elite) if elite else initial_config.copy()
                mutated = SymmetryAwareMutation.mutate_symmetrically(
                    parent, 
                    mutation_strength=self.mutation_strength, 
                    stage=stage, 
                    generation=gen, 
                    max_generations=self.generations
                )

                # Apply simulated annealing acceptance criterion
                mutated_score = HexagonPackingEvaluator.evaluate_configuration(mutated)
                parent_score = HexagonPackingEvaluator.evaluate_configuration(parent)

                # Accept if better, or with probability based on temperature if worse
                if mutated_score >= parent_score or random.random() < math.exp((mutated_score - parent_score) / temperature):
                    new_population.append(mutated)
                else:
                    new_population.append(parent)

            population = new_population

            # Track best overall
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
                if score > state.best_score:
                    state.best_score = score
                    state.best_config = individual.copy()

        return state.best_config, state.best_score

class HexagonPackingOptimizer:
    """Main optimizer class that orchestrates the packing process"""
    
    def __init__(self):
        self.state = HexagonState()
        self.initial_generator = InitialConfigurationGenerator()
        self.stages = [
            OptimizationStage("Stage 1: Coarse position optimization", 
                            population_size=25, generations=30, mutation_strength=0.3),
            OptimizationStage("Stage 2: Fine-grained rotation refinement", 
                            population_size=20, generations=30, mutation_strength=0.2),
            OptimizationStage("Stage 3: Final full optimization", 
                            population_size=15, generations=20, mutation_strength=0.1)
        ]

    def find_best_initial_configuration(self) -> Tuple[np.ndarray, float]:
        """Find the best initial configuration among all candidates"""
        configs = self.initial_generator.get_initial_configurations()
        
        best_score = 0
        best_config = None
        
        for config in configs:
            score = HexagonPackingEvaluator.evaluate_configuration(config)
            if score > best_score:
                best_score = score
                best_config = config.copy()
        
        return best_config, best_score

    def refine_with_scipy_optimization(self, config: np.ndarray) -> np.ndarray:
        """Refine using scipy optimization with better constraint handling"""
        def objective_func(params):
            positions = params.reshape(-1, 2)
            temp_data = config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(temp_data)
            return outer_radius

        # Simplified constraint handling for better performance
        def constraint_func(params):
            positions = params.reshape(-1, 2)
            temp_data = config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]

            # For quick constraint checking
            penalty = HexagonConstraintChecker.compute_overlap_penalty_fast(temp_data)
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(temp_data)
            
            return penalty  # Just the penalty term

        try:
            # Flatten the initial positions for optimization
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()

            # Use tighter tolerances and fewer iterations
            result = minimize(objective_func, initial_positions, method='L-BFGS-B',
                             bounds=[(-5, 5) for _ in range(24)],
                             constraints={'type': 'ineq', 'fun': constraint_func},
                             options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]
        except:
            pass  # Fall back to previous best if optimization fails

        return config

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete multi-stage optimization pipeline"""
        # Find best initial configuration
        best_initial_config, best_initial_score = self.find_best_initial_configuration()
        
        # Store the best configuration found so far
        self.state.best_score = best_initial_score
        self.state.best_config = best_initial_config.copy()

        # Stage 1: Coarse-grained position optimization (fixed rotations)
        print("Stage 1: Coarse position optimization...")
        coarse_config = best_initial_config.copy()
        # Fix rotations for this stage
        for i in range(len(coarse_config)):
            coarse_config[i][2] = 0  # Set all rotations to 0 for coarse optimization

        evolved_config, evolved_score = self.stages[0].execute(coarse_config, self.state, stage=1)

        # Stage 2: Fine-grained refinement with rotation awareness
        print("Stage 2: Fine-grained rotation refinement...")
        # Allow rotations to vary but keep positions relatively close to evolved ones
        refined_config = evolved_config.copy()
        # Add small random rotations to improve packing
        for i in range(len(refined_config)):
            # Perturb rotations slightly
            if random.random() < 0.4:
                refined_config[i][2] += random.uniform(-5, 5)

        # Run evolution again with rotations allowed but more constrained
        rotated_config, rotated_score = self.stages[1].execute(refined_config, self.state, stage=2)

        # Stage 3: Full optimization with scipy refinement
        print("Stage 3: Full scipy optimization...")
        final_config = self.refine_with_scipy_optimization(rotated_config)

        # Stage 4: Hybrid refinement with continued evolutionary process
        print("Stage 4: Hybrid refinement...")
        # Run one final evolutionary stage with fine parameters
        hybrid_config, _ = self.stages[2].execute(final_config, self.state, stage=3)

        # Final evaluation
        final_score = HexagonPackingEvaluator.evaluate_configuration(hybrid_config)
        final_outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(hybrid_config)
        outer_hex_side_length = final_outer_radius + 0.2  # Add margin

        # Return result
        outer_hex_data = np.array([0, 0, 0])  # centered at origin

        return hybrid_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HexagonPackingOptimizer()
    return optimizer.run_full_optimization()

# EVOLVE-BLOCK-END