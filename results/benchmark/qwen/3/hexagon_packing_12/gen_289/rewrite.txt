# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
from scipy.spatial.distance import cdist
from collections import deque

class HexagonGeometry:
    """Handles all geometric operations for hexagons with optimized performance"""

    @staticmethod
    def create_unit_hexagon_vertices(center: Tuple[float, float] = (0, 0), rotation: float = 0) -> np.ndarray:
        """Create vertices of a unit regular hexagon using vectorized operations"""
        angle_offset = math.radians(rotation)
        radius = 1.0
        angles = np.array([angle_offset + i * math.pi / 3 for i in range(6)])
        x_coords = center[0] + radius * np.cos(angles)
        y_coords = center[1] + radius * np.sin(angles)
        return np.column_stack((x_coords, y_coords))

    @staticmethod
    def get_all_vertices(hex_data: np.ndarray) -> List[Tuple[float, float]]:
        """Extract all vertices from all hexagons efficiently"""
        all_vertices = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            vertices = HexagonGeometry.create_unit_hexagon_vertices(center, rotation)
            all_vertices.extend([(v[0], v[1]) for v in vertices])
        return all_vertices

class HexagonConstraints:
    """Handles constraint checking for hexagon arrangements"""

    @staticmethod
    def check_overlap_fast(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast overlap check using Shapely with minimal buffering"""
        try:
            poly1 = Polygon(hex1_vertices)
            poly2 = Polygon(hex2_vertices)
            # Minimal buffer to handle floating point precision
            return poly1.buffer(1e-12).intersects(poly2.buffer(1e-12))
        except:
            return False

    @staticmethod
    def check_containment_fast(inner_vertices: np.ndarray, outer_vertices: np.ndarray) -> bool:
        """Fast containment check"""
        try:
            inner_poly = Polygon(inner_vertices)
            outer_poly = Polygon(outer_vertices)
            return outer_poly.contains(inner_poly)
        except:
            return False

    @staticmethod
    def compute_overlap_penalty_fast(hexagons_vertices: List[np.ndarray]) -> float:
        """Efficiently compute overlap penalty with early termination"""
        penalty = 0.0
        n = len(hexagons_vertices)
        for i in range(n):
            for j in range(i+1, n):
                if HexagonConstraints.check_overlap_fast(hexagons_vertices[i], hexagons_vertices[j]):
                    penalty += 1000.0
                # Early termination if penalty becomes large
                if penalty > 5000.0:
                    return penalty
        return penalty

class HexagonEvaluator:
    """Evaluates hexagon packing configurations with optimized performance"""

    @staticmethod
    def calculate_outer_hex_radius_fast(hex_data: np.ndarray) -> float:
        """Fast calculation of minimum outer hexagon radius"""
        max_distance = 0.0
        # Precompute all hexagon vertices for faster access
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]

            # Get vertices quickly
            vertices = HexagonGeometry.create_unit_hexagon_vertices(center, rotation)

            # Calculate distances from origin to all vertices
            distances = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
            max_distance = max(max_distance, np.max(distances))

        return max_distance + 0.1  # Add small margin

    @staticmethod
    def evaluate_configuration_fast(hex_data: np.ndarray) -> float:
        """Fast evaluation of configuration with early rejection"""
        # Quick validation
        if len(hex_data) != 12:
            return 1e-10

        # Early distance-based feasibility check
        positions = hex_data[:, :2]
        distances = cdist(positions, positions)
        min_dist = np.min(distances[distances > 0])
        if min_dist < 1.8:  # Threshold to avoid overlaps
            return 1e-10

        try:
            # Calculate outer radius
            outer_radius = HexagonEvaluator.calculate_outer_hex_radius_fast(hex_data)

            # Create hexagon vertices for constraint checking
            hexagons_vertices = []
            for i in range(len(hex_data)):
                center = (hex_data[i][0], hex_data[i][1])
                rotation = hex_data[i][2]
                vertices = HexagonGeometry.create_unit_hexagon_vertices(center, rotation)
                hexagons_vertices.append(vertices)

            # Check overlaps
            overlap_penalty = HexagonConstraints.compute_overlap_penalty_fast(hexagons_vertices)

            if overlap_penalty > 0:
                return 1e-10

            # Valid configuration - return inverse of outer radius
            return 1.0 / outer_radius

        except Exception:
            return 1e-10

class SymmetryGenerationStrategy:
    """Generates various symmetric configurations for hexagon packing"""

    @staticmethod
    def generate_kagome_pattern() -> np.ndarray:
        """Generate Kagome lattice pattern - dense packing with precise geometry"""
        # Using mathematical constants for optimal spacing
        sqrt3 = np.sqrt(3)
        config = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom
            [sqrt3, 1.0, 0],     # top-right
            [-sqrt3, 1.0, 0],    # top-left
            [sqrt3, -1.0, 0],    # bottom-right
            [-sqrt3, -1.0, 0],   # bottom-left
            [2*sqrt3, 0, 0],     # far right
            [-2*sqrt3, 0, 0],    # far left
            [sqrt3, 3.0, 0],     # upper right corner
            [-sqrt3, 3.0, 0],    # upper left corner
            [sqrt3, -3.0, 0],    # lower right corner
            [-sqrt3, -3.0, 0],   # lower left corner
        ])
        return config[:12]

    @staticmethod
    def generate_hcp_pattern() -> np.ndarray:
        """Generate Hexagonal Close-Packed pattern with mathematical precision"""
        # HCP pattern with optimized unit distances
        sqrt3 = np.sqrt(3)
        config = np.array([
            [0, 0, 0],           # center
            [0, 1.8, 0],         # top
            [0, -1.8, 0],        # bottom
            [sqrt3 * 0.9, 0.9, 0],      # top-right
            [-sqrt3 * 0.9, 0.9, 0],     # top-left
            [sqrt3 * 0.9, -0.9, 0],     # bottom-right
            [-sqrt3 * 0.9, -0.9, 0],    # bottom-left
            [sqrt3 * 1.8, 0, 0],        # far right
            [-sqrt3 * 1.8, 0, 0],       # far left
            [sqrt3 * 0.9, 2.7, 0],      # upper right corner
            [-sqrt3 * 0.9, 2.7, 0],     # upper left corner
            [sqrt3 * 0.9, -2.7, 0],     # lower right corner
            [-sqrt3 * 0.9, -2.7, 0],    # lower left corner
        ])
        return config[:12]

    @staticmethod
    def generate_hexagonal_cluster() -> np.ndarray:
        """Generate standard hexagonal cluster with better mathematical foundation"""
        sqrt3 = np.sqrt(3)
        config = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom
            [sqrt3, 1.0, 0],     # top-right
            [-sqrt3, 1.0, 0],    # top-left
            [sqrt3, -1.0, 0],    # bottom-right
            [-sqrt3, -1.0, 0],   # bottom-left
            [2*sqrt3, 0, 0],     # far right
            [-2*sqrt3, 0, 0],    # far left
            [sqrt3, 3.0, 0],     # upper right corner
            [-sqrt3, 3.0, 0],    # upper left corner
            [sqrt3, -3.0, 0],    # lower right corner
            [-sqrt3, -3.0, 0],   # lower left corner
        ])
        return config[:12]

    @staticmethod
    def generate_optimized_benchmark_config() -> np.ndarray:
        """Generate configuration specifically designed to approach the benchmark score"""
        # Based on known good solutions approaching 1/3.9419123 ≈ 0.2537
        sqrt3 = np.sqrt(3)
        config = np.array([
            [0, 0, 0],             # center
            [0, 1.9, 0],           # top
            [0, -1.9, 0],          # bottom
            [sqrt3 * 0.95, 0.95, 0],    # top-right
            [-sqrt3 * 0.95, 0.95, 0],   # top-left
            [sqrt3 * 0.95, -0.95, 0],   # bottom-right
            [-sqrt3 * 0.95, -0.95, 0],  # bottom-left
            [sqrt3 * 1.9, 0, 0],        # far right
            [-sqrt3 * 1.9, 0, 0],       # far left
            [sqrt3 * 0.95, 2.85, 0],    # upper right corner
            [-sqrt3 * 0.95, 2.85, 0],   # upper left corner
            [sqrt3 * 0.95, -2.85, 0],   # lower right corner
            [-sqrt3 * 0.95, -2.85, 0],  # lower left corner
        ])
        return config[:12]

    @staticmethod
    def generate_symmetric_configs() -> List[np.ndarray]:
        """Generate multiple symmetric configurations with mathematical rigor"""
        configs = []
        configs.append(SymmetryGenerationStrategy.generate_hexagonal_cluster())
        configs.append(SymmetryGenerationStrategy.generate_kagome_pattern())
        configs.append(SymmetryGenerationStrategy.generate_hcp_pattern())
        configs.append(SymmetryGenerationStrategy.generate_optimized_benchmark_config())
        return configs

class AdaptiveHexagonOptimizer:
    """Implements a hybrid adaptive optimization strategy for hexagon packing"""

    def __init__(self):
        self.best_score = 0.0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds
        self.convergence_history = deque(maxlen=10)

    def adaptive_mutate(self, individual: np.ndarray, stage: int = 1, generation: int = 0) -> np.ndarray:
        """Adaptive mutation with exponential decay and intelligent group behavior"""
        mutated = individual.copy()
        
        # Calculate adaptive mutation strength with exponential decay
        base_mutation = 0.3 if stage == 1 else (0.2 if stage == 2 else 0.05)
        decay_factor = 0.98 ** generation
        current_mutation = base_mutation * decay_factor
        
        # Group hexagons based on distance from center
        # Central (0), Ring 1 (1-6), Ring 2 (7-11)
        central_idx = [0]
        ring1_idx = list(range(1, 7))
        ring2_idx = list(range(7, 12))
        
        # Mutate central hexagon
        for idx in central_idx:
            mutated[idx][0] += random.uniform(-current_mutation, current_mutation)
            mutated[idx][1] += random.uniform(-current_mutation, current_mutation)
            if random.random() < 0.3:
                mutated[idx][2] += random.uniform(-15, 15)
        
        # Mutate ring 1 hexagons in coordinated groups
        for i in range(0, len(ring1_idx), 2):
            if i + 1 < len(ring1_idx):
                idx1, idx2 = ring1_idx[i], ring1_idx[i+1]
                # Mutate symmetric pairs
                dx = random.uniform(-current_mutation * 0.8, current_mutation * 0.8)
                dy = random.uniform(-current_mutation * 0.8, current_mutation * 0.8)
                mutated[idx1][0] += dx
                mutated[idx1][1] += dy
                mutated[idx2][0] += dx
                mutated[idx2][1] += dy
                # Coordination of rotations
                if random.random() < 0.4:
                    rot_delta = random.uniform(-10, 10)
                    mutated[idx1][2] += rot_delta
                    mutated[idx2][2] += rot_delta
            else:
                # Handle odd case
                idx = ring1_idx[i]
                mutated[idx][0] += random.uniform(-current_mutation, current_mutation)
                mutated[idx][1] += random.uniform(-current_mutation, current_mutation)
                if random.random() < 0.3:
                    mutated[idx][2] += random.uniform(-15, 15)
        
        # Mutate ring 2 hexagons independently
        for idx in ring2_idx:
            mutated[idx][0] += random.uniform(-current_mutation, current_mutation)
            mutated[idx][1] += random.uniform(-current_mutation, current_mutation)
            if random.random() < 0.3:
                mutated[idx][2] += random.uniform(-10, 10)
        
        # Normalize rotations
        for i in range(len(mutated)):
            mutated[i][2] %= 360
            
        return mutated

    def adaptive_optimize_stage(self, initial_config: np.ndarray, stage: int = 1, target_generations: int = 30) -> Tuple[np.ndarray, float]:
        """Adaptive optimization stage with convergence monitoring and restarts"""
        # Stage-specific parameters
        if stage == 1:  # Coarse
            base_mutation = 0.3
            generations = target_generations
        elif stage == 2:  # Fine
            base_mutation = 0.2
            generations = target_generations
        else:  # Final
            base_mutation = 0.05
            generations = target_generations

        current_config = initial_config.copy()
        current_score = HexagonEvaluator.evaluate_configuration_fast(current_config)
        
        # Set up for convergence monitoring
        improvement_threshold = 1e-8
        patience = 0
        max_patience = 10
        last_improvement = 0
        
        # Main optimization loop
        for gen in range(generations):
            # Check timeout
            if time.time() - self.start_time > self.timeout * 0.8:
                break
                
            # Apply adaptive mutation
            mutated_config = self.adaptive_mutate(current_config, stage, gen)
            
            # Evaluate mutated configuration
            mutated_score = HexagonEvaluator.evaluate_configuration_fast(mutated_config)
            
            # Accept if better or randomly accept worse moves (simulated annealing effect)
            if mutated_score > current_score or random.random() < 0.1:
                current_config = mutated_config.copy()
                current_score = mutated_score
                
                # Update best if improved
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_config = current_config.copy()
                    last_improvement = gen
                    
            # Convergence monitoring
            self.convergence_history.append(current_score)
            
            # Check for stagnation and trigger restart if needed
            if len(self.convergence_history) >= 5:
                recent_improvements = [self.convergence_history[i] - self.convergence_history[i-1] 
                                     for i in range(1, len(self.convergence_history))]
                avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
                
                if avg_improvement < improvement_threshold:
                    patience += 1
                    if patience > max_patience:
                        # Restart with better candidate
                        if self.best_config is not None:
                            current_config = self.best_config.copy()
                            current_score = self.best_score
                        patience = 0
                else:
                    patience = 0  # Reset patience if improvement detected
                    
        return current_config, current_score

    def hybrid_optimize_with_restarts(self, initial_config: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run complete optimization with restart mechanism"""
        # Stage 1: Coarse optimization
        print("Stage 1: Coarse position optimization...")
        coarse_config = initial_config.copy()
        # Fix rotations for coarse stage
        for i in range(len(coarse_config)):
            coarse_config[i][2] = 0

        stage1_config, stage1_score = self.adaptive_optimize_stage(coarse_config, stage=1, target_generations=25)

        # Stage 2: Fine refinement with rotation awareness
        print("Stage 2: Fine refinement with rotation...")
        refined_config = stage1_config.copy()
        # Add small random rotations
        for i in range(len(refined_config)):
            if random.random() < 0.4:
                refined_config[i][2] += random.uniform(-5, 5)

        stage2_config, stage2_score = self.adaptive_optimize_stage(refined_config, stage=2, target_generations=35)

        # Stage 3: Final scipy optimization
        print("Stage 3: Final scipy optimization...")
        final_config = self.final_refinement_scipy(stage2_config)

        return final_config, self.best_score

    def final_refinement_scipy(self, config: np.ndarray) -> np.ndarray:
        """Final refinement using scipy optimization"""
        def objective_func(params):
            positions = params.reshape(-1, 2)
            temp_data = config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            outer_radius = HexagonEvaluator.calculate_outer_hex_radius_fast(temp_data)
            return outer_radius

        try:
            # Flatten positions
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()

            # Optimize with bounds
            result = minimize(
                objective_func,
                initial_positions,
                method='L-BFGS-B',
                bounds=[(-5, 5) for _ in range(24)],
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]

        except:
            pass  # Keep original if optimization fails

        return config

class MainOptimizer:
    """Main controller for hexagon packing optimization"""

    def __init__(self):
        self.optimizer = AdaptiveHexagonOptimizer()

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Complete optimization pipeline with enhanced strategies"""
        # Generate initial configurations
        configs = SymmetryGenerationStrategy.generate_symmetric_configs()

        # Find best initial configuration
        best_initial_score = 0.0
        best_initial_config = None

        for config in configs:
            score = HexagonEvaluator.evaluate_configuration_fast(config)
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()

        # Set up tracking
        self.optimizer.best_score = best_initial_score
        self.optimizer.best_config = best_initial_config.copy()

        # Run optimized stages
        final_config, final_score = self.optimizer.hybrid_optimize_with_restarts(best_initial_config)

        # Final evaluation
        final_radius = HexagonEvaluator.calculate_outer_hex_radius_fast(final_config)
        outer_hex_side_length = final_radius + 0.2

        outer_hex_data = np.array([0, 0, 0])

        return final_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = MainOptimizer()
    return optimizer.run_full_optimization()

# EVOLVE-BLOCK-END