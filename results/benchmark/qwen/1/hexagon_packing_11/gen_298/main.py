# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
import warnings
import random
from collections import defaultdict
from scipy.spatial.distance import cdist
import math

warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

def get_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Get shapely polygon representation of hexagon"""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(hex_poly, outer_poly):
    """Check if hexagon is completely contained within outer hexagon"""
    return outer_poly.contains(hex_poly) or (outer_poly.intersects(hex_poly) and
                                           outer_poly.intersection(hex_poly).area == hex_poly.area)

def calculate_outer_hexagon_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed to contain all inner hexagons"""
    max_dist = 0
    outer_center = (0, 0)

    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)

    # Find maximum distance from center
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)

    # Add buffer for safety and account for hexagon shape
    return max_dist * 1.1  # Safety factor

def fast_collision_check(hex_poly1, hex_poly2):
    """Fast collision check using hexagon-specific optimizations"""
    # Early rejection using circle-based approximation for speed
    centroid1 = hex_poly1.centroid
    centroid2 = hex_poly2.centroid

    # Calculate distance between centroids
    dist_centroids = np.sqrt((centroid1.x - centroid2.x)**2 + (centroid1.y - centroid2.y)**2)

    # For unit hexagons, if centroids are further apart than 2*radius, no collision
    # Radius of hexagon is sqrt(3)/2 ≈ 0.866, so diameter ≈ 1.732
    # We use 2.5 as safe threshold to avoid false negatives
    if dist_centroids > 2.5:
        return False

    # Quick bounding box overlap test
    bbox1 = hex_poly1.bounds
    bbox2 = hex_poly2.bounds

    # Quick bounding box overlap test
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False

    # More precise check
    return hex_poly1.intersects(hex_poly2)

def build_optimized_spatial_grid(hexagons, grid_cell_size=2.0):
    """Build optimized spatial grid specifically for hexagonal geometry"""
    grid = defaultdict(list)

    for i, hex_poly in enumerate(hexagons):
        # Get hexagon bounds
        bbox = hex_poly.bounds
        min_x, min_y, max_x, max_y = bbox

        # Calculate grid cell indices for the bounding box
        start_col = int(min_x // grid_cell_size)
        end_col = int(max_x // grid_cell_size) + 1
        start_row = int(min_y // grid_cell_size)
        end_row = int(max_y // grid_cell_size) + 1

        # Add to all overlapping grid cells
        for col in range(start_col, end_col):
            for row in range(start_row, end_row):
                grid[(col, row)].append(i)

    return grid

def get_collision_candidates_optimized(grid, hex_index, hex_poly, grid_cell_size=2.0):
    """Get collision candidates using optimized spatial grid with hexagon-aware selection"""
    candidates = []
    bbox = hex_poly.bounds
    min_x, min_y, max_x, max_y = bbox

    # Calculate approximate grid cells that this hexagon might overlap
    # Use a larger search window to catch potential overlaps
    search_radius = 2  # Number of cells around to check

    start_col = int((min_x - grid_cell_size * search_radius) // grid_cell_size)
    end_col = int((max_x + grid_cell_size * search_radius) // grid_cell_size) + 1
    start_row = int((min_y - grid_cell_size * search_radius) // grid_cell_size)
    end_row = int((max_y + grid_cell_size * search_radius) // grid_cell_size) + 1

    # Check all neighboring grid cells for candidates
    for col in range(start_col, end_col):
        for row in range(start_row, end_row):
            candidates.extend(grid.get((col, row), []))

    # Remove duplicates and self-reference
    candidates = list(set(candidates))
    return [i for i in candidates if i != hex_index]

def build_hierarchical_spatial_grid(hexagons, coarse_grid_size=2.8, fine_grid_size=1.4):
    """Build hierarchical spatial grid with adaptive cell sizing based on hexagon density"""
    # Simplified approach: always use optimized single-grid approach for better consistency
    # The hierarchical approach was overcomplicating things without clear gains

    # For hexagon packing, a single optimized grid works better
    return build_optimized_spatial_grid(hexagons, grid_cell_size=2.0), None, False

def get_collision_candidates_hierarchical(grid_tuple, hex_index, hex_poly):
    """Get collision candidates using optimized single grid approach"""
    # Unpack the simplified grid structure
    grid, _, _ = grid_tuple
    return get_collision_candidates_optimized(grid, hex_index, hex_poly, grid_cell_size=2.0)

class GeometricPackingOptimizer:
    """Optimizes hexagon packing using a hybrid approach combining geometric insights with evolutionary algorithms"""

    def __init__(self, num_hexagons=11, side_length=1.0):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.hex_radius = side_length * np.sqrt(3) / 2  # Distance from center to corner
        self.grid_size = 5.0

    def generate_geometric_seed(self):
        """Generate initial seed using geometric packing theory and symmetry"""
        # Start with a systematic layout that respects hexagon symmetries
        positions = []
        angles = []

        # Central hexagon
        positions.append([0.0, 0.0])
        angles.append(0.0)

        # Hexagonal ring around center (6 hexagons)
        for i in range(6):
            angle = i * 60
            radius = 2.0 * self.hex_radius  # Spacing is 2 radii for unit hexagons
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y])
            angles.append(0.0)

        # Fill in the remaining positions with a structured pattern
        # Based on mathematical packing of hexagons in hexagonal lattice
        additional_positions = [
            (-3.0, 1.5), (3.0, 1.5),
            (-3.0, -1.5), (3.0, -1.5),
            (0.0, 3.0), (0.0, -3.0),
            (2.0, 2.0), (-2.0, -2.0),
            (-2.0, 2.0), (2.0, -2.0)
        ]

        # Ensure correct count
        for pos in additional_positions:
            if len(positions) < self.num_hexagons:
                positions.append(list(pos))
                angles.append(0.0)

        # Fill remaining slots if needed
        while len(positions) < self.num_hexagons:
            positions.append([0.0, 0.0])
            angles.append(0.0)

        return np.array(positions[:self.num_hexagons]), np.array(angles[:self.num_hexagons])

    def evaluate_feasibility(self, positions, angles):
        """Quick feasibility check to prune bad solutions early"""
        # Check if any hexagon is too close to boundaries (will cause containment issues)
        # Using a safety margin based on hexagon size
        safety_margin = 1.0

        for i, (pos, angle) in enumerate(zip(positions, angles)):
            # Check distance from center (should be reasonable for packing)
            dist_from_center = np.sqrt(pos[0]**2 + pos[1]**2)
            if dist_from_center > 15.0:  # Reasonable upper bound for 11 hexagons
                return False

        return True

    def calculate_geometric_objective(self, positions, angles):
        """Calculate objective based on geometric properties"""
        # Primary objective: maximize 1/outer_radius (minimize outer radius)
        outer_radius = calculate_outer_hexagon_radius(positions, angles)

        # Secondary objectives for better packing quality:
        # 1. Minimize average distance between hexagons (encourage even distribution)
        if len(positions) > 1:
            distances = cdist(positions, positions)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            avg_min_distance = np.mean(min_distances)
            # Penalize small distances (overlaps/collisions)
            distance_penalty = 0.0 if avg_min_distance > 2.0 else 1000.0 / (avg_min_distance + 0.1)
        else:
            distance_penalty = 0.0

        # Total objective (negative because we minimize)
        total_obj = -1.0 / outer_radius + distance_penalty
        return total_obj

    def evaluate_constraints(self, positions, angles, generation=None, recent_improvements=None):
        """Comprehensive constraint evaluation with adaptive penalty handling"""
        # Create hexagon polygons
        hexagons = []
        for i in range(len(positions)):
            hex_poly = get_hexagon_polygon(positions[i][0], positions[i][1], angles[i])
            hexagons.append(hex_poly)

        # Check containment
        outer_radius = calculate_outer_hexagon_radius(positions, angles)
        outer_hexagon = get_hexagon_polygon(0, 0, 0, outer_radius)

        containment_violations = 0
        overlap_violations = 0

        # Check containment
        for hex_poly in hexagons:
            if not check_containment(hex_poly, outer_hexagon):
                containment_violations += 1

        # Check overlaps using hierarchical spatial grid for efficiency
        coarse_grid, fine_grid, use_fine = build_hierarchical_spatial_grid(hexagons)
        grid_tuple = (coarse_grid, fine_grid, use_fine)

        # Check for overlaps
        for i in range(len(hexagons)):
            candidates = get_collision_candidates_hierarchical(grid_tuple, i, hexagons[i])
            for j in candidates:
                if i != j and fast_collision_check(hexagons[i], hexagons[j]):
                    overlap_violations += 1

        return containment_violations, overlap_violations

    def calculate_adaptive_penalty(self, cont_viol, overlap_viol, generation=None, recent_improvements=None):
        """Calculate adaptive penalty based on optimization progress"""
        # Base penalty coefficients
        base_containment_penalty = 10000000
        base_overlap_penalty = 10000000

        # Dynamic adjustment based on optimization progress
        if recent_improvements is not None and len(recent_improvements) > 10:
            # If we haven't improved recently, increase penalties
            recent_progress = np.mean(recent_improvements[-10:])
            if recent_progress < 1e-6:  # No significant improvement
                base_containment_penalty *= 2.0
                base_overlap_penalty *= 2.0
            elif recent_progress > 1e-3:  # Good progress, reduce penalties
                base_containment_penalty *= 0.5
                base_overlap_penalty *= 0.5

        return base_containment_penalty * cont_viol + base_overlap_penalty * overlap_viol

    def create_individual(self, positions, angles):
        """Create individual representation"""
        return np.concatenate([positions.flatten(), angles])

    def decode_individual(self, individual):
        """Decode individual back to positions and angles"""
        positions = individual[:22].reshape(-1, 2)
        angles = individual[22:]
        return positions, angles

    def mutate_individual(self, individual, mutation_rate=0.1, mutation_strength=0.2):
        """Specialized mutation operator for hexagon packing"""
        mutated = individual.copy()

        # Apply mutation to position and angle components
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Different mutation strategies for positions vs angles
                if i < 22:  # Position component
                    mutated[i] += random.gauss(0, mutation_strength)
                else:  # Angle component
                    mutated[i] += random.gauss(0, 15)  # Larger angle changes for orientation
                    # Keep angle within [0, 360) range
                    mutated[i] = mutated[i] % 360

        return mutated

    def crossover_individuals(self, parent1, parent2, crossover_rate=0.8, generation=None, population=None):
        """Enhanced crossover with structure preservation and diversity considerations"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        positions1, angles1 = self.decode_individual(parent1)
        positions2, angles2 = self.decode_individual(parent2)

        # Enhanced crossover that considers geometric relationships
        child1_positions = positions1.copy()
        child1_angles = angles1.copy()
        child2_positions = positions2.copy()
        child2_angles = angles2.copy()

        # Apply crossover with consideration for hexagon arrangements
        # Use a combination of uniform crossover and structure-aware swapping
        for i in range(len(positions1)):
            # For positions, consider proximity to centroids of clusters
            if random.random() < 0.7:  # 70% chance of structure-aware crossover
                # Swap complete position/angle sets for better geometric clustering
                if random.random() < 0.5:
                    child1_positions[i] = positions2[i]
                    child1_angles[i] = angles2[i]
                    child2_positions[i] = positions1[i]
                    child2_angles[i] = angles1[i]
            else:
                # Regular uniform crossover
                if random.random() < 0.5:
                    child1_positions[i] = positions2[i]
                    child1_angles[i] = angles2[i]
                if random.random() < 0.5:
                    child2_positions[i] = positions1[i]
                    child2_angles[i] = angles1[i]

        # Convert back to individual form
        child1 = self.create_individual(child1_positions, child1_angles)
        child2 = self.create_individual(child2_positions, child2_angles)

        return child1, child2

    def monitor_diversity(self, population, generation, max_generations):
        """Monitor population diversity and trigger diversity-preserving actions"""
        if len(population) < 2:
            return False

        # Calculate diversity metric (variance in fitness scores)
        fitness_scores = []
        for ind in population:
            positions, angles = self.decode_individual(ind)
            if not self.evaluate_feasibility(positions, angles):
                continue
            cont_viol, overlap_viol = self.evaluate_constraints(positions, angles)
            if cont_viol > 0 or overlap_viol > 0:
                continue
            obj = self.calculate_geometric_objective(positions, angles)
            fitness_scores.append(obj)

        if len(fitness_scores) < 2:
            return False

        diversity_variance = np.var(fitness_scores)

        # Trigger diversity preservation when variance is too low
        if diversity_variance < 1e-6 and generation > max_generations * 0.3:
            return True
        return False

    def optimize(self, max_iterations=150, population_size=20):
        """Refined optimization pipeline with progressive constraint management"""
        # Progressive constraint relaxation and tightening
        initial_bounds = [
            (-15.0, 15.0) if i < 22 else (0.0, 360.0)
            for i in range(33)  # 11*2 positions + 11 angles
        ]

        # Generate multiple initial populations with better initialization
        populations = []
        for i in range(5):  # Multiple initial seeds
            np.random.seed(i * 42)
            init_pos, init_ang = self.generate_geometric_seed()

            # Add more structured random variation to improve exploration
            init_pos += np.random.normal(0, 0.5, init_pos.shape) * (1.0 + i * 0.1)
            init_ang += np.random.normal(0, 5, init_ang.shape) * (1.0 + i * 0.05)

            populations.append(self.create_individual(init_pos, init_ang))

        # Evolutionary algorithm with adaptive parameters and progressive tightening
        current_population = populations
        best_fitness = float('inf')
        best_individual = None
        stagnation_counter = 0
        max_stagnation = 30
        recent_improvements = []

        for generation in range(max_iterations):
            # Adapt bounds based on progress (tighten constraints in later generations)
            if generation > max_iterations * 0.6:
                # Tighter bounds in later generations
                bounds = [(-10.0, 10.0) if i < 22 else (0.0, 360.0) for i in range(33)]
            elif generation > max_iterations * 0.3:
                # Medium bounds for middle generations
                bounds = [(-12.0, 12.0) if i < 22 else (0.0, 360.0) for i in range(33)]
            else:
                # Broader bounds for early generations
                bounds = initial_bounds

            # Evaluate fitness of current population
            fitness_scores = []
            for ind in current_population:
                positions, angles = self.decode_individual(ind)
                if not self.evaluate_feasibility(positions, angles):
                    fitness_scores.append(float('inf'))
                    continue

                # Check constraints
                cont_viol, overlap_viol = self.evaluate_constraints(positions, angles, generation, recent_improvements)
                if cont_viol > 0 or overlap_viol > 0:
                    # Adaptive penalty based on progress
                    penalty = self.calculate_adaptive_penalty(cont_viol, overlap_viol, generation, recent_improvements)
                    fitness_scores.append(1e10 + penalty)
                    continue

                # Calculate objective
                obj = self.calculate_geometric_objective(positions, angles)
                fitness_scores.append(obj)

            # Track best solution and recent improvements
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_individual = current_population[min_fitness_idx].copy()
                stagnation_counter = 0
                recent_improvements.append(best_fitness)
            else:
                stagnation_counter += 1
                recent_improvements.append(float('inf'))

            # Early termination if stagnation occurs
            if stagnation_counter >= max_stagnation:
                break

            # Diversity maintenance
            if self.monitor_diversity(current_population, generation, max_iterations):
                # Introduce targeted mutations to preserve diversity
                for i in range(len(current_population)):
                    if random.random() < 0.3:  # 30% chance to mutate
                        current_population[i] = self.mutate_individual(
                            current_population[i],
                            mutation_rate=0.2,
                            mutation_strength=0.5
                        )

            # Selection and reproduction
            # Tournament selection
            selected_indices = []
            tournament_size = 3

            for _ in range(population_size):
                tournament_indices = np.random.choice(len(current_population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                selected_indices.append(winner_idx)

            # Create new population via crossover and mutation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_individual.copy())

            # Generate offspring
            while len(new_population) < population_size:
                # Select two parents
                parent1_idx = selected_indices[len(new_population) % len(selected_indices)]
                parent2_idx = selected_indices[(len(new_population) + 1) % len(selected_indices)]

                parent1 = current_population[parent1_idx]
                parent2 = current_population[parent2_idx]

                # Crossover with diversity considerations
                child1, child2 = self.crossover_individuals(parent1, parent2, generation=generation, population=current_population)

                # Mutation with adaptive rates
                mutation_rate = 0.15 if generation < max_iterations * 0.7 else 0.2
                mutation_strength = 0.3 if generation < max_iterations * 0.7 else 0.5

                child1 = self.mutate_individual(child1, mutation_rate=mutation_rate, mutation_strength=mutation_strength)
                child2 = self.mutate_individual(child2, mutation_rate=mutation_rate, mutation_strength=mutation_strength)

                new_population.extend([child1, child2])

            # Trim to exact population size
            current_population = new_population[:population_size]

        # Final refinement with enhanced local optimization
        final_positions, final_angles = self.decode_individual(best_individual)
        refined_positions, refined_angles = self.local_refinement(final_positions, final_angles, max_iterations=100, generation=max_iterations)

        return refined_positions, refined_angles

    def local_refinement(self, positions, angles, max_iterations=50, generation=None):
        """Enhanced multi-phase local refinement with adaptive strategies"""
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_obj = self.calculate_geometric_objective(best_positions, best_angles)

        # Multi-phase adaptive local search with intelligent step size adjustment
        phase1_steps = [0.2, 0.1, 0.05]  # Broad exploration
        phase2_steps = [0.1, 0.05, 0.02]  # Fine-tuning
        phase3_steps = [0.05, 0.02, 0.01]  # Precision

        # Track improvement history for adaptive adjustments
        improvement_history = []
        patience = 15
        patience_counter = 0

        for iteration in range(max_iterations):
            improved = False
            current_phase = min(iteration // 20, 2)  # 3 distinct phases
            step_sizes = [phase1_steps, phase2_steps, phase3_steps][current_phase]
            current_step = step_sizes[min(iteration % 10, len(step_sizes)-1)]

            # Try perturbing each hexagon systematically
            for i in range(len(positions)):
                # Try position perturbations with adaptive step sizes
                for dim in range(2):
                    for delta in [-current_step, current_step]:
                        old_val = best_positions[i][dim]
                        best_positions[i][dim] = old_val + delta

                        # Check if this improves the solution
                        new_obj = self.calculate_geometric_objective(best_positions, best_angles)
                        if new_obj < best_obj:
                            best_obj = new_obj
                            improved = True
                            improvement_history.append(best_obj)
                        else:
                            best_positions[i][dim] = old_val

                # Try angle perturbations with varying magnitudes
                angle_delta_options = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
                if current_phase == 2:  # Fine phase - smaller steps
                    angle_delta_options = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

                for delta in angle_delta_options:
                    old_angle = best_angles[i]
                    best_angles[i] = (old_angle + delta) % 360

                    new_obj = self.calculate_geometric_objective(best_positions, best_angles)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        improved = True
                        improvement_history.append(best_obj)
                    else:
                        best_angles[i] = old_angle

            # Adaptive patience based on recent progress
            if improvement_history:
                recent_change = improvement_history[-1] - improvement_history[max(0, len(improvement_history)-5)]
                if abs(recent_change) < 1e-8:
                    patience = max(5, patience - 1)  # Reduce patience if stuck
                else:
                    patience = 15  # Reset patience

            if not improved:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
                improvement_history.append(best_obj)

        return best_positions, best_angles

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Initialize optimizer with geometric awareness
        optimizer = GeometricPackingOptimizer(num_hexagons=11, side_length=1.0)

        # Run optimization
        final_positions, final_angles = optimizer.optimize(max_iterations=150, population_size=20)

        # Create inner hex data
        inner_hex_data = np.column_stack([final_positions, final_angles])

        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])

        # Calculate outer hex side length
        outer_radius = calculate_outer_hexagon_radius(final_positions, final_angles)
        # Convert to side length for regular hexagon
        outer_hex_side_length = outer_radius / (np.sqrt(3) / 2)

        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial solution
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
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END