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
    """Fast collision check using bounding boxes and Shapely optimizations"""
    # Use Shapely's built-in spatial indexing for better performance
    # First do a quick bounding box test
    bbox1 = hex_poly1.bounds
    bbox2 = hex_poly2.bounds

    # Quick bounding box overlap test
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False

    # More precise check using Shapely's efficient intersection
    return hex_poly1.intersects(hex_poly2)

def build_spatial_grid(hexagons, grid_size=5.0):
    """Build optimized spatial grid for fast collision detection using hexagon properties"""
    grid = defaultdict(list)
    for i, hex_poly in enumerate(hexagons):
        # Use hexagon-specific bounding box that better fits the actual shape
        bbox = hex_poly.bounds
        min_x, min_y, max_x, max_y = bbox

        # Add padding to ensure we capture all potential neighbors
        # Since hexagons are regular, we can estimate a better grid spacing
        cell_width = grid_size
        cell_height = grid_size

        # Determine grid cells that this hexagon occupies
        start_col = int(min_x // cell_width)
        end_col = int(max_x // cell_width) + 1
        start_row = int(min_y // cell_height)
        end_row = int(max_y // cell_height) + 1

        for col in range(start_col, end_col):
            for row in range(start_row, end_row):
                grid[(col, row)].append(i)
    return grid

def get_collision_candidates(grid, hex_index, hex_poly, grid_size=5.0):
    """Get potential collision candidates with better hexagon-aware selection"""
    candidates = []
    bbox = hex_poly.bounds
    min_x, min_y, max_x, max_y = bbox

    # Calculate the grid cells this hexagon would occupy
    start_col = int(min_x // grid_size)
    end_col = int(max_x // grid_size) + 1
    start_row = int(min_y // grid_size)
    end_row = int(max_y // grid_size) + 1

    # Check neighboring grid cells for potential collisions
    for col in range(start_col - 1, end_col + 1):
        for row in range(start_row - 1, end_row + 1):
            candidates.extend(grid.get((col, row), []))

    # Remove self from candidates
    return [i for i in candidates if i != hex_index]

class OptimizedPackingOptimizer:
    """Optimizes hexagon packing with enhanced performance and reliability"""
    
    def __init__(self, num_hexagons=11, side_length=1.0):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.hex_radius = side_length * np.sqrt(3) / 2  # Distance from center to corner
        self.grid_size = 3.0  # Optimized grid size
        
    def generate_geometric_seed(self):
        """Generate initial seed using geometric packing theory and symmetry with improved layout"""
        # Start with a systematic layout that respects hexagon symmetries
        positions = []
        angles = []
        
        # Central hexagon
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        # Hexagonal ring around center (6 hexagons) with better spacing
        for i in range(6):
            angle = i * 60
            # Spacing is 2 radii plus a small gap for better packing
            radius = 2.0 * self.hex_radius + 0.1
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
            (-2.0, 2.0), (2.0, -2.0),
            (1.0, 3.5), (-1.0, -3.5)
        ]
        
        # Ensure correct count
        for pos in additional_positions:
            if len(positions) < self.num_hexagons:
                positions.append(list(pos))
                angles.append(0.0)
        
        # Fill remaining slots if needed with careful placement
        while len(positions) < self.num_hexagons:
            positions.append([0.0, 0.0])
            angles.append(0.0)
            
        return np.array(positions[:self.num_hexagons]), np.array(angles[:self.num_hexagons])
    
    def evaluate_feasibility(self, positions, angles):
        """Quick feasibility check to prune bad solutions early"""
        # Check if any hexagon is too far from center (will cause containment issues)
        for i, (pos, angle) in enumerate(zip(positions, angles)):
            # Check distance from center (should be reasonable for packing)
            dist_from_center = np.sqrt(pos[0]**2 + pos[1]**2)
            if dist_from_center > 20.0:  # Conservative upper bound for 11 hexagons
                return False
                
        return True
    
    def calculate_geometric_objective(self, positions, angles):
        """Calculate objective based on geometric properties"""
        # Primary objective: maximize 1/outer_radius (minimize outer radius)
        outer_radius = calculate_outer_hexagon_radius(positions, angles)
        
        # Secondary objectives for better packing quality:
        # 1. Encourage even distribution by penalizing clusters
        if len(positions) > 1:
            distances = cdist(positions, positions)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            avg_min_distance = np.mean(min_distances)
            # Penalize very small distances (overlaps/collisions)
            distance_penalty = 0.0 if avg_min_distance > 1.5 else 1000.0 / (avg_min_distance + 0.1)
        else:
            distance_penalty = 0.0
            
        # Total objective (negative because we minimize)
        total_obj = -1.0 / outer_radius + distance_penalty
        return total_obj
    
    def evaluate_constraints(self, positions, angles):
        """Comprehensive constraint evaluation with optimized checking"""
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
        
        # Check containment for all hexagons
        for hex_poly in hexagons:
            if not check_containment(hex_poly, outer_hexagon):
                containment_violations += 1
        
        # Check overlaps using spatial grid for efficiency
        grid = build_spatial_grid(hexagons, self.grid_size)
        
        # Check for overlaps efficiently
        for i in range(len(hexagons)):
            candidates = get_collision_candidates(grid, i, hexagons[i], self.grid_size)
            for j in candidates:
                if i != j and fast_collision_check(hexagons[i], hexagons[j]):
                    overlap_violations += 1
        
        return containment_violations, overlap_violations
    
    def create_individual(self, positions, angles):
        """Create individual representation"""
        return np.concatenate([positions.flatten(), angles])
    
    def decode_individual(self, individual):
        """Decode individual back to positions and angles"""
        positions = individual[:22].reshape(-1, 2)
        angles = individual[22:]
        return positions, angles
    
    def mutate_individual(self, individual, generation=0, max_generations=100):
        """Specialized mutation operator with adaptive parameters"""
        mutated = individual.copy()
        
        # Adaptive mutation based on generation
        # Start with higher mutation rate and decrease over time
        mutation_rate = 0.2 - (0.15 * generation / max_generations)
        mutation_rate = max(0.05, mutation_rate)
        
        mutation_strength = 0.3 - (0.2 * generation / max_generations)
        mutation_strength = max(0.05, mutation_strength)
        
        # Apply mutation to position and angle components
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Different mutation strategies for positions vs angles
                if i < 22:  # Position component
                    mutated[i] += random.gauss(0, mutation_strength)
                else:  # Angle component
                    mutated[i] += random.gauss(0, 20)  # Larger angle changes for orientation
                    # Keep angle within [0, 360) range
                    mutated[i] = mutated[i] % 360
                    
        return mutated
    
    def crossover_individuals(self, parent1, parent2, generation=0, max_generations=100):
        """Specialized crossover with adaptive probability"""
        # Adaptive crossover rate based on generation
        crossover_rate = 0.8 + (0.1 * generation / max_generations)  # Start low, increase
        
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover with bias toward preserving good geometric layouts
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Cross over in chunks to maintain some structural integrity
        chunk_size = 3
        for i in range(0, len(parent1), chunk_size):  
            if i + chunk_size - 1 < len(parent1):
                if random.random() < 0.5:
                    child1[i:i+chunk_size], child2[i:i+chunk_size] = \
                        child2[i:i+chunk_size], child1[i:i+chunk_size]
        
        return child1, child2
    
    def optimize(self, max_iterations=120, population_size=25):
        """Main optimization loop with enhanced EA parameters"""
        # Generate multiple initial populations with better geometric seeds
        populations = []
        for i in range(6):  # More diverse initial seeds
            np.random.seed(i * 42 + 12345)  # Fixed seed for reproducibility
            init_pos, init_ang = self.generate_geometric_seed()
            
            # Add more substantial random variation to initial solutions
            init_pos += np.random.normal(0, 0.8, init_pos.shape)
            init_ang += np.random.normal(0, 8, init_ang.shape)
            
            populations.append(self.create_individual(init_pos, init_ang))
        
        # Evolutionary algorithm with adaptive parameters
        current_population = populations
        best_fitness = float('inf')
        best_individual = None
        stagnation_counter = 0
        max_stagnation = 25
        
        for generation in range(max_iterations):
            # Evaluate fitness of current population
            fitness_scores = []
            for ind in current_population:
                positions, angles = self.decode_individual(ind)
                if not self.evaluate_feasibility(positions, angles):
                    fitness_scores.append(float('inf'))
                    continue
                    
                # Check constraints
                cont_viol, overlap_viol = self.evaluate_constraints(positions, angles)
                if cont_viol > 0 or overlap_viol > 0:
                    # High penalty for constraint violations
                    fitness_scores.append(1e10 + (cont_viol + overlap_viol) * 5e6)
                    continue
                
                # Calculate objective
                obj = self.calculate_geometric_objective(positions, angles)
                fitness_scores.append(obj)
            
            # Track best solution
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_individual = current_population[min_fitness_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Early termination if stagnation occurs
            if stagnation_counter >= max_stagnation:
                break
            
            # Selection and reproduction
            # Tournament selection with adaptive tournament size
            tournament_size = max(3, 5 - generation // 20)
            selected_indices = []
            
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
                
                # Crossover with adaptive rate
                child1, child2 = self.crossover_individuals(parent1, parent2, generation, max_iterations)
                
                # Mutation with adaptive rate
                child1 = self.mutate_individual(child1, generation, max_iterations)
                child2 = self.mutate_individual(child2, generation, max_iterations)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            current_population = new_population[:population_size]
        
        # Final refinement with enhanced local optimization
        final_positions, final_angles = self.decode_individual(best_individual)
        refined_positions, refined_angles = self.enhanced_local_refinement(final_positions, final_angles)
        
        return refined_positions, refined_angles
    
    def enhanced_local_refinement(self, positions, angles, max_iterations=100):
        """Enhanced local refinement with multi-strategy approach"""
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_obj = self.calculate_geometric_objective(best_positions, best_angles)
        
        # Multi-step refinement with adaptive strategies
        step_sizes = [0.15, 0.1, 0.05, 0.02]
        patience = 15
        patience_counter = 0
        
        # Track recent improvements for adaptive stepping
        recent_improvements = []
        
        for iteration in range(max_iterations):
            improved = False
            current_step = step_sizes[min(iteration // 20, len(step_sizes)-1)]
            
            # Try perturbing each hexagon systematically
            for i in range(len(positions)):
                # Try position perturbations in all directions
                for dim in range(2):
                    # Try multiple step sizes for each dimension
                    for delta_mult in [-1, 1]:
                        for step_size in [current_step, current_step * 0.5, current_step * 2.0]:
                            old_val = best_positions[i][dim]
                            best_positions[i][dim] = old_val + delta_mult * step_size
                            
                            # Check if this improves the solution
                            new_obj = self.calculate_geometric_objective(best_positions, best_angles)
                            if new_obj < best_obj:
                                best_obj = new_obj
                                improved = True
                            else:
                                best_positions[i][dim] = old_val
                
                # Try angle perturbations with various steps
                old_angle = best_angles[i]
                angle_steps = [-5.0, -2.5, -1.0, 1.0, 2.5, 5.0]
                for delta in angle_steps:
                    best_angles[i] = (old_angle + delta) % 360
                    
                    new_obj = self.calculate_geometric_objective(best_positions, best_angles)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        improved = True
                    else:
                        best_angles[i] = old_angle
            
            # Adaptive step size adjustment
            recent_improvements.append(1 if improved else 0)
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
            
            # If we haven't improved recently, reduce step size
            if sum(recent_improvements) == 0 and current_step > 0.005:
                current_step *= 0.8
            elif sum(recent_improvements) > 3 and current_step < 0.3:
                current_step = min(0.3, current_step * 1.2)
            
            # Check for improvement
            if not improved:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
        
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
        optimizer = OptimizedPackingOptimizer(num_hexagons=11, side_length=1.0)
        
        # Run optimization
        final_positions, final_angles = optimizer.optimize(max_iterations=120, population_size=25)
        
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