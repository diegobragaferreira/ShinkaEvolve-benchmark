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
    """Fast collision check using bounding boxes"""
    bbox1 = hex_poly1.bounds
    bbox2 = hex_poly2.bounds
    
    # Quick bounding box overlap test
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False
    
    # More precise check
    return hex_poly1.intersects(hex_poly2)

def build_spatial_grid(hexagons, grid_size=5.0):
    """Build spatial grid for fast collision detection"""
    grid = defaultdict(list)
    for i, hex_poly in enumerate(hexagons):
        bbox = hex_poly.bounds
        min_x, min_y, max_x, max_y = bbox
        for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
            for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
                grid[(x,y)].append(i)
    return grid

def get_collision_candidates(grid, hex_index, hex_poly):
    """Get potential collision candidates efficiently"""
    candidates = []
    bbox = hex_poly.bounds
    min_x, min_y, max_x, max_y = bbox
    
    for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
        for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
            candidates.extend(grid.get((x,y), []))
    return [i for i in candidates if i != hex_index]

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
    
    def evaluate_constraints(self, positions, angles):
        """Comprehensive constraint evaluation"""
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
        for hex_poly in hexagags:
            if not check_containment(hex_poly, outer_hexagon):
                containment_violations += 1
        
        # Check overlaps using spatial grid for efficiency
        grid = build_spatial_grid(hexagons, self.grid_size)
        
        # Check for overlaps
        for i in range(len(hexagons)):
            candidates = get_collision_candidates(grid, i, hexagons[i])
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
    
    def crossover_individuals(self, parent1, parent2, crossover_rate=0.8):
        """Specialized crossover for hexagon positioning"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover with some bias toward preserving good geometric layouts
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Cross over in chunks to maintain some structural integrity
        for i in range(0, len(parent1), 3):  # Process groups of 3 elements
            if i + 2 < len(parent1):
                if random.random() < 0.5:
                    child1[i:i+3], child2[i:i+3] = child2[i:i+3], child1[i:i+3]
        
        return child1, child2
    
    def optimize(self, max_iterations=150, population_size=20):
        """Main optimization loop"""
        # Generate multiple initial populations with geometric seeds
        populations = []
        for i in range(5):  # Multiple initial seeds
            np.random.seed(i * 42)
            init_pos, init_ang = self.generate_geometric_seed()
            
            # Add some random variation to initial solutions
            init_pos += np.random.normal(0, 0.5, init_pos.shape)
            init_ang += np.random.normal(0, 5, init_ang.shape)
            
            populations.append(self.create_individual(init_pos, init_ang))
        
        # Evolutionary algorithm with adaptive parameters
        current_population = populations
        best_fitness = float('inf')
        best_individual = None
        stagnation_counter = 0
        max_stagnation = 30
        
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
                    fitness_scores.append(1e10 + (cont_viol + overlap_viol) * 1e6)
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
                
                # Crossover
                child1, child2 = self.crossover_individuals(parent1, parent2)
                
                # Mutation
                child1 = self.mutate_individual(child1, mutation_rate=0.15, mutation_strength=0.3)
                child2 = self.mutate_individual(child2, mutation_rate=0.15, mutation_strength=0.3)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            current_population = new_population[:population_size]
        
        # Final refinement with local optimization
        final_positions, final_angles = self.decode_individual(best_individual)
        refined_positions, refined_angles = self.local_refinement(final_positions, final_angles)
        
        return refined_positions, refined_angles
    
    def local_refinement(self, positions, angles, max_iterations=50):
        """Fine-tune solution with local search"""
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_obj = self.calculate_geometric_objective(best_positions, best_angles)
        
        # Gradient-like local search with geometric-aware perturbations
        step_sizes = [0.1, 0.05, 0.01]
        patience = 10
        patience_counter = 0
        
        for iteration in range(max_iterations):
            improved = False
            current_step = step_sizes[min(iteration // 10, len(step_sizes)-1)]
            
            # Try perturbing each hexagon
            for i in range(len(positions)):
                # Try position perturbations
                for dim in range(2):
                    for delta in [-current_step, current_step]:
                        old_val = best_positions[i][dim]
                        best_positions[i][dim] = old_val + delta
                        
                        # Check if this improves the solution
                        new_obj = self.calculate_geometric_objective(best_positions, best_angles)
                        if new_obj < best_obj:
                            best_obj = new_obj
                            improved = True
                        else:
                            best_positions[i][dim] = old_val
                
                # Try angle perturbations
                for delta in [-2.0, -1.0, 1.0, 2.0]:
                    old_angle = best_angles[i]
                    best_angles[i] = (old_angle + delta) % 360
                    
                    new_obj = self.calculate_geometric_objective(best_positions, best_angles)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        improved = True
                    else:
                        best_angles[i] = old_angle
            
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