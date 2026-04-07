# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution
import time
from typing import Tuple, List
import warnings
from numba import jit, prange
import random

# JIT compile geometric functions for performance
@jit(nopython=True)
def hexagon_vertices_jit(center_x: float, center_y: float, rotation_deg: float, side_length: float = 1.0) -> np.ndarray:
    """JIT compiled function to generate hexagon vertices"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices + close loop
    unit_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Apply rotation and translation
    rotation_rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    rotated_vertices = rotation_matrix @ unit_vertices.T
    translated_vertices = rotated_vertices.T * side_length + np.array([center_x, center_y])
    
    return translated_vertices

@jit(nopython=True)
def distance_point_to_line_segment(point_x: float, point_y: float,
                                   line_start_x: float, line_start_y: float,
                                   line_end_x: float, line_end_y: float) -> float:
    """JIT compiled function to compute distance from point to line segment"""
    A = point_x - line_start_x
    B = point_y - line_start_y
    C = line_end_x - line_start_x
    D = line_end_y - line_start_y

    dot = A*C + B*D
    len_sq = C*C + D*D
    if len_sq == 0:
        return np.sqrt(A*A + B*B)
    param = dot / len_sq
    param = max(0, min(1, param))
    xx = line_start_x + param * C
    yy = line_start_y + param * D
    dx = point_x - xx
    dy = point_y - yy
    return np.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def point_in_hexagon_jit(point_x: float, point_y: float, hex_vertices: np.ndarray) -> bool:
    """JIT compiled function to check if point is inside hexagon using ray casting"""
    # Ray casting algorithm
    x, y = point_x, point_y
    n = len(hex_vertices)
    inside = False
    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class HexagonPacker:
    """Advanced hexagon packing optimizer using a hybrid evolutionary approach"""
    
    def __init__(self):
        self.side_length = 1.0
        self.hex_width = 2.0
        self.hex_height = np.sqrt(3)
        self.hex_apothem = np.sqrt(3) / 2
        
    def create_hexagon(self, center_x: float, center_y: float, rotation_deg: float) -> Tuple[np.ndarray, float]:
        """Create hexagon with vertices and center distance"""
        vertices = hexagon_vertices_jit(center_x, center_y, rotation_deg, self.side_length)
        center_dist = np.sqrt(center_x*center_x + center_y*center_y)
        return vertices, center_dist
    
    def distance_between_hex_centers(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between hexagon centers"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_hexagon_contained(self, vertices: np.ndarray, outer_side_length: float) -> bool:
        """Check if all hexagon vertices are within outer hexagon using mathematical approach"""
        # For a regular hexagon centered at origin with side length s, 
        # maximum distance from center is s
        # So we just need to verify that max distance to any vertex <= outer_side_length * sqrt(3)/2
        max_dist = 0.0
        for vertex in vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_dist = max(max_dist, dist)
        
        # For outer hexagon with side length R, max distance is R
        # We want max distance within hexagon <= R * sqrt(3)/2  
        return max_dist <= outer_side_length * np.sqrt(3) / 2
    
    def check_overlap(self, vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
        """Check overlap using Shapely for precision"""
        try:
            poly1 = Polygon(vertices1)
            poly2 = Polygon(vertices2)
            return poly1.intersects(poly2)
        except:
            # Fallback to geometric distance-based check
            # Check if centers are close enough to potentially overlap
            center1 = np.mean(vertices1, axis=0)
            center2 = np.mean(vertices2, axis=0)
            dist_centers = np.sqrt(np.sum((center1 - center2)**2))
            return dist_centers < 2.0  # Approximate sum of radii
    
    def compute_outer_hexagon_side_length(self, inner_hex_data: np.ndarray) -> float:
        """Compute minimum outer hexagon side length"""
        # Collect all vertices
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = hexagon_vertices_jit(center_x, center_y, angle, self.side_length)
            all_vertices.extend(vertices)
        
        if not all_vertices:
            return 10.0
            
        all_vertices = np.array(all_vertices)
        
        # Find centroid
        centroid = np.mean(all_vertices, axis=0)
        
        # Find maximum distance from centroid to any vertex
        distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
        max_distance = np.max(distances)
        
        # Convert to outer hexagon side length
        # For regular hexagon, circumradius = side_length
        # But to contain all vertices, we need side_length >= max_distance * 2/sqrt(3)
        side_length = max_distance * 2 / np.sqrt(3)
        
        return side_length
    
    def generate_symmetric_initial(self) -> np.ndarray:
        """Generate highly symmetric initial configuration"""
        # Start with a central hex and surround with 11 others in a ring pattern
        positions = []
        
        # Central hexagon
        positions.append([0.0, 0.0, 0.0])
        
        # Ring 1: 6 hexagons evenly distributed
        for i in range(6):
            angle = i * 60  # degrees
            radius = 2.0  # distance from center
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0.0])
        
        # Ring 2: 5 hexagons in a pentagon-like arrangement
        for i in range(5):
            angle = i * 72 + 18  # offset to create asymmetry but maintain overall symmetry
            radius = 3.5
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0.0])
        
        return np.array(positions)

    def evaluate_fitness(self, solution: np.ndarray, outer_side_length: float = 10.0) -> Tuple[float, bool]:
        """Evaluate fitness of a solution configuration"""
        # Reshape solution
        hex_data = solution.reshape(-1, 3)
        
        # Check containment
        for i in range(len(hex_data)):
            center_x, center_y, angle = hex_data[i]
            vertices, _ = self.create_hexagon(center_x, center_y, angle)
            if not self.is_hexagon_contained(vertices, outer_side_length):
                return 1e10, False  # Invalid configuration - heavy penalty
        
        # Check for overlaps
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                vertices1, _ = self.create_hexagon(*hex_data[i])
                vertices2, _ = self.create_hexagon(*hex_data[j])
                if self.check_overlap(vertices1, vertices2):
                    return 1e10, False  # Invalid configuration - heavy penalty
        
        # Calculate objective value (negative of 1/outer_side_length)
        objective = -1.0 / outer_side_length
        return objective, True

    def mutate_solution(self, solution: np.ndarray, mutation_rate: float = 0.1, 
                       position_magnitude: float = 0.3, rotation_magnitude: float = 15.0) -> np.ndarray:
        """Apply mutation to solution"""
        mutated = solution.copy()
        
        # Mutate each element with probability mutation_rate
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                if i % 3 == 2:  # rotation parameter
                    mutated[i] += np.random.normal(0, rotation_magnitude)
                    # Keep within 0-360 range
                    mutated[i] = mutated[i] % 360
                else:  # position parameter
                    mutated[i] += np.random.normal(0, position_magnitude)
        
        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents"""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Swap segments at random points
        crossover_point = np.random.randint(0, len(parent1))
        child1[crossover_point:] = parent2[crossover_point:]
        child2[crossover_point:] = parent1[crossover_point:]
        
        return child1, child2

    def optimize_with_evolution(self, max_generations: int = 50, pop_size: int = 20) -> Tuple[np.ndarray, float]:
        """Evolutionary optimization approach"""
        # Initialize population
        population = []
        for i in range(pop_size):
            if i == 0:
                # First individual is the symmetric initial solution
                individual = self.generate_symmetric_initial().flatten()
            else:
                # Generate variations of the symmetric solution
                base = self.generate_symmetric_initial()
                variation = base + np.random.normal(0, 0.5, base.shape)
                individual = variation.flatten()
            
            population.append(individual)
        
        best_solution = None
        best_fitness = float('inf')
        best_outer_side_length = float('inf')
        
        # Evolutionary loop
        for generation in range(max_generations):
            # Evaluate all individuals
            fitness_scores = []
            valid_individuals = []
            
            for individual in population:
                outer_side_length = self.compute_outer_hexagon_side_length(individual.reshape(-1, 3))
                fitness, is_valid = self.evaluate_fitness(individual, outer_side_length)
                
                if is_valid:
                    fitness_scores.append(fitness)
                    valid_individuals.append((individual, fitness, outer_side_length))
                else:
                    fitness_scores.append(float('inf'))
            
            # Update best solution
            if valid_individuals:
                best_idx = np.argmin([score for _, score, _ in valid_individuals])
                individual, fitness, side_length = valid_individuals[best_idx]
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
                    best_outer_side_length = side_length
            
            # Selection and reproduction
            if len(valid_individuals) > 0:
                # Sort valid individuals by fitness
                sorted_individuals = sorted(valid_individuals, key=lambda x: x[1])
                sorted_individuals = sorted_individuals[:pop_size//2]  # Keep top half
                
                # Generate new population
                new_population = []
                # Keep best individuals
                for individual, _, _ in sorted_individuals:
                    new_population.append(individual.copy())
                
                # Fill rest with offspring
                while len(new_population) < pop_size:
                    # Tournament selection
                    parent1_idx = np.random.randint(0, len(sorted_individuals))
                    parent2_idx = np.random.randint(0, len(sorted_individuals))
                    
                    parent1 = sorted_individuals[parent1_idx][0]
                    parent2 = sorted_individuals[parent2_idx][0]
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    # Mutate children
                    child1 = self.mutate_solution(child1)
                    child2 = self.mutate_solution(child2)
                    
                    new_population.extend([child1, child2])
                
                population = new_population[:pop_size]
            else:
                # If no valid individuals, create new random ones
                population = []
                for i in range(pop_size):
                    base = self.generate_symmetric_initial()
                    variation = base + np.random.normal(0, 0.5, base.shape)
                    population.append(variation.flatten())
        
        return best_solution, best_outer_side_length

    def optimize_with_local_refinement(self, initial_solution: np.ndarray, max_iter: int = 100) -> Tuple[np.ndarray, float]:
        """Refine solution with local search approach"""
        current_solution = initial_solution.copy()
        current_outer_side = self.compute_outer_hexagon_side_length(current_solution.reshape(-1, 3))
        
        for iteration in range(max_iter):
            # Create a slightly modified version
            modified = current_solution.copy()
            
            # Perturb a few hexagons
            num_changes = min(3, len(modified) // 3)
            indices_to_change = np.random.choice(len(modified), num_changes, replace=False)
            
            for idx in indices_to_change:
                if idx % 3 == 2:  # Rotation
                    modified[idx] += np.random.normal(0, 5.0)
                    modified[idx] = modified[idx] % 360
                else:  # Position
                    modified[idx] += np.random.normal(0, 0.1)
            
            # Test the modified solution
            outer_side_length = self.compute_outer_hexagon_side_length(modified.reshape(-1, 3))
            fitness, is_valid = self.evaluate_fitness(modified, outer_side_length)
            
            if is_valid and fitness < -1.0/current_outer_side:
                current_solution = modified.copy()
                current_outer_side = outer_side_length
        
        return current_solution, current_outer_side

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
        packer = HexagonPacker()
        
        # Phase 1: Evolutionary optimization
        print("Starting evolutionary optimization...")
        best_solution, best_side_length = packer.optimize_with_evolution(max_generations=30, pop_size=15)
        
        # Phase 2: Local refinement
        print("Performing local refinement...")
        refined_solution, refined_side_length = packer.optimize_with_local_refinement(best_solution, max_iter=50)
        
        # Final validation
        final_hex_data = refined_solution.reshape(-1, 3)
        final_side_length = refined_side_length
        
        # Compute outer hexagon data
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
        
        # Calculate benchmark ratio
        inv_outer_hex_side_length = 1.0 / final_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537
        
        # Output metrics for verification
        print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")
        
        return final_hex_data, outer_hex_data, final_side_length
        
    except Exception as e:
        warnings.warn(f"Error in hexagon packing: {e}")
        # Fallback to simple symmetric arrangement
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
            [0, -4, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END