# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from typing import Tuple, List
import time

def hexagon_packing_11():
    """
    Evolves a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Hexagon vertices generator
    def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
        """Generate vertices of a regular hexagon with given center, rotation and side length."""
        angle_rad = np.radians(angle_deg)
        # Vertices of a unit hexagon centered at origin
        base_vertices = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        # Rotate and translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = base_vertices @ rotation_matrix.T
        translated_vertices = rotated_vertices + np.array([center_x, center_y])
        return translated_vertices
    
    # Check if hexagon is totally contained within outer hexagon
    def is_contained(hex_vertices, outer_hex_vertices):
        """Check if all vertices of hexagon are within outer hexagon."""
        hex_polygon = Polygon(hex_vertices)
        outer_polygon = Polygon(outer_hex_vertices)
        return outer_polygon.contains(hex_polygon)
    
    # Check if two hexagons intersect
    def do_intersect(hex1_vertices, hex2_vertices):
        """Check if two hexagons intersect using Shapely."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    # Calculate minimum distance between two hexagons
    def min_distance(hex1_vertices, hex2_vertices):
        """Calculate minimum distance between two hexagons."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.distance(poly2)
    
    # Create initial population
    def create_individual():
        """Create a valid random individual (11 hexagon positions and angles)."""
        individual = []
        for _ in range(11):
            # Random position within reasonable bounds for first hexagon
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            angle = np.random.uniform(0, 360)
            individual.append([x, y, angle])
        return np.array(individual)
    
    # Fitness calculation
    def evaluate_fitness(individual):
        """Evaluate how good a solution is based on overlap and containment."""
        # Create hexagon vertices for all inner hexagons
        hexagons = []
        for i, (x, y, angle) in enumerate(individual):
            vertices = generate_hexagon_vertices(x, y, angle)
            hexagons.append(vertices)
        
        # Check for overlaps
        overlap_count = 0
        total_pairs = 0
        for i in range(len(hexagons)):
            for j in range(i+1, len(hexagons)):
                total_pairs += 1
                if do_intersect(hexagons[i], hexagons[j]):
                    overlap_count += 1
        
        # Check containment in default outer hexagon (side length 10)
        outer_vertices = generate_hexagon_vertices(0, 0, 0, 10)
        containment_checks = [is_contained(hexagons[i], outer_vertices) for i in range(len(hexagons))]
        uncontained_count = sum(1 for check in containment_checks if not check)

        # Calculate total area covered by hexagons
        total_area = 11 * (3 * np.sqrt(3) / 2)  # Area of 11 unit hexagons
        
        # Fitness: penalize overlaps and non-containment
        # We want to minimize overlaps and maximize packing efficiency
        if overlap_count > 0 or uncontained_count > 0:
            return 0  # Invalid solution
        else:
            # Calculate effective radius of outer hexagon needed
            max_dist = 0
            for vertices in hexagons:
                # Find maximum distance from origin to any vertex
                distances = np.sqrt(np.sum((vertices)**2, axis=1))
                max_dist = max(max_dist, np.max(distances))
            
            # Outer hexagon needs to be slightly larger than max distance
            # Add some buffer to ensure full containment
            outer_radius_buffered = max_dist + 0.1
            
            # Convert radius to side length of hexagon (r = s * sqrt(3))
            outer_side_length = outer_radius_buffered / np.sqrt(3)
            # Return inverse of side length (we want to maximize 1/side_length so minimize side_length)
            return 1.0 / outer_side_length
    
    # Mutation operator
    def mutate(individual, generation, max_generations):
        """Mutate an individual with adaptive rate."""
        mutated = individual.copy()
        # Adaptive mutation rate - higher in early generations
        mut_rate = 0.5 * (1 - generation/max_generations) + 0.05
        for i in range(len(mutated)):
            if np.random.random() < mut_rate:
                # Mutate either position or angle
                if np.random.random() < 0.8:  # Mutate position
                    mutated[i][0] += np.random.normal(0, 0.5)
                    mutated[i][1] += np.random.normal(0, 0.5)
                else:  # Mutate angle
                    mutated[i][2] += np.random.normal(0, 30)
                    # Normalize angle to [0, 360)
                    mutated[i][2] = mutated[i][2] % 360
        return mutated
    
    # Crossover operator for hexagons
    def crossover(parent1, parent2):
        """Crossover two individuals by mixing their hexagon positions."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        # For each hexagon, decide which parent to take from
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()
        return child1, child2
    
    # Main evolutionary algorithm
    def evolve():
        # Parameters
        population_size = 50
        num_generations = 100
        elite_size = 5
        seed = 42
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        best_fitness = 0
        best_individual = None
        start_time = time.time()
        
        for generation in range(num_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = evaluate_fitness(individual)
                fitness_scores.append(fitness)
                
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]
            
            # Track best solution
            if sorted_fitness[0] > best_fitness:
                best_fitness = sorted_fitness[0]
                best_individual = sorted_population[0].copy()
            
            # Early stopping if we've found a good solution quickly
            if time.time() - start_time > 170:  # Leave 10 seconds for final processing
                break
                
            # Create new population
            new_population = []
            
            # Elite preservation
            for i in range(elite_size):
                new_population.append(sorted_population[i].copy())
            
            # Generate offspring via crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection
                parent1_idx = np.random.randint(0, population_size//2)
                parent2_idx = np.random.randint(0, population_size//2)
                parent1 = sorted_population[parent1_idx]
                parent2 = sorted_population[parent2_idx]
                
                # Crossover
                child1, child2 = crossover(parent1, parent2)
                
                # Mutation
                child1 = mutate(child1, generation, num_generations)
                child2 = mutate(child2, generation, num_generations)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:population_size]
        
        return best_individual, best_fitness
    
    # Run evolution
    best_solution, best_fitness = evolve()
    
    # Final validation and adjustments
    # Recalculate exact outer hexagon size needed
    def calculate_outer_hex_side_length(inner_hexas):
        # Create hexagon vertices for all inner hexagons
        hexagons = []
        for i, (x, y, angle) in enumerate(inner_hexas):
            vertices = generate_hexagon_vertices(x, y, angle)
            hexagons.append(vertices)
        
        # Find bounding box of all hexagons
        all_points = np.vstack(hexagons)
        min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        
        # Get the distance from origin to the farthest point
        distances_from_origin = np.sqrt(all_points[:, 0]**2 + all_points[:, 1]**2)
        max_distance = np.max(distances_from_origin)
        
        # Add buffer to make sure they fit
        outer_radius = max_distance + 0.2
        
        # Convert to side length of outer hexagon
        # For a regular hexagon, r = s * sqrt(3)
        outer_side_length = outer_radius / np.sqrt(3)
        
        return outer_side_length
    
    # Calculate final outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(best_solution)
    
    # Create output format as expected
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    return best_solution, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
