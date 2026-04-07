# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import random
from typing import Tuple, List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class ConstraintValidator:
    """Efficient constraint validation system."""
    
    @staticmethod
    def validate_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
        """Validate that all circles are within bounds and non-overlapping."""
        n = len(circles)
        
        # Check boundary constraints efficiently
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False

        # Check overlap constraints with early termination
        if n > 1:
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Use optimized distance matrix computation
            distances = cdist(positions, positions)
            
            # Check for overlaps with minimal overhead
            for i in range(n):
                for j in range(i+1, n):
                    if distances[i, j] < (radii[i] + radii[j]):
                        return False
                        
        return True

class PatternGenerator:
    """Generate diverse initial patterns for better exploration."""
    
    @staticmethod
    def generate_hexagonal_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Generate hexagonal lattice pattern."""
        circles = np.zeros((n, 3))
        
        # Determine grid size based on rectangle dimensions
        aspect_ratio = rect_width / rect_height
        rows = int(np.ceil(np.sqrt(n / aspect_ratio)) + 1)
        cols = int(np.ceil(n / rows) + 1)
        
        # Calculate spacing
        spacing_x = rect_width / (cols + 1) if cols > 0 else rect_width
        spacing_y = rect_height / (rows + 1) if rows > 0 else rect_height
        
        # Use actual hexagonal spacing: horizontal spacing = 2*r, vertical spacing = sqrt(3)*r
        min_radius = min(rect_width, rect_height) * 0.05
        hex_spacing_x = min_radius * 2.0
        hex_spacing_y = min_radius * np.sqrt(3.0)
        
        # Take the smaller spacing to ensure good packing
        actual_spacing_x = min(spacing_x, hex_spacing_x)
        actual_spacing_y = min(spacing_y, hex_spacing_y)
        
        placed = 0
        for row in range(rows):
            if placed >= n:
                break
            for col in range(cols):
                if placed >= n:
                    break
                    
                # Offset every other row for hexagonal arrangement
                offset = (row % 2) * (actual_spacing_x / 2)
                x = offset + col * actual_spacing_x + actual_spacing_x / 2
                y = row * actual_spacing_y + actual_spacing_y / 2
                
                # Ensure within bounds
                x = np.clip(x, min_radius, rect_width - min_radius)
                y = np.clip(y, min_radius, rect_height - min_radius)
                
                # Adjust radius to prevent boundary issues
                max_radius = min(x, y, rect_width - x, rect_height - y)
                r = min(min_radius, max_radius * 0.9)
                
                circles[placed] = [x, y, r]
                placed += 1
                
        # Fill remaining positions
        for i in range(placed, n):
            x = np.random.uniform(min_radius, rect_width - min_radius)
            y = np.random.uniform(min_radius, rect_height - min_radius)
            r = np.random.uniform(0.005, min_radius * 0.5)
            circles[i] = [x, y, r]
            
        return circles

    @staticmethod
    def generate_triangular_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Generate triangular lattice pattern."""
        circles = np.zeros((n, 3))
        
        # Determine grid size
        sqrt_n = int(np.ceil(np.sqrt(n)))
        aspect_ratio = rect_width / rect_height
        cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
        rows = int(np.ceil(n / cols))
        
        # Calculate spacing
        spacing_x = rect_width / (cols + 1) if cols > 0 else rect_width
        spacing_y = rect_height / (rows + 1) if rows > 0 else rect_height
        
        # Adjust for triangular packing
        min_radius = min(rect_width, rect_height) * 0.05
        tri_spacing_x = min_radius * 2.0
        tri_spacing_y = min_radius * np.sqrt(3.0)
        
        spacing_x = min(spacing_x, tri_spacing_x)
        spacing_y = min(spacing_y, tri_spacing_y)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Slight offset for triangular pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Keep within bounds
                x = np.clip(x, min_radius, rect_width - min_radius)
                y = np.clip(y, min_radius, rect_height - min_radius)
                
                circles[idx] = [x, y, min_radius]
                idx += 1
            if idx >= n:
                break
                
        return circles

    @staticmethod
    def generate_square_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Generate square grid pattern."""
        circles = np.zeros((n, 3))
        
        # Optimized for rectangle dimensions
        sqrt_n = int(np.ceil(np.sqrt(n)))
        aspect_ratio = rect_width / rect_height
        cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
        rows = int(np.ceil(n / cols))
        
        spacing_x = rect_width / (cols + 1) if cols > 0 else rect_width
        spacing_y = rect_height / (rows + 1) if rows > 0 else rect_height
        
        # Conservative approach to radius sizing
        min_radius = min(rect_width, rect_height) * 0.05
        spacing_x = max(spacing_x, min_radius * 2)
        spacing_y = max(spacing_y, min_radius * 2)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Keep within bounds
                x = np.clip(x, min_radius, rect_width - min_radius)
                y = np.clip(y, min_radius, rect_height - min_radius)
                
                circles[idx] = [x, y, min_radius]
                idx += 1
            if idx >= n:
                break
                
        return circles

    @staticmethod
    def generate_random_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Generate random initial pattern."""
        circles = np.zeros((n, 3))
        
        # Generate random positions and radii
        for i in range(n):
            x = np.random.uniform(0.05 * rect_width, rect_width * 0.95)
            y = np.random.uniform(0.05 * rect_height, rect_height * 0.95)
            
            # Start with reasonable radius
            max_radius = min(rect_width, rect_height) * 0.1
            r = np.random.uniform(0.02 * max_radius, 0.5 * max_radius)
            
            circles[i] = [x, y, r]
            
        return circles

class AdaptiveMutator:
    """Intelligent mutation system based on local constraint density."""
    
    def __init__(self):
        self.position_base_delta = 0.05
        self.radius_base_delta = 0.01

    def compute_constraint_density(self, circles: np.ndarray) -> np.ndarray:
        """Compute constraint density for each circle based on neighbors."""
        n = len(circles)
        density_scores = np.zeros(n)
        
        for i in range(n):
            x1, y1, r1 = circles[i]
            nearby_count = 0
            
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < (r1 + r2):
                        nearby_count += 1
                        
            density_scores[i] = nearby_count
            
        return density_scores

    def compute_voronoi_area(self, circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Compute Voronoi cell areas for each circle."""
        try:
            # Add boundary points for proper Voronoi calculation
            points = circles[:, :2].copy()
            
            # Add boundary points to make Voronoi more meaningful
            boundary_points = [
                [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
                [rect_width/2, 0], [rect_width/2, rect_height],
                [0, rect_height/2], [rect_width, rect_height/2]
            ]
            points = np.vstack([points, boundary_points])
            
            vor = Voronoi(points)
            
            # For each original point, compute Voronoi cell area
            areas = []
            for i in range(len(circles)):
                region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1
                
                if region_idx != -1 and region_idx < len(vor.regions):
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) >= 3:
                        # Compute area using shoelace formula
                        vertices = np.array([vor.vertices[j] for j in region])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            areas.append(area)
                        else:
                            areas.append(1.0)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
                    
            return np.array(areas)
        except:
            # Fallback to uniform distribution if Voronoi fails
            return np.ones(len(circles))

    def mutate_radius(self, circles: np.ndarray, idx: int, 
                      voronoi_areas: np.ndarray = None,
                      constraint_densities: np.ndarray = None) -> np.ndarray:
        """Mutate radius with adaptive delta."""
        new_circles = circles.copy()
        old_r = new_circles[idx, 2]
        
        # Adaptive delta based on Voronoi area (smaller area = denser = smaller mutation)
        delta = self.radius_base_delta
        
        if voronoi_areas is not None and len(voronoi_areas) > idx:
            # Use inverse relationship: smaller Voronoi area = more constrained
            area_factor = max(0.1, 1.0 / (1.0 + voronoi_areas[idx] * 0.1))
            delta *= area_factor
            
        # Adjust based on constraint density
        if constraint_densities is not None and len(constraint_densities) > idx:
            constraint_factor = 1.0 / (1.0 + constraint_densities[idx] * 0.2)
            delta *= constraint_factor
            
        # Apply small random perturbation
        delta_r = np.random.uniform(-delta, delta)
        new_r = old_r + delta_r
        
        # Ensure positive radius
        new_r = max(0.001, new_r)
        new_circles[idx, 2] = new_r
        
        return new_circles

    def mutate_position(self, circles: np.ndarray, idx: int,
                        voronoi_areas: np.ndarray = None,
                        constraint_densities: np.ndarray = None) -> np.ndarray:
        """Mutate position with adaptive delta."""
        new_circles = circles.copy()
        old_x, old_y = new_circles[idx, 0], new_circles[idx, 1]
        
        # Adaptive delta based on Voronoi area
        delta = self.position_base_delta
        
        if voronoi_areas is not None and len(voronoi_areas) > idx:
            # Use inverse relationship: smaller Voronoi area = more constrained
            area_factor = max(0.1, 1.0 / (1.0 + voronoi_areas[idx] * 0.1))
            delta *= area_factor
            
        # Adjust based on constraint density
        if constraint_densities is not None and len(constraint_densities) > idx:
            constraint_factor = 1.0 / (1.0 + constraint_densities[idx] * 0.2)
            delta *= constraint_factor
            
        # Apply small random perturbation
        delta_x = np.random.uniform(-delta, delta)
        delta_y = np.random.uniform(-delta, delta)
        
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        
        # Ensure within bounds
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        
        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
        
        return new_circles

class CirclePacker:
    """Main circle packing optimizer using enhanced evolutionary approach."""
    
    def __init__(self, rect_width: float = 1.0, rect_height: float = 1.0, n_circles: int = 21):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.n_circles = n_circles
        self.validator = ConstraintValidator()
        self.pattern_gen = PatternGenerator()
        self.mutator = AdaptiveMutator()
        self.population_size = 50
        self.generations = 150
        self.elite_size = 10
        self.tournament_size = 7

    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Calculate fitness as sum of radii with constraint penalty."""
        if not self.validator.validate_solution(circles, self.rect_width, self.rect_height):
            return -np.inf
            
        return np.sum(circles[:, 2])

    def local_refinement(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply local refinement to improve solution quality."""
        current_circles = circles.copy()
        best_circles = current_circles.copy()
        best_fitness = self.evaluate_fitness(current_circles)
        
        # Compute constraint and Voronoi metrics once
        constraint_densities = self.mutator.compute_constraint_density(current_circles)
        voronoi_areas = self.mutator.compute_voronoi_area(current_circles, self.rect_width, self.rect_height)
        
        for iteration in range(max_iter):
            improved = False
            
            # Process circles in order of increasing constraint density (most constrained first)
            sorted_indices = np.argsort(constraint_densities)
            
            for idx in sorted_indices:
                # Try both position and radius mutations
                mutated_pos = self.mutator.mutate_position(
                    current_circles, idx, voronoi_areas, constraint_densities
                )
                
                mutated_rad = self.mutator.mutate_radius(
                    current_circles, idx, voronoi_areas, constraint_densities
                )
                
                # Evaluate both mutations
                pos_fitness = self.evaluate_fitness(mutated_pos)
                rad_fitness = self.evaluate_fitness(mutated_rad)
                
                # Choose the better valid mutation
                if pos_fitness > rad_fitness:
                    if self.validator.validate_solution(mutated_pos, self.rect_width, self.rect_height):
                        current_circles = mutated_pos
                        improved = True
                else:
                    if self.validator.validate_solution(mutated_rad, self.rect_width, self.rect_height):
                        current_circles = mutated_rad
                        improved = True
                        
            # Update best solution
            current_fitness = self.evaluate_fitness(current_circles)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_circles = current_circles.copy()
                
        return best_circles

    def optimize(self) -> np.ndarray:
        """Main optimization routine with multi-strategy enhancement."""
        # Try multiple initialization strategies
        initial_patterns = [
            self.pattern_gen.generate_hexagonal_pattern(self.n_circles, self.rect_width, self.rect_height),
            self.pattern_gen.generate_triangular_pattern(self.n_circles, self.rect_width, self.rect_height),
            self.pattern_gen.generate_square_pattern(self.n_circles, self.rect_width, self.rect_height),
            self.pattern_gen.generate_random_pattern(self.n_circles, self.rect_width, self.rect_height)
        ]
        
        best_solution = None
        best_score = -np.inf
        
        # Multi-start optimization
        for i, seed_pattern in enumerate(initial_patterns):
            # Apply local refinement to improve initial quality
            refined_seed = self.local_refinement(seed_pattern, max_iter=30)
            
            # Perform evolutionary search on this seed
            evolved_solution = self._evolutionary_search(refined_seed)
            
            # Final local refinement
            final_solution = self.local_refinement(evolved_solution, max_iter=50)
            
            score = self.evaluate_fitness(final_solution)
            if score > best_score and self.validator.validate_solution(final_solution, self.rect_width, self.rect_height):
                best_score = score
                best_solution = final_solution.copy()
                
        # Return the best solution found
        if best_solution is None:
            # Fallback to a decent initial pattern with refinement
            fallback_pattern = self.pattern_gen.generate_hexagonal_pattern(self.n_circles, self.rect_width, self.rect_height)
            best_solution = self.local_refinement(fallback_pattern, max_iter=100)
            
        return best_solution

    def _evolutionary_search(self, initial_solution: np.ndarray) -> np.ndarray:
        """Run evolutionary search on provided initial solution."""
        # Initialize population with variations of the initial solution
        population = [initial_solution.copy()]
        
        # Add diverse variants
        for _ in range(self.population_size - 1):
            variant = initial_solution.copy()
            # Add small random noise to each circle
            for i in range(self.n_circles):
                if np.random.random() < 0.5:
                    variant[i, 0] += np.random.uniform(-0.02, 0.02)
                    variant[i, 1] += np.random.uniform(-0.02, 0.02)
                    variant[i, 2] += np.random.uniform(-0.005, 0.005)
            # Enforce bounds
            for i in range(self.n_circles):
                variant[i, 0] = np.clip(variant[i, 0], variant[i, 2], self.rect_width - variant[i, 2])
                variant[i, 1] = np.clip(variant[i, 1], variant[i, 2], self.rect_height - variant[i, 2])
                variant[i, 2] = max(0.001, variant[i, 2])
            population.append(variant)
            
        # Evolutionary algorithm
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(individual) for individual in population]
            
            # Sort by fitness descending
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Keep elite
            elite = population[:self.elite_size]
            
            # Generate new population
            new_population = elite[:]
            
            # Create offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]
                parent2_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]
                
                parent1 = population[parent1_idx].copy()
                parent2 = population[parent2_idx].copy()
                
                # Crossover (uniform)
                child1 = parent1.copy()
                child2 = parent2.copy()
                for i in range(self.n_circles):
                    if np.random.random() < 0.5:
                        child1[i] = parent2[i]
                        child2[i] = parent1[i]
                        
                # Mutate children
                child1 = self._adaptive_mutation(child1)
                child2 = self._adaptive_mutation(child2)
                
                # Repair to enforce constraints
                child1 = self._repair_solution(child1)
                child2 = self._repair_solution(child2)
                
                new_population.extend([child1, child2])
                
            population = new_population[:self.population_size]
            
        # Return best solution from final population
        fitness_scores = [self.evaluate_fitness(individual) for individual in population]
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]

    def _adaptive_mutation(self, circles: np.ndarray) -> np.ndarray:
        """Apply adaptive mutation with Voronoi-based scaling."""
        mutated = circles.copy()
        
        # Compute metrics once
        constraint_densities = self.mutator.compute_constraint_density(mutated)
        voronoi_areas = self.mutator.compute_voronoi_area(mutated, self.rect_width, self.rect_height)
        
        # Apply mutations to each circle
        for i in range(self.n_circles):
            if np.random.random() < 0.3:  # Mutation probability
                # Choose mutation type
                mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])
                
                if mutation_type == 'position':
                    mutated = self.mutator.mutate_position(
                        mutated, i, voronoi_areas, constraint_densities
                    )
                else:
                    mutated = self.mutator.mutate_radius(
                        mutated, i, voronoi_areas, constraint_densities
                    )
                    
        return mutated

    def _repair_solution(self, circles: np.ndarray) -> np.ndarray:
        """Repair solution to ensure all constraints are met."""
        repaired = circles.copy()
        
        # Ensure positive radii
        repaired[:, 2] = np.maximum(repaired[:, 2], 0.001)
        
        # Enforce bounds
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            x = np.clip(x, r, self.rect_width - r)
            y = np.clip(y, r, self.rect_height - r)
            repaired[i] = [x, y, r]
            
        return repaired

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4 => width + height = 2
    # Using 1.2 width and 0.8 height for better packing efficiency
    rect_width = 1.2
    rect_height = 0.8
    
    # Create packer and optimize
    packer = CirclePacker(rect_width, rect_height, 21)
    circles = packer.optimize()
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")