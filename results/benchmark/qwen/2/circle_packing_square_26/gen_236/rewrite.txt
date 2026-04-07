# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

class PhysicsGuidedCirclePacker:
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.max_iterations = 1000
        
    def validate_circles(self, circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and non-overlapping."""
        if len(circles) != self.n_circles:
            return False
            
        # Check containment constraints
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1 - r or y < r or y > 1 - r:
                return False
        
        # Check overlap constraints
        for i in range(self.n_circles):
            x1, y1, r1 = circles[i]
            for j in range(i+1, self.n_circles):
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
        
        return True

    def calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate total radius sum as fitness."""
        return np.sum(circles[:, 2])

    def physics_initialization(self) -> np.ndarray:
        """Initialize circles using physics-based repulsion to avoid overlaps."""
        circles = np.zeros((self.n_circles, 3))
        
        # Start with a hexagonal grid-like pattern
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Place circles in a grid with slight randomness
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= self.n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                # Add randomness to avoid perfect grid
                x += np.random.normal(0, spacing_x * 0.1)
                y += np.random.normal(0, spacing_y * 0.1)
                # Clip to valid range
                x = np.clip(x, spacing_x * 0.1, 1 - spacing_x * 0.1)
                y = np.clip(y, spacing_y * 0.1, 1 - spacing_y * 0.1)
                
                # Calculate max possible radius at this position
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                r = min(min_dist_to_edge * 0.4, 0.15)
                
                circles[count] = [x, y, r]
                count += 1
            if count >= self.n_circles:
                break
        
        # Apply physics repulsion to remove overlaps
        self._apply_repulsion_force(circles, max_iterations=100)
        
        return circles

    def _apply_repulsion_force(self, circles: np.ndarray, max_iterations: int = 100) -> None:
        """Apply physics repulsion to resolve overlaps."""
        for iteration in range(max_iterations):
            any_movement = False
            
            # Calculate forces between overlapping circles
            for i in range(self.n_circles):
                x1, y1, r1 = circles[i]
                force_x, force_y = 0.0, 0.0
                
                for j in range(self.n_circles):
                    if i == j:
                        continue
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if distance < r1 + r2:
                        # Repulsion force
                        if distance > 0.001:
                            force_magnitude = (r1 + r2 - distance) * 0.1
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            force_x += dx * force_magnitude
                            force_y += dy * force_magnitude
                        else:
                            # Random force if too close
                            force_x += np.random.normal(0, 0.01)
                            force_y += np.random.normal(0, 0.01)
                
                # Apply force with bounds checking
                if abs(force_x) > 0 or abs(force_y) > 0:
                    new_x = x1 + force_x
                    new_y = y1 + force_y
                    # Keep within bounds
                    new_x = np.clip(new_x, r1, 1 - r1)
                    new_y = np.clip(new_y, r1, 1 - r1)
                    
                    if new_x != x1 or new_y != y1:
                        circles[i, 0] = new_x
                        circles[i, 1] = new_y
                        any_movement = True
            
            # Stop if no movement occurred
            if not any_movement:
                break

    def adaptive_mutate(self, circles: np.ndarray) -> np.ndarray:
        """Specialized mutation that considers geometric relationships."""
        mutated = circles.copy()
        
        # Determine mutation intensity based on current fitness
        current_fitness = self.calculate_fitness(mutated)
        base_intensity = max(0.01, 0.1 - current_fitness * 0.001)
        
        for i in range(self.n_circles):
            if np.random.random() < 0.3:  # 30% mutation rate
                # Choose mutation type based on circle importance
                if np.random.random() < 0.7:  # Position mutation
                    # Mutate position with adaptive intensity
                    intensity = base_intensity * (0.5 + np.random.random() * 0.5)
                    mutated[i, 0] += np.random.normal(0, intensity)
                    mutated[i, 1] += np.random.normal(0, intensity)
                    # Keep within bounds
                    mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                    mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
                else:  # Radius mutation
                    # Mutate radius with smaller changes
                    intensity = base_intensity * 0.3
                    mutated[i, 2] *= np.exp(np.random.normal(0, intensity))
                    mutated[i, 2] = max(0.001, mutated[i, 2])
        
        return mutated

    def geometric_local_search(self, circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Perform geometric optimization that modifies circle positions and radii."""
        optimized = circles.copy()
        
        # Local optimization using geometric transformations
        for iteration in range(max_iterations):
            improved = False
            
            # Try to expand radii where possible
            for i in range(self.n_circles):
                x, y, r = optimized[i]
                original_r = r
                
                # Try to increase radius safely
                max_possible_r = min(x, 1-x, y, 1-y) - 0.001
                
                # Check if we can increase radius
                candidates = []
                test_steps = [0.01, 0.005, 0.002]
                
                for step in test_steps:
                    new_r = min(r + step, max_possible_r)
                    if new_r > r:
                        # Test if this new radius works
                        temp_circles = optimized.copy()
                        temp_circles[i, 2] = new_r
                        
                        # Check constraints
                        valid = True
                        for j in range(self.n_circles):
                            if i != j:
                                dist = np.sqrt((temp_circles[i, 0] - temp_circles[j, 0])**2 + 
                                             (temp_circles[i, 1] - temp_circles[j, 1])**2)
                                if dist < new_r + temp_circles[j, 2]:
                                    valid = False
                                    break
                        
                        if valid:
                            candidates.append((new_r, step))
                
                # Choose best candidate
                if candidates:
                    best_r, _ = max(candidates, key=lambda x: x[0])
                    optimized[i, 2] = best_r
                    improved = True
            
            # Try position adjustments
            for i in range(self.n_circles):
                x, y, r = optimized[i]
                
                # Try small movements
                best_x, best_y = x, y
                best_r = r
                best_fitness = self.calculate_fitness(optimized)
                
                # Sample movement directions
                moves = [(0, 0), (0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)]
                
                for dx, dy in moves:
                    new_x = x + dx
                    new_y = y + dy
                    new_x = np.clip(new_x, r, 1 - r)
                    new_y = np.clip(new_y, r, 1 - r)
                    
                    # Test if these adjustments improve the configuration
                    temp_circles = optimized.copy()
                    temp_circles[i, 0] = new_x
                    temp_circles[i, 1] = new_y
                    
                    # Check if it's valid
                    valid = True
                    for j in range(self.n_circles):
                        if i != j:
                            dist = np.sqrt((temp_circles[i, 0] - temp_circles[j, 0])**2 + 
                                         (temp_circles[i, 1] - temp_circles[j, 1])**2)
                            if dist < temp_circles[i, 2] + temp_circles[j, 2]:
                                valid = False
                                break
                    
                    if valid:
                        new_fitness = self.calculate_fitness(temp_circles)
                        if new_fitness > best_fitness:
                            best_x, best_y = new_x, new_y
                            best_fitness = new_fitness
                
                if best_x != x or best_y != y:
                    optimized[i, 0] = best_x
                    optimized[i, 1] = best_y
                    improved = True
            
            # Early termination if no improvement
            if not improved:
                break
        
        return optimized

    def evolve(self, generations: int = 500) -> np.ndarray:
        """Main evolutionary algorithm with physics-guided initialization."""
        # Initialize population with physics-based approach
        population = []
        for _ in range(20):  # Smaller population since we're more efficient
            circles = self.physics_initialization()
            if not self.validate_circles(circles):
                # Fallback to grid initialization
                circles = self._grid_initialization()
            population.append(circles)
        
        best_fitness = -float('inf')
        best_individual = None
        
        for generation in range(generations):
            # Evaluate fitness
            fitnesses = [self.calculate_fitness(circles) for circles in population]
            
            # Track best
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
            
            # Print progress
            if generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
            
            # Selection based on fitness
            # Use tournament selection with size 3
            selected = []
            for _ in range(len(population)):
                tournament_indices = np.random.choice(len(population), 3, replace=False)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
                selected.append(population[winner_idx])
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            if best_individual is not None:
                new_population.append(best_individual)
            
            # Generate offspring
            while len(new_population) < len(population):
                # Select parents
                parent1 = selected[np.random.randint(0, len(selected))]
                parent2 = selected[np.random.randint(0, len(selected))]
                
                # Crossover (uniform)
                child = parent1.copy()
                mask = np.random.rand(*parent1.shape) > 0.5
                child[mask] = parent2[mask]
                
                # Mutation with physics-inspired approach
                child = self.adaptive_mutate(child)
                
                # Local search refinement
                child = self.geometric_local_search(child)
                
                # Ensure validity
                if self.validate_circles(child):
                    new_population.append(child)
                else:
                    # Repair if needed
                    new_population.append(self._repair_invalid(child))
            
            population = new_population[:len(population)]
        
        # Final optimization on best individual
        if best_individual is not None:
            final_result = self.geometric_local_search(best_individual, max_iterations=100)
            if self.validate_circles(final_result):
                return final_result
            else:
                return best_individual
        else:
            # Fallback to best from final population
            fitnesses = [self.calculate_fitness(circles) for circles in population]
            best_idx = np.argmax(fitnesses)
            return self.geometric_local_search(population[best_idx], max_iterations=100)

    def _grid_initialization(self) -> np.ndarray:
        """Fallback grid initialization."""
        circles = np.zeros((self.n_circles, 3))
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= self.n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                r = min(spacing_x, spacing_y) * 0.3
                circles[count] = [x, y, r]
                count += 1
            if count >= self.n_circles:
                break
        
        return circles

    def _repair_invalid(self, circles: np.ndarray) -> np.ndarray:
        """Repair invalid configuration."""
        repaired = circles.copy()
        # Apply physics-based repair
        self._apply_repulsion_force(repaired, max_iterations=50)
        return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    try:
        packer = PhysicsGuidedCirclePacker(n_circles=26)
        circles = packer.evolve(generations=500)
        
        # Validate result
        if not packer.validate_circles(circles):
            # Fallback if validation fails
            circles = packer._grid_initialization()
            circles = packer._repair_invalid(circles)
        
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Physics-guided evolution completed in {eval_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during physics-guided evolution: {e}")
        # Fallback to simple grid
        packer = PhysicsGuidedCirclePacker(n_circles=26)
        circles = packer._grid_initialization()
    
    return circles

# EVOLVE-BLOCK-END