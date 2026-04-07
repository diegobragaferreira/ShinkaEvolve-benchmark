# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import Tuple, List, Optional, Any
import time
from dataclasses import dataclass
from collections import defaultdict
import math

# Set seeds for deterministic behavior
random.seed(42)
np.random.seed(42)

@dataclass
class Circle:
    """Represents a circle with x, y position and radius"""
    x: float
    y: float
    r: float
    
    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.r = float(self.r)
    
    def to_array(self) -> np.ndarray:
        """Convert circle to numpy array format [x, y, r]"""
        return np.array([self.x, self.y, self.r])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Circle':
        """Create circle from numpy array format [x, y, r]"""
        return cls(arr[0], arr[1], arr[2])

@dataclass
class CircleConfiguration:
    """Container for a complete circle packing configuration"""
    circles: List[Circle]
    
    def __post_init__(self):
        if not isinstance(self.circles, list):
            self.circles = list(self.circles)
    
    def to_numpy(self) -> np.ndarray:
        """Convert configuration to numpy array of shape (n, 3)"""
        return np.array([c.to_array() for c in self.circles])
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'CircleConfiguration':
        """Create configuration from numpy array of shape (n, 3)"""
        circles = [Circle.from_array(row) for row in arr]
        return cls(circles)
    
    def sum_radii(self) -> float:
        """Calculate sum of all circle radii"""
        return sum(c.r for c in self.circles)
    
    def copy(self) -> 'CircleConfiguration':
        """Create a deep copy of this configuration"""
        return CircleConfiguration([Circle(c.x, c.y, c.r) for c in self.circles])

class CircleValidator:
    """Handles validation of circle configurations against constraints"""
    
    @staticmethod
    def is_valid_position(circle: Circle, existing_circles: List[Circle]) -> bool:
        """Check if a circle position is valid (within bounds and no collisions)"""
        # Check boundary constraints
        if circle.x - circle.r < 0 or circle.x + circle.r > 1 or \
           circle.y - circle.r < 0 or circle.y + circle.r > 1:
            return False

        # Check collision with existing circles
        for existing_circle in existing_circles:
            if CircleValidator._circles_overlap(circle, existing_circle):
                return False

        return True
    
    @staticmethod
    def _circles_overlap(circle1: Circle, circle2: Circle) -> bool:
        """Check if two circles overlap using squared distances for efficiency"""
        dx = circle1.x - circle2.x
        dy = circle1.y - circle2.y
        distance_squared = dx * dx + dy * dy
        return distance_squared < (circle1.r + circle2.r) ** 2
    
    @staticmethod
    def validate_complete_configuration(circles: List[Circle]) -> Tuple[bool, float]:
        """Validate entire configuration and return (is_valid, penalty_score)"""
        penalty = 0.0
        total_radius = 0.0
        
        for i, circle in enumerate(circles):
            total_radius += circle.r
            
            # Boundary constraint penalty
            if circle.x - circle.r < 0 or circle.x + circle.r > 1 or \
               circle.y - circle.r < 0 or circle.y + circle.r > 1:
                penalty += 1000  # Large penalty for boundary violations
            
            # Overlap penalty with all previous circles
            for j in range(i):
                if CircleValidator._circles_overlap(circle, circles[j]):
                    overlap = (circles[i].r + circles[j].r) - \
                              math.sqrt((circle.x - circles[j].x)**2 + (circle.y - circles[j].y)**2)
                    penalty += overlap * 100
        
        return penalty == 0.0, penalty

class GreedyInitializer:
    """Provides methods for initializing circle configurations"""
    
    @staticmethod
    def place_circles_greedy(max_circles: int) -> List[Circle]:
        """Place circles greedily with maximum radius"""
        circles: List[Circle] = []
        
        # Predefined strategic positions for initial placement
        strategic_positions = [
            (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),  # corners
            (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),  # edges
            (0.5, 0.5),  # center
        ]
        
        # Place initial strategic circles
        placed = 0
        for i, (x, y) in enumerate(strategic_positions[:min(9, max_circles)]):
            if placed >= max_circles:
                break
            # Try to place with maximum possible radius
            max_radius = min(x, 1-x, y, 1-y)
            new_circle = Circle(x, y, max_radius)
            if CircleValidator.is_valid_position(new_circle, circles):
                circles.append(new_circle)
                placed += 1
        
        # Fill remaining spots with greedy approach
        while placed < max_circles:
            best_circle = None
            best_radius = 0.0
            
            # Try to place circles in multiple candidate positions
            candidates = []
            # Sample random positions in the square
            for _ in range(1000):
                x = random.uniform(0.01, 0.99)
                y = random.uniform(0.01, 0.99)
                # Estimate maximum radius for this position
                max_radius = min(x, 1-x, y, 1-y)
                candidates.append(Circle(x, y, max_radius))
            
            # Find the best valid circle among candidates
            for candidate in candidates:
                if candidate.r <= best_radius:
                    continue
                if CircleValidator.is_valid_position(candidate, circles):
                    best_circle = candidate
                    best_radius = candidate.r
            
            if best_circle is None:
                # If we can't find a valid circle, use a small radius and place anyway
                x = random.uniform(0.01, 0.99)
                y = random.uniform(0.01, 0.99)
                test_circle = Circle(x, y, 0.01)
                if CircleValidator.is_valid_position(test_circle, circles):
                    circles.append(test_circle)
                    placed += 1
                else:
                    break  # Can't place more circles
            else:
                circles.append(best_circle)
                placed += 1
        
        return circles

class CircleFitnessCalculator:
    """Handles fitness calculations for circle configurations"""
    
    @staticmethod
    def calculate_fitness(config: CircleConfiguration) -> Tuple[float, float]:
        """
        Calculate fitness for circle packing configuration.
        Returns (sum_of_radii, penalty_score)
        """
        circles = config.circles
        n = len(circles)
        
        # Calculate sum of radii (primary objective)
        sum_radii = config.sum_radii()
        
        # Penalty for constraint violations
        penalty = 0.0
        
        # Boundary constraint penalty (negative if outside bounds)
        for circle in circles:
            if circle.x - circle.r < 0 or circle.x + circle.r > 1 or \
               circle.y - circle.r < 0 or circle.y + circle.r > 1:
                penalty += 1000  # Large penalty for boundary violations
        
        # Overlap penalty (positive if overlapping)
        # Only check pairs that could potentially overlap
        for i in range(n):
            for j in range(i+1, n):
                c1 = circles[i]
                c2 = circles[j]
                
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                distance_squared = dx * dx + dy * dy
                distance = math.sqrt(distance_squared)
                
                if distance < c1.r + c2.r:  # Circles overlap
                    overlap = (c1.r + c2.r - distance) * 100
                    penalty += overlap
        
        # Return sum of radii with penalty
        return sum_radii, penalty

class EvolutionaryOptimizer:
    """Handles evolutionary optimization of circle configurations"""
    
    def __init__(self, pop_size: int = 50, generations: int = 100,
                 crossover_prob: float = 0.8, mutation_prob: float = 0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
    
    def _initialize_population(self, n_circles: int) -> List[CircleConfiguration]:
        """Initialize population with diverse circle configurations"""
        population = []
        
        # Create initial configurations using different strategies
        for _ in range(self.pop_size):
            # Strategy 1: Greedy placement with random perturbation
            circles = GreedyInitializer.place_circles_greedy(n_circles)
            
            # Add small random perturbations to improve diversity
            for circle in circles:
                if random.random() < 0.3:  # 30% chance to perturb
                    circle.x += np.random.normal(0, 0.02)
                    circle.y += np.random.normal(0, 0.02)
                    circle.x = np.clip(circle.x, 0.01, 0.99)
                    circle.y = np.clip(circle.y, 0.01, 0.99)
            
            population.append(CircleConfiguration(circles))
        
        return population
    
    def _evaluate_individual(self, config: CircleConfiguration) -> float:
        """Evaluate individual and return fitness value"""
        sum_radii, penalty = CircleFitnessCalculator.calculate_fitness(config)
        # Fitness is sum of radii minus penalty (since we maximize sum_radii)
        # We penalize invalid configurations heavily
        fitness = sum_radii - penalty
        return fitness
    
    def _mutate_individual(self, config: CircleConfiguration) -> CircleConfiguration:
        """Mutate a circle configuration"""
        mutated_config = config.copy()
        
        for circle in mutated_config.circles:
            if random.random() < self.mutation_prob:
                # Mutate position
                circle.x += np.random.normal(0, 0.01)
                circle.y += np.random.normal(0, 0.01)
                # Clamp to valid range
                circle.x = np.clip(circle.x, 0.01, 0.99)
                circle.y = np.clip(circle.y, 0.01, 0.99)
                
            if random.random() < self.mutation_prob:
                # Mutate radius
                circle.r += np.random.normal(0, 0.005)
                # Ensure positive radius
                circle.r = max(0.005, circle.r)
                
                # Adjust position if needed due to radius change
                max_radius = min(circle.x, 1-circle.x, circle.y, 1-circle.y)
                if circle.r > max_radius:
                    circle.r = max_radius * 0.9  # Scale down radius
        
        return mutated_config
    
    def _crossover(self, config1: CircleConfiguration, config2: CircleConfiguration) -> Tuple[CircleConfiguration, CircleConfiguration]:
        """Perform crossover between two configurations"""
        # Simple uniform crossover
        child1_circles = []
        child2_circles = []
        
        n = min(len(config1.circles), len(config2.circles))
        
        for i in range(n):
            if random.random() < 0.5:
                child1_circles.append(Circle(config1.circles[i].x, config1.circles[i].y, config1.circles[i].r))
                child2_circles.append(Circle(config2.circles[i].x, config2.circles[i].y, config2.circles[i].r))
            else:
                child1_circles.append(Circle(config2.circles[i].x, config2.circles[i].y, config2.circles[i].r))
                child2_circles.append(Circle(config1.circles[i].x, config1.circles[i].y, config1.circles[i].r))
        
        return CircleConfiguration(child1_circles), CircleConfiguration(child2_circles)
    
    def optimize(self, n_circles: int = 32) -> CircleConfiguration:
        """Main optimization loop using evolutionary algorithm"""
        # Initialize population
        population = self._initialize_population(n_circles)
        
        # Evaluate initial population
        fitness_scores = [self._evaluate_individual(individual) for individual in population]
        
        # Evolution loop
        for gen in range(self.generations):
            # Select parents (tournament selection)
            selected_indices = []
            for _ in range(self.pop_size):
                tournament_size = 3
                tournament_indices = random.sample(range(self.pop_size), tournament_size)
                winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
                selected_indices.append(winner_index)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            best_idx = max(range(self.pop_size), key=lambda i: fitness_scores[i])
            new_population.append(population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1_idx = selected_indices[len(new_population) // 2]
                parent2_idx = selected_indices[len(new_population) // 2 + 1]
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                if random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                # Mutation
                child1 = self._mutate_individual(child1)
                child2 = self._mutate_individual(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.pop_size]
            
            # Evaluate new population
            fitness_scores = [self._evaluate_individual(individual) for individual in population]
        
        # Return best individual
        best_idx = max(range(self.pop_size), key=lambda i: fitness_scores[i])
        return population[best_idx]

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        # Start with greedy initialization for good baseline
        start_time = time.time()
        
        # Create optimizer instance
        optimizer = EvolutionaryOptimizer(pop_size=50, generations=100)
        
        # Run optimization
        optimized_config = optimizer.optimize(n_circles=32)
        
        # Validate final result
        sum_radii, penalty = CircleFitnessCalculator.calculate_fitness(optimized_config)
        
        # Convert to numpy array format
        circles = optimized_config.to_numpy()
        
        end_time = time.time()
        
        # Early validation and fallback logic to ensure valid result
        if penalty > 100:  # High penalty indicates serious constraint violations
            # Return greedy solution as fallback
            print("Optimization produced invalid configuration, using greedy solution")
            greedy_circles = GreedyInitializer.place_circles_greedy(32)
            return np.array([c.to_array() for c in greedy_circles])
        
        return circles
    
    except Exception as e:
        # Fallback in case of any error
        print(f"Error in optimization: {e}")
        # Return greedy solution as final fallback
        greedy_circles = GreedyInitializer.place_circles_greedy(32)
        return np.array([c.to_array() for c in greedy_circles])

# EVOLVE-BLOCK-END
