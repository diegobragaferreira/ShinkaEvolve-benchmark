# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
import math
from typing import Tuple, List
import warnings

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    class ConstraintPropagationOptimizer:
        def __init__(self, n_circles: int = 26):
            self.n_circles = n_circles
            self.max_iterations = 10000
            self.temperature = 1.0
            self.cooling_rate = 0.9995
            self.min_temperature = 0.001
            self.max_local_iterations = 100
            
        def initialize_positions(self) -> np.ndarray:
            """Initialize positions using a more structured approach"""
            individual = np.zeros((self.n_circles, 3))
            
            # Distribute circles in a grid-like pattern
            rows = int(np.ceil(np.sqrt(self.n_circles)))
            cols = int(np.ceil(self.n_circles / rows))
            
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)
            
            for i in range(self.n_circles):
                row = i // cols
                col = i % cols
                
                base_x = (col + 1) * spacing_x
                base_y = (row + 1) * spacing_y
                
                # Add small random perturbation
                individual[i, 0] = np.clip(base_x + np.random.uniform(-spacing_x/6, spacing_x/6), 0.01, 0.99)
                individual[i, 1] = np.clip(base_y + np.random.uniform(-spacing_y/6, spacing_y/6), 0.01, 0.99)
                
                # Start with a reasonable radius
                max_radius = min(0.5 - individual[i, 0], 0.5 - individual[i, 1],
                               individual[i, 0], individual[i, 1])
                individual[i, 2] = max_radius * 0.3
                
            return individual
        
        def calculate_constraints_violation(self, individual: np.ndarray) -> Tuple[float, float, float]:
            """Calculate all types of constraint violations"""
            # Containment violations
            containment_penalty = 0.0
            for i in range(len(individual)):
                x, y, r = individual[i]
                # Calculate boundary violations
                left_violation = max(0, r - x)
                right_violation = max(0, x + r - 1)
                bottom_violation = max(0, r - y)
                top_violation = max(0, y + r - 1)
                
                if left_violation > 0 or right_violation > 0 or bottom_violation > 0 or top_violation > 0:
                    containment_penalty += 1000 * (left_violation + right_violation + bottom_violation + top_violation)
            
            # Overlap violations
            overlap_penalty = 0.0
            if len(individual) > 1:
                distances = cdist(individual[:, :2], individual[:, :2])
                for i in range(len(individual)):
                    for j in range(i+1, len(individual)):
                        distance = distances[i, j]
                        r1, r2 = individual[i, 2], individual[j, 2]
                        if distance < r1 + r2:
                            overlap = (r1 + r2) - distance
                            overlap_penalty += overlap * 1000 * (1 + overlap * 0.1)
            
            return containment_penalty, overlap_penalty, containment_penalty + overlap_penalty
        
        def find_valid_radius(self, individual: np.ndarray, index: int, position_only: bool = False) -> float:
            """Find the maximum valid radius for a circle at given position"""
            x, y = individual[index, 0], individual[index, 1]
            # Calculate boundary constraints
            max_radius_boundary = min(x, 1-x, y, 1-y)
            
            if position_only:
                return max_radius_boundary
            
            # Include overlap constraints with other circles
            max_radius = max_radius_boundary
            for i in range(len(individual)):
                if i != index:
                    r = individual[i, 2]
                    # Distance to center of existing circle
                    dist = np.sqrt((individual[i, 0] - x)**2 + (individual[i, 1] - y)**2)
                    # Maximum radius to avoid overlap
                    max_radius_to_this_circle = dist - r
                    max_radius = min(max_radius, max_radius_to_this_circle)
            
            return max(0.001, max_radius)
        
        def update_radius(self, individual: np.ndarray, index: int) -> None:
            """Update radius of a circle to maximum valid value"""
            new_radius = self.find_valid_radius(individual, index)
            individual[index, 2] = new_radius
        
        def optimize_single_circle(self, individual: np.ndarray, index: int, max_iter: int = 50) -> bool:
            """Optimize position and radius of one circle"""
            old_x, old_y, old_r = individual[index, 0], individual[index, 1], individual[index, 2]
            
            # Try to increase radius first
            new_radius = self.find_valid_radius(individual, index)
            if new_radius > old_r:
                individual[index, 2] = new_radius
            
            # Then try to reposition for better results
            best_x, best_y, best_r = individual[index, 0], individual[index, 1], individual[index, 2]
            best_score = np.sum(individual[:, 2]) - self.calculate_constraints_violation(individual)[2]
            
            # Try small perturbations
            for _ in range(max_iter):
                # Random small perturbation
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                
                new_x = np.clip(old_x + dx, 0.01, 0.99)
                new_y = np.clip(old_y + dy, 0.01, 0.99)
                
                # Create temporary individual to test
                temp_individual = individual.copy()
                temp_individual[index, 0] = new_x
                temp_individual[index, 1] = new_y
                
                # Update its radius
                temp_individual[index, 2] = self.find_valid_radius(temp_individual, index)
                
                # Calculate new score
                new_score = np.sum(temp_individual[:, 2]) - self.calculate_constraints_violation(temp_individual)[2]
                
                # Accept better solutions or accept with probability based on temperature
                if new_score > best_score:
                    best_x, best_y, best_r = new_x, new_y, temp_individual[index, 2]
                    best_score = new_score
                elif self.temperature > self.min_temperature:
                    # Accept worse solutions with some probability
                    delta = new_score - best_score
                    if np.random.random() < np.exp(delta / self.temperature):
                        best_x, best_y, best_r = new_x, new_y, temp_individual[index, 2]
                        best_score = new_score
            
            individual[index, 0] = best_x
            individual[index, 1] = best_y
            individual[index, 2] = best_r
            return True
        
        def propagate_constraints(self, individual: np.ndarray, max_iter: int = 1000) -> bool:
            """Iteratively improve solution by propagating constraints"""
            improvement_count = 0
            
            for iteration in range(max_iter):
                old_sum = np.sum(individual[:, 2])
                
                # Optimization passes
                for i in range(self.n_circles):
                    # Update radius first
                    original_radius = individual[i, 2]
                    self.update_radius(individual, i)
                    
                    # Then optimize position/radius together
                    self.optimize_single_circle(individual, i, max_iter=20)
                    
                    if individual[i, 2] > original_radius:
                        improvement_count += 1
                
                # Early stopping if no significant improvement
                if abs(np.sum(individual[:, 2]) - old_sum) < 1e-6:
                    break
                    
            return improvement_count > 0
        
        def local_optimization(self, individual: np.ndarray) -> bool:
            """Perform local optimization to improve solution quality"""
            improved = False
            original_total = np.sum(individual[:, 2])
            
            # Try to improve each circle
            for i in range(self.n_circles):
                old_r = individual[i, 2]
                # Try to increase radius
                self.update_radius(individual, i)
                
                # If radius increased, see if we can also adjust position
                if individual[i, 2] > old_r:
                    improved = True
                    
            return improved
        
        def solve(self) -> np.ndarray:
            """Main solving routine using constraint propagation and optimization"""
            # Initialize
            individual = self.initialize_positions()
            
            # Initial constraint checking
            best_individual = individual.copy()
            best_score = np.sum(individual[:, 2]) - self.calculate_constraints_violation(individual)[2]
            
            # Multi-stage optimization
            for iteration in range(self.max_iterations):
                # Stage 1: Constraint propagation and local optimization
                self.propagate_constraints(individual, max_iter=100)
                self.local_optimization(individual)
                
                # Stage 2: Simulated Annealing components
                if self.temperature > self.min_temperature:
                    # Random perturbation and acceptance
                    test_individual = individual.copy()
                    
                    # Perturb a few circles
                    for _ in range(3):  # Perturb 3 circles
                        idx = np.random.randint(0, self.n_circles)
                        # Small position change
                        test_individual[idx, 0] += np.random.normal(0, 0.005)
                        test_individual[idx, 1] += np.random.normal(0, 0.005)
                        
                        # Clip to valid range
                        test_individual[idx, 0] = np.clip(test_individual[idx, 0], 0.01, 0.99)
                        test_individual[idx, 1] = np.clip(test_individual[idx, 1], 0.01, 0.99)
                        
                        # Update radius
                        self.update_radius(test_individual, idx)
                    
                    # Calculate scores
                    current_score = np.sum(individual[:, 2]) - self.calculate_constraints_violation(individual)[2]
                    test_score = np.sum(test_individual[:, 2]) - self.calculate_constraints_violation(test_individual)[2]
                    
                    # Accept with probability based on temperature
                    if test_score > current_score or (
                        self.temperature > self.min_temperature and 
                        np.random.random() < np.exp((test_score - current_score) / self.temperature)
                    ):
                        individual = test_individual.copy()
                        
                    # Cool down temperature
                    self.temperature *= self.cooling_rate
                
                # Update best solution
                current_total = np.sum(individual[:, 2])
                current_penalty = self.calculate_constraints_violation(individual)[2]
                current_score = current_total - current_penalty
                
                if current_score > best_score:
                    best_score = current_score
                    best_individual = individual.copy()
                
                # Check for convergence
                if iteration % 100 == 0:
                    pass  # Could add logging here
            
            return best_individual
    
    # Create and run optimizer
    optimizer = ConstraintPropagationOptimizer(n_circles=26)
    result = optimizer.solve()
    
    # Final cleanup to ensure constraints
    violations = optimizer.calculate_constraints_violation(result)
    if violations[2] > 0:
        # Do one final constraint satisfaction pass
        for i in range(optimizer.n_circles):
            # Ensure valid radius
            max_radius = optimizer.find_valid_radius(result, i, position_only=True)
            if result[i, 2] > max_radius:
                result[i, 2] = max_radius
                
            # Ensure within bounds
            result[i, 0] = np.clip(result[i, 0], result[i, 2], 1 - result[i, 2])
            result[i, 1] = np.clip(result[i, 1], result[i, 2], 1 - result[i, 2])
    
    return result

# EVOLVE-BLOCK-END