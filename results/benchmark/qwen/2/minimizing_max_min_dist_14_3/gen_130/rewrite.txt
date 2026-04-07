# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
import time
from typing import Tuple, List, Optional
import math

@jit(nopython=True)
def compute_min_max_ratio_numba(points):
    """Optimized distance computation using numba"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0

    for i in range(n):
        for j in range(i+1, n):
            # Compute squared distance to avoid sqrt computation
            dist_sq = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2 + (points[i,2]-points[j,2])**2
            dist = np.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return min_dist, max_dist

class PointGenerator:
    """Handles generation of initial point configurations"""
    
    @staticmethod
    def fibonacci_sphere(n: int) -> np.ndarray:
        """Generate n points distributed evenly on a unit sphere using Fibonacci method"""
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    @staticmethod
    def random_on_sphere(n: int) -> np.ndarray:
        """Generate n random points on unit sphere"""
        points = np.random.randn(n, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    @staticmethod
    def perturbed_fibonacci(n: int, noise_level: float = 0.03) -> np.ndarray:
        """Generate Fibonacci points with added noise"""
        points = PointGenerator.fibonacci_sphere(n)
        noise = np.random.normal(0, noise_level, points.shape)
        points = points + noise
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

class Evaluator:
    """Handles evaluation of point configurations"""
    
    @staticmethod
    def compute_min_max_ratio(points: np.ndarray) -> Tuple[float, float, float]:
        """Compute the minimum and maximum distances between all pairs of points, and return their ratio."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        # Use numba-optimized version
        min_distance, max_distance = compute_min_max_ratio_numba(points)

        # Avoid division by zero
        if max_distance == 0:
            ratio = 0.0
        else:
            ratio = min_distance / max_distance

        return min_distance, max_distance, ratio

class Optimizer:
    """Main optimization engine using simulated annealing"""
    
    def __init__(self, max_iterations: int = 100000, initial_temp: float = 1.0, 
                 min_temp: float = 0.0001, cooling_rate: float = 0.9995):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.ratio_history = []
        
    def project_to_unit_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points to the unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Handle case where norm might be zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def spherical_perturb(self, points: np.ndarray, target_point_idx: int, 
                         temperature: float) -> np.ndarray:
        """Apply perturbation on the tangent plane of the unit sphere at target point"""
        # Create a copy of the points
        new_points = points.copy()
        
        # Get the target point
        target_point = points[target_point_idx]
        
        # Generate random perturbation in tangent plane
        perturbation = np.random.normal(0, 0.01 * temperature, 3)
        
        # Project the perturbation onto the tangent plane (orthogonal to the normal)
        normal = target_point
        proj = np.dot(perturbation, normal)
        tangent_perturbation = perturbation - proj * normal
        
        # Apply the perturbation
        new_points[target_point_idx] = target_point + tangent_perturbation
        
        # Project back to unit sphere
        new_points = self.project_to_unit_sphere(new_points)
        
        return new_points
    
    def adaptive_perturbation_strategy(self, points: np.ndarray, current_ratio: float, 
                                     temperature: float) -> np.ndarray:
        """Apply adaptive perturbation based on current configuration analysis"""
        # Analyze current distribution to decide which point to perturb
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Get minimum and maximum distances for analysis
        min_distances = np.min(distances, axis=1)
        max_distances = np.max(distances, axis=1)
        
        # Calculate average distance per point for reference
        avg_distances = np.mean(distances, axis=1)
        
        # Prefer perturbing points that:
        # 1. Are among the closest points (to possibly increase minimum)
        # 2. Are among the farthest points (to possibly decrease maximum)  
        # 3. Or are in intermediate positions
        
        # Score points based on their potential impact
        scores = np.zeros(len(points))
        for i in range(len(points)):
            # Weight by how close they are to min vs max distances
            min_dist = min_distances[i]
            max_dist = max_distances[i]
            avg_dist = avg_distances[i]
            
            # Score based on being too close (helps increase min) or too far (helps decrease max)
            if min_dist < avg_dist * 0.5:  # Very close - prioritize increasing their distance
                scores[i] = -min_dist
            elif max_dist > avg_dist * 2.0:  # Very far - prioritize decreasing their distance
                scores[i] = max_dist
            else:  # Medium distance - less critical
                scores[i] = 0
        
        # Choose point to perturb based on scores (higher score means more important)
        if np.sum(np.abs(scores)) > 0:
            # Use weighted probability based on scores
            probs = np.abs(scores)
            probs = probs / np.sum(probs)
            target_idx = np.random.choice(len(points), p=probs)
        else:
            # Fallback to random selection
            target_idx = np.random.randint(len(points))
        
        return self.spherical_perturb(points, target_idx, temperature)
    
    def adaptive_cooling(self, iteration: int) -> float:
        """Adaptive cooling schedule that adjusts based on convergence"""
        # Base cooling rate
        base_cooling = self.cooling_rate

        # Check recent convergence
        if len(self.ratio_history) > 10:
            recent_improvement = self.ratio_history[-1] - self.ratio_history[-10]
            if recent_improvement < 1e-8:
                # Slow improvement, cool faster
                return base_cooling * 1.05
            elif recent_improvement > 1e-6:
                # Fast improvement, cool slower
                return base_cooling * 0.95

        return base_cooling
    
    def optimize_single_run(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Perform optimization from a single starting configuration"""
        # Initialize state
        current_points = initial_points.copy()
        current_min_dist, current_max_dist, current_ratio = Evaluator.compute_min_max_ratio(initial_points)
        
        # Track best solution
        best_points = current_points.copy()
        best_min_dist = current_min_dist
        best_max_dist = current_max_dist
        best_ratio = current_ratio
        
        # Simulated Annealing parameters
        temp = self.initial_temp
        last_improvement_iter = 0
        iteration = 0
        
        # Different temperature schedules for different phases
        temp_schedule = [
            {"temp": 1.0, "duration": 50000},   # High temperature for exploration
            {"temp": 0.5, "duration": 50000},   # Medium temperature for refinement  
            {"temp": 0.1, "duration": 50000}    # Low temperature for fine-tuning
        ]
        
        current_phase = 0
        phase_iterations = 0
        
        while iteration < self.max_iterations:
            # Check if we need to advance to next temperature phase
            if phase_iterations >= temp_schedule[current_phase]["duration"]:
                current_phase = min(current_phase + 1, len(temp_schedule) - 1)
                temp = temp_schedule[current_phase]["temp"]
                phase_iterations = 0
            
            # Perturb the current solution
            new_points = self.adaptive_perturbation_strategy(current_points, current_ratio, temp)

            # Compute new ratio
            new_min_dist, new_max_dist, new_ratio = Evaluator.compute_min_max_ratio(new_points)

            # Accept or reject the new solution using Metropolis criterion
            if new_ratio > current_ratio:
                # Always accept better solutions
                current_points = new_points
                current_ratio = new_ratio
                current_min_dist = new_min_dist
                current_max_dist = new_max_dist

                # Update best solution if this is better
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
                    best_min_dist = new_min_dist
                    best_max_dist = new_max_dist
                    last_improvement_iter = iteration
                    self.ratio_history.append(new_ratio)
            else:
                # Accept worse solutions with probability based on temperature
                if temp > 0:  # Avoid division by zero
                    acceptance_prob = np.exp((new_ratio - current_ratio) / temp)
                    if np.random.rand() < acceptance_prob:
                        current_points = new_points
                        current_ratio = new_ratio
                        current_min_dist = new_min_dist
                        current_max_dist = new_max_dist
                        self.ratio_history.append(new_ratio)

            # Apply adaptive cooling
            temp = max(temp * self.adaptive_cooling(iteration), self.min_temp)
            
            # Increment counters
            iteration += 1
            phase_iterations += 1

            # Early stopping if no improvement in a long time
            if iteration - last_improvement_iter > 30000:
                break
                
        return best_points, best_min_dist, best_max_dist, best_ratio

class EvolutionManager:
    """Manages the overall evolutionary process and multiple runs"""
    
    def __init__(self):
        self.point_generator = PointGenerator()
        self.evaluator = Evaluator()
        self.optimizer = Optimizer()
        self.initialization_strategies = [
            lambda: self.point_generator.fibonacci_sphere(14),
            lambda: self.point_generator.random_on_sphere(14),
            lambda: self.point_generator.perturbed_fibonacci(14),
            lambda: self.point_generator.perturbed_fibonacci(14, 0.05),
            lambda: self.point_generator.random_on_sphere(14)
        ]

    def run_multiple_starts(self) -> np.ndarray:
        """Run optimization from multiple starting configurations"""
        best_points = None
        best_ratio = -np.inf
        best_min_dist = 0
        best_max_dist = 0

        # Run optimization from each initialization strategy
        for strategy_idx, strategy_func in enumerate(self.initialization_strategies):
            # Set seed for reproducibility
            np.random.seed(strategy_idx)
            
            try:
                # Generate initial points
                points = strategy_func()
                
                # Run optimization
                optimized_points, min_dist, max_dist, ratio = self.optimizer.optimize_single_run(points)
                
                # Update global best if this run was better
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    best_min_dist = min_dist
                    best_max_dist = max_dist
                    
            except Exception as e:
                print(f"Error in strategy {strategy_idx}: {e}")
                continue

        # Ensure the result is properly normalized (should already be done, but extra safety)
        if best_points is not None:
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            best_points = best_points / norms
        else:
            # Fallback to Fibonacci if nothing worked
            best_points = self.point_generator.fibonacci_sphere(14)
            
        return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create evolution manager and run optimization
    manager = EvolutionManager()
    return manager.run_multiple_starts()

# EVOLVE-BLOCK-END