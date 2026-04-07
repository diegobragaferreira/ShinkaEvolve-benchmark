# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
import math

class GeometricEvolutionOptimizer:
    def __init__(self, num_points=16, dimensions=2):
        self.num_points = num_points
        self.dimensions = dimensions
        self.benchmark_threshold = 1 / np.sqrt(12.889266112)
        self.best_ratio = -np.inf
        self.best_points = None
        
    def _compute_min_max_ratio(self, points):
        """Efficiently compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def _objective_function(self, x):
        """Objective function to maximize min/max distance ratio."""
        points = x.reshape(self.num_points, self.dimensions)
        ratio = self._compute_min_max_ratio(points)
        return -ratio  # Negative for minimization
    
    def _create_hexagonal_pattern(self):
        """Generate points in hexagonal packing pattern."""
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                # Offset every other row
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) / 3.0
                y = i / 3.0
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                points.append([x, y])
                
        return np.array(points)
    
    def _create_fibonacci_pattern(self):
        """Generate points using Fibonacci spiral approach."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(self.num_points):
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
            
        return np.array(points)
    
    def _create_regular_grid(self):
        """Generate regular 4x4 grid pattern."""
        points = []
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) / 4.0
                y = (i + 0.5) / 4.0
                points.append([x, y])
        return np.array(points)
    
    def _create_perturbed_grid(self, base_points, perturbation_scale=0.05):
        """Apply adaptive perturbations to base points."""
        np.random.seed(42)
        perturbed = base_points + np.random.normal(0, perturbation_scale, base_points.shape)
        perturbed = np.clip(perturbed, 0.05, 0.95)
        return perturbed
    
    def _create_symmetric_breaking_constraints(self, points):
        """Apply constraints to break symmetry and maintain unique solutions."""
        # Enforce lexicographic ordering of points to prevent symmetric solutions
        sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
        return points[sorted_indices]
    
    def _adaptive_mutation(self, points, current_ratio):
        """Apply mutation with adaptive strength based on solution quality."""
        # Determine mutation strength based on current solution quality
        if current_ratio < 0.1:
            mutation_strength = 0.08
        elif current_ratio < 0.2:
            mutation_strength = 0.04
        else:
            mutation_strength = 0.02
            
        # Apply mutation
        np.random.seed(int(time.time()) % 1000000)
        mutated = points + np.random.normal(0, mutation_strength, points.shape)
        mutated = np.clip(mutated, 0.05, 0.95)
        return mutated
    
    def _tournament_selection(self, candidates, fitness_scores):
        """Select best candidates using tournament selection."""
        # Select top 3 candidates based on fitness
        sorted_indices = np.argsort(fitness_scores)[:min(3, len(candidates))]
        return [candidates[i] for i in sorted_indices]
    
    def _hybrid_optimization(self, initial_points):
        """Apply hybrid optimization combining global and local strategies."""
        # Stage 1: Global search with L-BFGS-B for coarse optimization
        bounds = [(0.05, 0.95) for _ in range(self.num_points * self.dimensions)]
        
        try:
            result1 = minimize(
                self._objective_function,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result1.success:
                stage1_points = result1.x.reshape(self.num_points, self.dimensions)
            else:
                stage1_points = initial_points.copy()
        except Exception:
            stage1_points = initial_points.copy()
            
        # Stage 2: Local refinement with SLSQP
        try:
            result2 = minimize(
                self._objective_function,
                stage1_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result2.success:
                optimized_points = result2.x.reshape(self.num_points, self.dimensions)
            else:
                optimized_points = stage1_points.copy()
        except Exception:
            optimized_points = stage1_points.copy()
            
        return optimized_points
    
    def _evolutionary_restart_strategy(self):
        """Implement evolutionary restart strategy with multiple patterns."""
        # Generate diverse initial configurations
        initial_configs = []
        
        # Create different patterns
        patterns = [
            self._create_hexagonal_pattern(),
            self._create_fibonacci_pattern(),
            self._create_regular_grid()
        ]
        
        # Perturb each pattern with different strengths
        for i, base_pattern in enumerate(patterns):
            np.random.seed(i * 100)
            perturbed_1 = self._create_perturbed_grid(base_pattern, 0.03)
            perturbed_2 = self._create_perturbed_grid(base_pattern, 0.06)
            initial_configs.extend([base_pattern, perturbed_1, perturbed_2])
        
        # Add some random configurations for diversity
        for i in range(3):
            np.random.seed(1000 + i)
            random_config = np.random.uniform(0.05, 0.95, (self.num_points, self.dimensions))
            initial_configs.append(random_config)
        
        # Evaluate fitness of all initial configurations
        fitness_scores = []
        for config in initial_configs:
            ratio = self._compute_min_max_ratio(config)
            fitness_scores.append(ratio)
        
        # Select best configurations using tournament selection
        selected_configs = self._tournament_selection(initial_configs, fitness_scores)
        
        return selected_configs
    
    def optimize(self):
        """Main optimization loop with evolutionary restarts."""
        # Get initial configurations using evolutionary approach
        initial_configs = self._evolutionary_restart_strategy()
        
        # Try each configuration with hybrid optimization
        for i, initial_config in enumerate(initial_configs):
            # Apply symmetry breaking constraints
            constrained_config = self._create_symmetric_breaking_constraints(initial_config)
            
            # Apply hybrid optimization
            optimized_points = self._hybrid_optimization(constrained_config)
            
            # Evaluate final ratio
            ratio = self._compute_min_max_ratio(optimized_points)
            
            # Keep track of best solution
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_points = optimized_points.copy()
                
                # Early termination if benchmark beaten
                if ratio >= self.benchmark_threshold:
                    break
        
        # If no good solution found, use fallback method
        if self.best_points is None:
            # Start with hexagonal pattern and optimize
            fallback_points = self._create_hexagonal_pattern()
            self.best_points = self._hybrid_optimization(fallback_points)
            self.best_ratio = self._compute_min_max_ratio(self.best_points)
            
        return self.best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    
    # Initialize optimizer
    optimizer = GeometricEvolutionOptimizer(16, 2)
    
    # Perform optimization
    points = optimizer.optimize()
    
    return points

# EVOLVE-BLOCK-END