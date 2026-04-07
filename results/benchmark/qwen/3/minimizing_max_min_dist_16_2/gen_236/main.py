# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import warnings

class HybridEvolutionaryGradientOptimizer:
    """Enhanced hybrid optimizer combining evolutionary and gradient-based approaches."""

    def __init__(self, n_points=16, dimensions=2, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _compute_ratio(self, points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0

        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max == 0:
            return 0

        return d_min / d_max

    def _compute_penalty(self, points, penalty_weight=1000):
        """Compute penalty for points outside bounds."""
        penalty = 0
        # Check if any point is outside [0,1] bounds
        out_of_bounds = np.logical_or(points < 0, points > 1)
        if np.any(out_of_bounds):
            # Apply penalty based on how far out of bounds they are
            penalty = penalty_weight * np.sum(np.abs(points[out_of_bounds]))
        return penalty

    def _generate_fibonacci_spiral_points(self):
        """Generate points using Fibonacci spiral pattern."""
        points = np.zeros((self.n_points, self.dimensions))
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.n_points):
            # Spiral positioning
            z = 1 - (i / (self.n_points - 1)) * 2  # z coordinate from -1 to 1
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)
            
            # Project to 2D square
            x = (radius * np.cos(phi) + 1) / 2
            y = (radius * np.sin(phi) + 1) / 2
            
            points[i] = [x, y]
        
        # Add small perturbations to break symmetries
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_hexagonal_grid(self):
        """Generate points in a more sophisticated hexagonal grid pattern."""
        points = []
        
        # Create a 4x4 triangular lattice with better spacing control
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 0.5)
        spacing_y = spacing_x * np.sqrt(3) / 2

        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    x = j * spacing_x + (i % 2) * spacing_x / 2
                    y = i * spacing_y
                    
                    # Apply prime-based systematic perturbations to break symmetries
                    prime_i = (i * 7) % 11
                    prime_j = (j * 13) % 17
                    asymmetry_factor = 0.01
                    
                    x_pert = asymmetry_factor * np.sin(prime_i * 0.3) * np.cos(prime_j * 0.7)
                    y_pert = asymmetry_factor * np.cos(prime_i * 0.4) * np.sin(prime_j * 0.6)
                    
                    points.append([x + x_pert, y + y_pert])

        points = np.array(points[:self.n_points])
        
        # Normalize and add final noise
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0 and y_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
                
                # Scale to fit nicely in [0,1] square
                points[:, 0] *= 0.95
                points[:, 1] *= 0.95
                points[:, 0] += 0.025
                points[:, 1] += 0.025

        points += np.random.normal(0, 0.002, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_random_points(self):
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)

    def _evolutionary_search(self):
        """Simplified but effective evolutionary optimization."""
        # Generate population using diverse strategies
        population = []
        
        # Add Fibonacci spiral points
        population.append(self._generate_fibonacci_spiral_points())
        
        # Add hexagonal grid points  
        population.append(self._generate_hexagonal_grid())
        
        # Add random points
        population.append(self._generate_random_points())
        
        # Add variations of the hexagonal grid
        hex_variant = self._generate_hexagonal_grid()
        hex_variant += np.random.normal(0, 0.01, hex_variant.shape)
        hex_variant = np.clip(hex_variant, 0, 1)
        population.append(hex_variant)
        
        best_ratio = -np.inf
        best_individual = None

        # Simple evolutionary process with fewer generations
        generations = 30
        elite_size = 2
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                ratio = self._compute_ratio(individual)
                fitness = ratio  # Simple ratio-based fitness
                fitness_scores.append(fitness)
                if fitness > best_ratio:
                    best_ratio = fitness
                    best_individual = individual.copy()
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            
            # Create new population with elitism
            new_population = population[:elite_size].copy()
            
            # Generate offspring through crossover
            while len(new_population) < len(population):
                parent1, parent2 = population[0], population[1]
                # Uniform crossover
                mask = np.random.rand(*parent1.shape) > 0.5
                child = np.where(mask, parent1, parent2).copy()
                
                # Mutation
                if np.random.rand() < 0.3:
                    mutation_strength = 0.015
                    for i in range(len(child)):
                        child[i] += np.random.normal(0, mutation_strength, 2)
                
                child = np.clip(child, 0, 1)
                new_population.append(child)
            
            population = new_population[:len(population)]

        return best_individual if best_individual is not None else self._generate_fibonacci_spiral_points()

    def _neighborhood_move(self, points, neighbor_size=2):
        """Apply neighborhood-based move to improve exploration."""
        new_points = points.copy()
        # Select random subset of points to move together
        indices = np.random.choice(len(points), min(neighbor_size, len(points)), replace=False)
        
        # Move selected points by small amounts
        for idx in indices:
            new_points[idx] += np.random.normal(0, 0.01, 2)
        
        # Clamp to bounds
        new_points = np.clip(new_points, 0, 1)
        return new_points

    def _adaptive_local_optimization(self, points, max_iter=100):
        """Simple adaptive local optimization using gradient estimation."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self._compute_ratio(points)  # Minimize negative ratio (maximize ratio)
        
        def constraint_function(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            violations = np.concatenate([
                np.minimum(points[:, 0], 0),
                np.minimum(points[:, 1], 0),
                np.maximum(points[:, 0] - 1, 0),
                np.maximum(points[:, 1] - 1, 0)
            ])
            return violations
        
        # Flatten for optimization
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(len(x0))]
        constraints = {'type': 'ineq', 'fun': constraint_function}
        
        best_points = points.copy()
        best_ratio = self._compute_ratio(best_points)
        
        # Try SLSQP optimization
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-6, 'eps': 1e-4}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, self.dimensions)
                refined_points = np.clip(refined_points, 0, 1)
                current_ratio = self._compute_ratio(refined_points)
                if current_ratio > best_ratio:
                    best_points = refined_points
                    best_ratio = current_ratio
        except:
            pass
        
        return best_points

    def _simulated_annealing_refinement(self, points, max_iter=3000):
        """Refine solution using adaptive simulated annealing."""
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Adaptive cooling schedule
        temperature = 0.1
        cooling_rate = 0.9995
        min_temperature = 1e-6

        # Main optimization loop
        for iteration in range(max_iter):
            # Generate neighbor solution
            new_points = self._neighborhood_move(current_points, neighbor_size=3)

            # Calculate acceptance probability
            new_ratio = self._compute_ratio(new_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temperature):
                current_points = new_points.copy()
                current_ratio = new_ratio

                # Update best if improved
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()

            # Cool down temperature
            temperature *= cooling_rate
            
            # Early stopping if temperature gets too low
            if temperature < min_temperature:
                break

        return best_points

    def _smart_gradient_refinement(self, points, max_iter=300):
        """Refine solution using smart gradient-based approach."""
        final_points = points.copy()
        best_ratio = self._compute_ratio(final_points)

        for iteration in range(max_iter):
            # Estimate gradient by finite differences
            gradient = np.zeros_like(final_points)
            eps = 1e-4

            for i in range(len(final_points)):
                for j in range(self.dimensions):
                    # Perturb point coordinate
                    points_plus = final_points.copy()
                    points_plus[i, j] += eps
                    points_plus = np.clip(points_plus, 0, 1)

                    points_minus = final_points.copy()
                    points_minus[i, j] -= eps
                    points_minus = np.clip(points_minus, 0, 1)

                    ratio_plus = self._compute_ratio(points_plus)
                    ratio_minus = self._compute_ratio(points_minus)

                    gradient[i, j] = (ratio_plus - ratio_minus) / (2 * eps)

            # Use adaptive step size
            step_size = 0.01 * (1.0 - iteration / max_iter)  # Decrease over time

            # Update points
            final_points = final_points + step_size * gradient

            # Ensure bounds
            final_points = np.clip(final_points, 0, 1)

            # Check for convergence
            if np.all(np.abs(gradient) < 1e-6):
                break

            # Update best ratio
            current_ratio = self._compute_ratio(final_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio

        return final_points

    def optimize(self):
        """Main optimization routine with simplified multi-start approach."""
        best_solution = None
        best_ratio = -np.inf

        # Run multiple optimization attempts from different starting points
        start_points_strategies = [
            self._generate_fibonacci_spiral_points,
            self._generate_hexagonal_grid,
            self._generate_random_points
        ]

        for i, strategy in enumerate(start_points_strategies):
            try:
                # Generate initial points
                initial_points = strategy()

                # Stage 1: Evolutionary search
                evolved_points = self._evolutionary_search()

                # Stage 2: Local optimization
                optimized_points = self._adaptive_local_optimization(evolved_points)

                # Stage 3: Simulated annealing refinement
                sa_points = self._simulated_annealing_refinement(optimized_points)

                # Stage 4: Gradient refinement
                final_points = self._smart_gradient_refinement(sa_points)

                # Evaluate final result
                final_ratio = self._compute_ratio(final_points)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_solution = final_points.copy()

            except Exception as e:
                warnings.warn(f"Error in optimization attempt {i+1}: {str(e)}")
                continue

        # If no valid results, return the last attempted configuration
        if best_solution is None:
            return self._generate_fibonacci_spiral_points()

        return best_solution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = HybridEvolutionaryGradientOptimizer(n_points=16, dimensions=2, seed=42)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END