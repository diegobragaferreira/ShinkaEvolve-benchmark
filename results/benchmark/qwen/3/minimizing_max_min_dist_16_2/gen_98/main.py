# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import distance
import warnings
import time

class SphericalCodeEvolutionOptimizer:
    """Optimizes point distribution using spherical code theory with evolutionary refinement."""
    
    def __init__(self, n_points=16, dimensions=2, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _project_sphere_to_square(self, points_3d):
        """Project points from 3D sphere to 2D square using stereographic projection."""
        # Stereographic projection from north pole (0,0,1) to plane z=0
        points_2d = np.zeros((len(points_3d), 2))
        for i, point in enumerate(points_3d):
            # Normalize to unit sphere
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm
            # Stereographic projection from north pole
            x, y, z = point
            if z != 1:  # Avoid singularity at north pole
                scale = 1 / (1 - z)
                points_2d[i] = [x * scale, y * scale]
            else:  # North pole maps to infinity, clip to large value
                points_2d[i] = [1000, 1000]
        # Normalize to [0,1] square
        points_2d = np.clip(points_2d, -100, 100)  # Prevent extreme values
        # Scale to unit square [0,1]
        min_vals = np.min(points_2d, axis=0)
        max_vals = np.max(points_2d, axis=0)
        if np.allclose(min_vals, max_vals):
            return np.ones_like(points_2d) * 0.5
        ranges = max_vals - min_vals
        points_2d = (points_2d - min_vals) / ranges
        points_2d = np.clip(points_2d, 0, 1)
        return points_2d

    def _generate_spherical_code_points(self):
        """Generate points using spherical code approximation (Fibonacci-like on sphere)."""
        points = []
        # Generate points using Fibonacci spiral on sphere
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(self.n_points):
            # Parameterization of sphere using Fibonacci sequence
            phi = np.arccos(1 - 2*(i/(self.n_points - 1)))  # Polar angle
            theta = (i * golden_ratio) % (2 * np.pi)  # Azimuthal angle
            
            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            points.append([x, y, z])
        
        return np.array(points)

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

    def _compute_energy(self, points):
        """Compute electrostatic energy (inverse distance sum) for quality assessment."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        energy = np.sum(1.0 / distances)
        return energy

    def _generate_diverse_initializations(self):
        """Generate multiple diverse initial configurations."""
        initial_configs = []
        
        # 1. Spherical code points (primary)
        spherical_points = self._generate_spherical_code_points()
        square_points = self._project_sphere_to_square(spherical_points)
        initial_configs.append(square_points)
        
        # 2. Perturbed spherical code
        perturbed_points = spherical_points.copy()
        perturbed_points += np.random.normal(0, 0.02, perturbed_points.shape)
        # Normalize to unit sphere
        for i in range(len(perturbed_points)):
            norm = np.linalg.norm(perturbed_points[i])
            if norm > 0:
                perturbed_points[i] = perturbed_points[i] / norm
        perturbed_square = self._project_sphere_to_square(perturbed_points)
        initial_configs.append(perturbed_square)
        
        # 3. Random points with sphere projection
        random_points = np.random.rand(self.n_points, 3)
        # Project to unit sphere
        for i in range(len(random_points)):
            norm = np.linalg.norm(random_points[i])
            if norm > 0:
                random_points[i] = random_points[i] / norm
        random_square = self._project_sphere_to_square(random_points)
        initial_configs.append(random_square)
        
        # 4. Hexagonal grid in square projection
        hex_points = []
        rows, cols = 4, 4
        spacing_x, spacing_y = 1.0/cols, 1.0/rows
        for i in range(rows):
            for j in range(cols):
                if len(hex_points) < self.n_points:
                    x = (j + (i % 2) * 0.5) * spacing_x
                    y = i * spacing_y
                    hex_points.append([x, y])
        # Convert to 3D by setting z to 0
        hex_points_3d = np.array(hex_points[:self.n_points])
        # Add small z component to make it non-degenerate
        hex_points_3d = np.column_stack([hex_points_3d, np.random.normal(0, 0.001, self.n_points)])
        # Normalize
        for i in range(len(hex_points_3d)):
            norm = np.linalg.norm(hex_points_3d[i])
            if norm > 0:
                hex_points_3d[i] = hex_points_3d[i] / norm
        hex_square = self._project_sphere_to_square(hex_points_3d)
        initial_configs.append(hex_square)
        
        return initial_configs

    def _evolutionary_selection(self, population):
        """Select best individuals using tournament selection."""
        # Evaluate each individual
        fitness_scores = []
        for individual in population:
            # Combine ratio and energy (higher ratio, lower energy is better)
            ratio = self._compute_ratio(individual)
            energy = self._compute_energy(individual)
            # Energy should be minimized (lower is better), ratio maximized
            fitness = ratio * (1.0 / (1e-8 + energy))
            fitness_scores.append(fitness)
        
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            # Select 3 random individuals for tournament
            tournament_indices = np.random.choice(len(population), 3, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected

    def _crossover_uniform(self, parent1, parent2):
        """Uniform crossover between two parents."""
        mask = np.random.rand(*parent1.shape) > 0.5
        child = np.where(mask, parent1, parent2).copy()
        return child

    def _mutate(self, individual, mutation_rate=0.3, mutation_strength=0.02):
        """Apply mutation to an individual."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.rand() < mutation_rate:
                mutated[i] += np.random.normal(0, mutation_strength, 2)
        return mutated

    def _evolutionary_search(self, max_generations=50):
        """Perform evolutionary optimization across multiple initial configurations."""
        # Generate diverse initial population
        population = self._generate_diverse_initializations()
        
        # Add some random diversity to initial population
        for i in range(len(population)):
            population[i] += np.random.normal(0, 0.01, population[i].shape)
            population[i] = np.clip(population[i], 0, 1)
        
        best_fitness = -np.inf
        best_individual = None
        
        for generation in range(max_generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                ratio = self._compute_ratio(individual)
                energy = self._compute_energy(individual)
                fitness = ratio * (1.0 / (1e-8 + energy))
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selection
            selected = self._evolutionary_selection(population)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            for i in range(min(3, len(selected))):
                new_population.append(selected[sorted_indices[i]].copy())
            
            # Generate offspring
            while len(new_population) < len(selected):
                # Select two parents
                parent1 = selected[np.random.randint(len(selected))]
                parent2 = selected[np.random.randint(len(selected))]
                
                # Crossover
                child = self._crossover_uniform(parent1, parent2)
                
                # Mutation
                child = self._mutate(child, 0.3, 0.015)
                
                # Ensure bounds
                child = np.clip(child, 0, 1)
                new_population.append(child)
            
            population = new_population[:len(selected)]
        
        return best_individual if best_individual is not None else population[0]

    def _local_optimization(self, points, max_iter=100):
        """Use L-BFGS-B for local optimization refining the solution."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self._compute_ratio(points)  # Minimize negative ratio (maximize ratio)
        
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(len(x0))]
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, self.dimensions)
                refined_points = np.clip(refined_points, 0, 1)
                return refined_points
        except:
            pass
        
        return points

    def _gradient_refinement(self, points, max_iter=100):
        """Refine solution using gradient estimation."""
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
            
            # Update with adaptive step size
            step_size = 0.01 * (1.0 - iteration / max_iter)  # Decrease over time
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
        """Main optimization routine."""
        # Stage 1: Evolutionary search with spherical code foundation
        print("Starting evolutionary search with spherical code...")
        evolved_points = self._evolutionary_search(max_generations=30)
        
        # Stage 2: Local optimization
        print("Starting local optimization...")
        local_optimized = self._local_optimization(evolved_points, max_iter=200)
        
        # Stage 3: Gradient refinement
        print("Starting gradient refinement...")
        refined_points = self._gradient_refinement(local_optimized, max_iter=100)
        
        # Final evaluation and selection
        final_ratio = self._compute_ratio(refined_points)
        evolved_ratio = self._compute_ratio(evolved_points)
        local_ratio = self._compute_ratio(local_optimized)
        
        if final_ratio >= evolved_ratio and final_ratio >= local_ratio:
            return refined_points
        elif evolved_ratio >= local_ratio:
            return evolved_points
        else:
            return local_optimized

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = SphericalCodeEvolutionOptimizer(n_points=16, dimensions=2, seed=42)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END