# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings

class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio with hybrid approach."""
    
    def __init__(self, n_points=16, dimensions=2, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)
    
    def _compute_distances(self, points):
        """Compute pairwise distances between points."""
        if len(points.shape) == 1:
            points = points.reshape(-1, self.dimensions)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        return distances
    
    def _compute_ratio(self, points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        
        distances = self._compute_distances(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0
            
        return d_min / d_max
    
    def _compute_voronoi_uniformity(self, points):
        """Compute Voronoi cell uniformity factor."""
        try:
            vor = Voronoi(points)
            areas = []
            for region in vor.regions:
                if not any(v == -1 for v in region):  # Skip infinite regions
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) >= 3:
                        # Simple area calculation for convex polygons
                        area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] - 
                                           polygon[(i+1)%len(polygon)][0] * polygon[i][1] 
                                           for i in range(len(polygon))))
                        areas.append(area)
            avg_area = np.mean(areas) if areas else 0
            return avg_area / (1.0/self.n_points) if avg_area > 0 else 0
        except:
            return 0
    
    def _compute_fitness(self, points):
        """Compute combined fitness with Voronoi uniformity."""
        ratio = self._compute_ratio(points)
        uniformity = self._compute_voronoi_uniformity(points)
        return ratio * (1 + 0.5 * uniformity)
    
    def _generate_hexagonal_grid(self):
        """Generate points in a sophisticated hexagonal pattern."""
        points = []
        center_x, center_y = 0.5, 0.5
        
        # Create a 4x4 grid with alternating rows for better coverage
        rows, cols = 4, 4
        spacing = 0.25
        
        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    x = j * spacing + (i % 2) * spacing/2
                    y = i * spacing
                    
                    # Add slight variations to break symmetries
                    x += np.random.normal(0, 0.005, 1)[0]
                    y += np.random.normal(0, 0.005, 1)[0]
                    
                    points.append([x, y])
        
        # Clip to valid bounds
        points = np.array(points[:self.n_points])
        points = np.clip(points, 0, 1)
        return points
    
    def _generate_spiral_pattern(self):
        """Generate points using a spiral pattern."""
        points = []
        center_x, center_y = 0.5, 0.5
        angle_step = 0.5
        radius_step = 0.01
        angle = 0
        radius = 0.05
        
        while len(points) < self.n_points:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Add small noise to prevent perfect symmetry
            x += np.random.normal(0, 0.005)
            y += np.random.normal(0, 0.005)
            
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append([x, y])
            
            angle += angle_step
            radius += radius_step
            
            if radius > 0.5:
                break
        
        return np.array(points[:self.n_points])
    
    def _generate_grid_points(self):
        """Generate points in a regular grid pattern."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        return points
    
    def _generate_initial_population(self, pop_size=50):
        """Generate diverse initial population."""
        population = []
        
        # Add different types of initial configurations
        for _ in range(pop_size // 3):
            points = self._generate_hexagonal_grid()
            points += np.random.normal(0, 0.01, points.shape)
            points = np.clip(points, 0, 1)
            population.append(points.copy())
        
        for _ in range(pop_size // 3):
            points = self._generate_spiral_pattern()
            points += np.random.normal(0, 0.01, points.shape)
            points = np.clip(points, 0, 1)
            population.append(points.copy())
        
        for _ in range(pop_size - len(population)):
            points = np.random.rand(self.n_points, self.dimensions)
            population.append(points.copy())
        
        return population
    
    def _evolutionary_optimize(self):
        """Perform evolutionary optimization to find good starting points."""
        population = self._generate_initial_population(50)
        best_fitness = -np.inf
        best_individual = None
        
        # Evolution parameters
        generations = 100
        elite_size = 5
        tournament_size = 3
        
        for generation in range(generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness = self._compute_fitness(individual)
                fitness_scores.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            for i in range(elite_size):
                new_population.append(population[i].copy())
            
            # Generate offspring through tournament selection and crossover
            while len(new_population) < len(population):
                # Tournament selection
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # Select second parent
                tournament_indices2 = np.random.choice(len(population), tournament_size)
                tournament_fitness2 = [fitness_scores[i] for i in tournament_indices2]
                winner_idx2 = tournament_indices2[np.argmax(tournament_fitness2)]
                
                # Crossover (uniform)
                parent1, parent2 = population[winner_idx], population[winner_idx2]
                mask = np.random.rand(*parent1.shape) > 0.5
                child = np.where(mask, parent1, parent2).copy()
                
                # Mutation
                mutation_strength = 0.01
                for i in range(len(child)):
                    if np.random.rand() < 0.3:  # 30% chance of mutation
                        child[i] += np.random.normal(0, mutation_strength, 2)
                
                # Clamp to bounds
                child = np.clip(child, 0, 1)
                new_population.append(child)
            
            population = new_population[:len(population)]
        
        return best_individual if best_individual is not None else self._generate_hexagonal_grid()
    
    def _local_refinement(self, points):
        """Refine solution using gradient-based methods."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self._compute_ratio(points)  # Minimize negative ratio (maximize ratio)
        
        def constraint_function(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            # Ensure all coordinates are in [0,1]  
            return np.concatenate([
                points.flatten() - 1,  # x_i - 1 <= 0
                -points.flatten()     # x_i >= 0
            ])
        
        # Flatten for optimization
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(len(x0))]
        constraints = {'type': 'ineq', 'fun': constraint_function}
        
        # First optimization attempt
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, self.dimensions)
                # Clip to ensure bounds
                refined_points = np.clip(refined_points, 0, 1)
                return refined_points
        except:
            pass
        
        return points
    
    def optimize(self):
        """Main optimization routine combining evolutionary and local refinement."""
        # Step 1: Use evolutionary optimization to find good starting points
        print("Starting evolutionary optimization...")
        initial_solution = self._evolutionary_optimize()
        
        # Step 2: Refine with local optimization
        print("Starting local refinement...")
        refined_solution = self._local_refinement(initial_solution)
        
        # Step 3: Final local refinement with gradient ascent
        final_points = refined_solution.copy()
        best_ratio = self._compute_ratio(final_points)
        
        # Gradient ascent refinement
        for _ in range(100):
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
            
            # Update points
            step_size = 0.01
            final_points = final_points + step_size * gradient
            
            # Ensure bounds
            final_points = np.clip(final_points, 0, 1)
            
            # Check for convergence
            if np.all(np.abs(gradient) < 1e-6):
                break
        
        # Final evaluation
        final_ratio = self._compute_ratio(final_points)
        if final_ratio > best_ratio:
            return final_points
        else:
            return refined_solution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDispersionOptimizer(n_points=16, dimensions=2, seed=42)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END