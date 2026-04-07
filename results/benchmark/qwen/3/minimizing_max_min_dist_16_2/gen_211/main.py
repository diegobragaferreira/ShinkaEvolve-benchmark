# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a volumetric evolutionary approach based on simplex geometry optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    class VolumetricEvolutionaryOptimizer:
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
            
        def _compute_volume_ratio(self, points):
            """Compute volume-based quality metric for better geometric distribution."""
            if len(points) < 3:
                return 0
                
            try:
                hull = ConvexHull(points)
                volume = hull.volume
                # Normalize by the maximum possible volume in unit square
                max_volume = 1.0  # For unit square
                return volume / max_volume if max_volume > 0 else 0
            except:
                return 0
                
        def _create_initial_configuration(self):
            """Create diverse initial configurations using geometric principles."""
            configurations = []
            
            # 1. Equilateral triangular lattice
            triangular_points = self._generate_triangular_lattice()
            configurations.append(triangular_points)
            
            # 2. Hexagonal grid with systematic perturbations
            hex_points = self._generate_hexagonal_grid()
            configurations.append(hex_points)
            
            # 3. Fibonacci spiral projection
            fib_points = self._generate_fibonacci_spiral()
            configurations.append(fib_points)
            
            # 4. Random with boundary avoidance
            random_points = self._generate_boundary_aware_random()
            configurations.append(random_points)
            
            # 5. Perturbed uniform grid
            uniform_points = self._generate_perturbed_grid()
            configurations.append(uniform_points)
            
            return configurations
            
        def _generate_triangular_lattice(self):
            """Generate points in triangular lattice pattern."""
            points = []
            # Arrange in triangular grid (4x4)
            for i in range(4):
                for j in range(4):
                    if len(points) < self.n_points:
                        x = j + (i % 2) * 0.5
                        y = i * np.sqrt(3)/2
                        points.append([x, y])
            
            points = np.array(points[:self.n_points])
            # Normalize to [0,1] square
            max_x = 3 + 0.5
            max_y = 3 * np.sqrt(3)/2
            points[:, 0] /= max_x
            points[:, 1] /= max_y
            points += np.random.normal(0, 0.01, points.shape)
            points = np.clip(points, 0, 1)
            return points
            
        def _generate_hexagonal_grid(self):
            """Generate points in hexagonal grid with enhanced symmetry breaking."""
            points = []
            # Create 4x4 grid with hexagonal packing
            for i in range(4):
                for j in range(4):
                    if len(points) < self.n_points:
                        x = j + 0.5 * (i % 2)
                        y = i * np.sqrt(3)/2
                        
                        # Add structured perturbations
                        perturbation = 0.01 * np.sin(i * 0.5 + j * 0.3)
                        x += perturbation * np.cos(i * 0.2)
                        y += perturbation * np.sin(j * 0.4)
                        
                        points.append([x, y])
            
            points = np.array(points[:self.n_points])
            # Normalize to [0,1] square
            max_x = 3 + 0.5
            max_y = 3 * np.sqrt(3)/2
            points[:, 0] /= max_x
            points[:, 1] /= max_y
            points += np.random.normal(0, 0.005, points.shape)
            points = np.clip(points, 0, 1)
            return points
            
        def _generate_fibonacci_spiral(self):
            """Generate points using Fibonacci spiral."""
            points = []
            golden_ratio = (1 + np.sqrt(5)) / 2
            
            for i in range(min(16, 32)):  # More points for better spiral
                angle = i * 2 * np.pi / golden_ratio
                radius = np.sqrt(i / 31.0)  # Normalize to [0,1]
                x = 0.5 + radius * np.cos(angle) * 0.4
                y = 0.5 + radius * np.sin(angle) * 0.4
                points.append([x, y])
                
            points = np.array(points[:self.n_points])
            # Add small noise for symmetry breaking
            points += np.random.normal(0, 0.01, points.shape)
            points = np.clip(points, 0, 1)
            return points
            
        def _generate_boundary_aware_random(self):
            """Generate random points with boundary awareness."""
            points = np.random.rand(self.n_points, self.dimensions)
            
            # Push points away from boundaries
            for i in range(len(points)):
                # Boundary repulsion force
                boundary_repulsion = 0.02
                if points[i, 0] < boundary_repulsion:
                    points[i, 0] = boundary_repulsion + np.random.uniform(0, boundary_repulsion/2)
                elif points[i, 0] > 1 - boundary_repulsion:
                    points[i, 0] = 1 - boundary_repulsion - np.random.uniform(0, boundary_repulsion/2)
                    
                if points[i, 1] < boundary_repulsion:
                    points[i, 1] = boundary_repulsion + np.random.uniform(0, boundary_repulsion/2)
                elif points[i, 1] > 1 - boundary_repulsion:
                    points[i, 1] = 1 - boundary_repulsion - np.random.uniform(0, boundary_repulsion/2)
                    
            return points
            
        def _generate_perturbed_grid(self):
            """Generate perturbed uniform grid."""
            points = np.zeros((self.n_points, self.dimensions))
            side = int(np.ceil(np.sqrt(self.n_points)))
            
            for i in range(self.n_points):
                j = i // side
                k = i % side
                points[i, 0] = k / (side - 1) if side > 1 else 0.5
                points[i, 1] = j / (side - 1) if side > 1 else 0.5
                
            # Add systematic perturbations
            for i in range(self.n_points):
                points[i, 0] += np.random.normal(0, 0.015, 1)[0] * (1.0 + 0.1 * i)
                points[i, 1] += np.random.normal(0, 0.015, 1)[0] * (1.0 + 0.1 * i)
                
            points = np.clip(points, 0, 1)
            return points
            
        def _simplex_evolution(self, initial_points, max_generations=100):
            """Evolutionary optimization using simplex-based approach."""
            # Create initial simplex from multiple configurations
            configurations = self._create_initial_configuration()
            population = configurations[:4]  # Use first 4 configurations
            
            # Pad to ensure we have at least 4 points in population
            while len(population) < 4:
                population.append(initial_points.copy())
                
            best_points = initial_points.copy()
            best_ratio = self._compute_ratio(best_points)
            
            # Track performance
            history = []
            
            for generation in range(max_generations):
                # Evaluate population
                fitness_scores = []
                for points in population:
                    ratio = self._compute_ratio(points)
                    volume_ratio = self._compute_volume_ratio(points)
                    # Combine metrics with weighted sum
                    combined_fitness = ratio * (1 + 0.3 * volume_ratio)
                    fitness_scores.append(combined_fitness)
                    
                    if combined_fitness > best_ratio:
                        best_ratio = combined_fitness
                        best_points = points.copy()
                        
                # Sort by fitness
                sorted_indices = np.argsort(fitness_scores)[::-1]
                population = [population[i] for i in sorted_indices]
                fitness_scores = [fitness_scores[i] for i in sorted_indices]
                
                # Create new population using simplex evolution
                new_population = []
                
                # Elitism: keep top performers
                elite_count = max(2, len(population) // 3)
                new_population.extend(population[:elite_count])
                
                # Generate offspring through simplex operations
                while len(new_population) < len(population):
                    # Select parents from top 50%
                    parent_indices = np.random.choice(len(population[:len(population)//2]), 2, replace=False)
                    parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
                    
                    # Create offspring via simplex combination
                    # Mix points between parents with adaptive weights
                    alpha = 0.3 + 0.7 * np.random.rand()  # Adaptive mixing coefficient
                    
                    # Simplex operation: convex combination
                    child = alpha * parent1 + (1 - alpha) * parent2
                    
                    # Add noise for diversity
                    noise_magnitude = 0.01 * (1 - generation/max_generations)
                    child += np.random.normal(0, noise_magnitude, child.shape)
                    
                    # Ensure bounds
                    child = np.clip(child, 0, 1)
                    new_population.append(child)
                
                population = new_population[:len(population)]
                
                # Track progress
                history.append(best_ratio)
                
                # Early stopping criteria
                if len(history) > 10:
                    recent_change = abs(history[-1] - history[-10])
                    if recent_change < 1e-8:
                        break
                        
            return best_points
            
        def _local_improvement(self, points, max_iter=300):
            """Local refinement using gradient-like descent."""
            current_points = points.copy()
            current_ratio = self._compute_ratio(current_points)
            best_points = current_points.copy()
            best_ratio = current_ratio
            
            # Adaptive step sizes
            step_sizes = [0.05, 0.02, 0.01, 0.005]
            
            for i, step_size in enumerate(step_sizes):
                # For each step size, make several iterations
                iterations = max_iter // len(step_sizes)
                
                for _ in range(iterations):
                    # Estimate gradient using finite differences
                    gradient = np.zeros_like(current_points)
                    eps = 1e-4
                    
                    for j in range(len(current_points)):
                        for k in range(self.dimensions):
                            # Perturb point
                            points_plus = current_points.copy()
                            points_plus[j, k] += eps
                            points_plus = np.clip(points_plus, 0, 1)
                            
                            points_minus = current_points.copy()
                            points_minus[j, k] -= eps
                            points_minus = np.clip(points_minus, 0, 1)
                            
                            # Calculate finite difference
                            ratio_plus = self._compute_ratio(points_plus)
                            ratio_minus = self._compute_ratio(points_minus)
                            
                            gradient[j, k] = (ratio_plus - ratio_minus) / (2 * eps)
                    
                    # Update points
                    current_points = current_points + step_size * gradient
                    
                    # Bound checking
                    current_points = np.clip(current_points, 0, 1)
                    
                    # Evaluate
                    new_ratio = self._compute_ratio(current_points)
                    
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = current_points.copy()
                        
                    # Convergence check
                    if np.all(np.abs(gradient) < 1e-6):
                        break
                        
            return best_points
            
        def _structured_optimization(self, points):
            """Apply structured optimization approach."""
            # Phase 1: Simplex evolution
            evolved = self._simplex_evolution(points, max_generations=50)
            
            # Phase 2: Local refinement
            refined = self._local_improvement(evolved, max_iter=200)
            
            # Phase 3: Boundary-aware smoothing
            final_points = refined.copy()
            
            # Apply boundary correction for better distribution
            for i in range(len(final_points)):
                # If too close to boundary, move inward
                boundary_threshold = 0.01
                if final_points[i, 0] < boundary_threshold:
                    final_points[i, 0] = boundary_threshold + np.random.uniform(0, boundary_threshold/2)
                elif final_points[i, 0] > 1 - boundary_threshold:
                    final_points[i, 0] = 1 - boundary_threshold - np.random.uniform(0, boundary_threshold/2)
                    
                if final_points[i, 1] < boundary_threshold:
                    final_points[i, 1] = boundary_threshold + np.random.uniform(0, boundary_threshold/2)
                elif final_points[i, 1] > 1 - boundary_threshold:
                    final_points[i, 1] = 1 - boundary_threshold - np.random.uniform(0, boundary_threshold/2)
                    
            return final_points
            
        def optimize(self):
            """Main optimization procedure."""
            # Get initial configurations
            configurations = self._create_initial_configuration()
            
            best_solution = None
            best_ratio = -np.inf
            
            # Multi-start approach - try multiple configurations
            for i, initial_points in enumerate(configurations):
                try:
                    # Apply structured optimization
                    optimized = self._structured_optimization(initial_points)
                    ratio = self._compute_ratio(optimized)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = optimized.copy()
                        
                except Exception as e:
                    warnings.warn(f"Optimization failed for configuration {i}: {str(e)}")
                    continue
                    
            # If no success, return one of the initial configurations
            if best_solution is None:
                return self._generate_triangular_lattice()
                
            return best_solution
            
    # Execute optimization
    optimizer = VolumetricEvolutionaryOptimizer(n_points=16, dimensions=2, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END