# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import warnings

class HybridPointOptimizer:
    """Hybrid optimizer combining evolutionary and gradient-based approaches for point dispersion."""

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

    def _compute_fitness_with_voronoi(self, points):
        """Compute fitness including Voronoi uniformity."""
        ratio = self._compute_ratio(points)
        try:
            vor = Voronoi(points)
            areas = []
            for region in vor.regions:
                if not any(v == -1 for v in region):  # Skip infinite regions
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) >= 3:
                        area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] -
                                           polygon[(i+1)%len(polygon)][0] * polygon[i][1]
                                           for i in range(len(polygon))))
                        areas.append(area)
            avg_area = np.mean(areas) if areas else 0
            uniformity = avg_area / (1.0/self.n_points) if avg_area > 0 else 0
            return ratio * (1 + 0.5 * uniformity)
        except:
            return ratio

    def _generate_fibonacci_spiral_points(self):
        """Generate points using Fibonacci spiral on sphere projected to 2D."""
        points = np.zeros((self.n_points, self.dimensions))
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.n_points):
            z = 1 - (i / (self.n_points - 1)) * 2  # z coordinate from -1 to 1
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)

            # Convert to Cartesian coordinates on unit sphere
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)

            # Map from sphere to square [0,1] x [0,1] using stereographic projection
            x_norm = (x + 1) / 2
            y_norm = (y + 1) / 2

            points[i] = [x_norm, y_norm]

        # Add small perturbations to break symmetries
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_hexagonal_grid(self):
        """Generate points in a hexagonal grid pattern with enhanced symmetry breaking."""
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) < self.n_points:
                    x = j * 0.25 + (i % 2) * 0.125
                    y = i * 0.25

                    # Enhanced symmetry breaking using mathematical perturbations
                    # Apply prime-based and trigonometric perturbations for better asymmetry
                    prime_i = (i * 7) % 11  # Prime-based perturbation for i
                    prime_j = (j * 13) % 17  # Prime-based perturbation for j

                    # Create more complex asymmetry patterns
                    asymmetry_factor = 0.01

                    # Use sine/cosine combinations with prime multipliers
                    x_pert = asymmetry_factor * np.sin(prime_i * 0.3) * np.cos(prime_j * 0.7)
                    y_pert = asymmetry_factor * np.cos(prime_i * 0.4) * np.sin(prime_j * 0.6)

                    # Add position-based unique perturbation
                    unique_pert = asymmetry_factor * 0.5 * np.sin((i + j) * 0.3 + i * j * 0.1)

                    x += x_pert + unique_pert
                    y += y_pert + unique_pert

                    points.append([x, y])

        # Add final small noise to ensure complete asymmetry
        points = np.array(points[:self.n_points])
        points += np.random.normal(0, 0.002, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_initial_population(self, pop_size=30):
        """Generate diverse initial population."""
        population = []

        # Add Fibonacci spiral points
        for _ in range(pop_size // 3):
            points = self._generate_fibonacci_spiral_points()
            population.append(points.copy())

        # Add hexagonal grid points
        for _ in range(pop_size // 3):
            points = self._generate_hexagonal_grid()
            population.append(points.copy())

        # Add random points
        for _ in range(pop_size - len(population)):
            points = np.random.rand(self.n_points, self.dimensions)
            population.append(points.copy())

        return population

    def _evolutionary_search(self):
        """Perform evolutionary optimization to find good starting points."""
        population = self._generate_initial_population(30)
        best_fitness = -np.inf
        best_individual = None

        # Evolution parameters
        generations = 50
        elite_size = 3
        tournament_size = 3

        for generation in range(generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness = self._compute_fitness_with_voronoi(individual)
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
                if np.random.rand() < 0.3:  # 30% chance of mutation
                    mutation_strength = 0.01
                    for i in range(len(child)):
                        child[i] += np.random.normal(0, mutation_strength, 2)

                # Clamp to bounds
                child = np.clip(child, 0, 1)
                new_population.append(child)

            population = new_population[:len(population)]

        return best_individual if best_individual is not None else self._generate_fibonacci_spiral_points()

    def _optimize_single_stage(self, points, method='SLSQP'):
        """Optimize points using specified method."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self._compute_ratio(points)  # Minimize negative ratio (maximize ratio)

        def constraint_function(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            # Return negative values for constraint violations (we want >= 0)
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

        try:
            if method == 'SLSQP':
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
                )
            else:  # L-BFGS-B
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )

            if result.success:
                refined_points = result.x.reshape(-1, self.dimensions)
                refined_points = np.clip(refined_points, 0, 1)
                return refined_points
        except Exception as e:
            warnings.warn(f"Optimization error: {str(e)}")
            return points

        return points

    def _gradient_ascent_refinement(self, points, max_iter=100):
        """Refine solution using gradient ascent."""
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

            # Update points
            step_size = 0.01
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
        # Stage 1: Evolutionary search for good starting configuration
        print("Starting evolutionary search...")
        initial_solution = self._evolutionary_search()

        # Stage 2: Local optimization with SLSQP
        print("Starting SLSQP optimization...")
        slsqp_result = self._optimize_single_stage(initial_solution, 'SLSQP')

        # Stage 3: Local optimization with L-BFGS-B
        print("Starting L-BFGS-B optimization...")
        lbfgsb_result = self._optimize_single_stage(slsqp_result, 'L-BFGS-B')

        # Stage 4: Gradient ascent refinement
        print("Starting gradient ascent refinement...")
        refined_points = self._gradient_ascent_refinement(lbfgsb_result)

        # Final check and return best solution
        final_ratio = self._compute_ratio(refined_points)
        slsqp_ratio = self._compute_ratio(slsqp_result)
        lbfgsb_ratio = self._compute_ratio(lbfgsb_result)

        if final_ratio >= slsqp_ratio and final_ratio >= lbfgsb_ratio:
            return refined_points
        elif slsqp_ratio >= lbfgsb_ratio:
            return slsqp_result
        else:
            return lbfgsb_result

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = HybridPointOptimizer(n_points=16, dimensions=2, seed=42)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END