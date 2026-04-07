# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import warnings
import time

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

    def _compute_fitness_with_voronoi_and_penalty(self, points):
        """Compute fitness including Voronoi uniformity and boundary penalties."""
        ratio = self._compute_ratio(points)
        penalty = self._compute_penalty(points)

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
            # Apply penalty to reduce fitness of infeasible solutions
            return (ratio * (1 + 0.5 * uniformity)) - penalty
        except:
            return ratio - penalty

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
        """Generate points in a more sophisticated hexagonal grid pattern with enhanced symmetry breaking."""
        # Create a proper triangular lattice that better approximates optimal point distribution
        points = []

        # Determine grid dimensions for 16 points
        # Using a hexagonal packing approach with approximately sqrt(16) = 4 points per side
        rows = 4
        cols = 4

        # Calculate spacing for hexagonal packing in unit square
        # For optimal hexagonal packing, we want points roughly equidistant
        spacing_x = 1.0 / (cols - 0.5)  # Adjusted spacing to fit better
        spacing_y = spacing_x * np.sqrt(3) / 2  # Height of equilateral triangle

        # Generate points in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    # Offset every other row for hexagonal packing
                    x = j * spacing_x + (i % 2) * spacing_x / 2
                    y = i * spacing_y

                    # Enhanced symmetry breaking with prime-based and unique perturbations
                    # Use prime numbers to create unique asymmetry patterns
                    prime_i = (i * 7) % 11  # Prime-based perturbation for i
                    prime_j = (j * 13) % 17  # Prime-based perturbation for j

                    # Apply rotational and displacement asymmetry
                    angle_factor = 0.01
                    rot_angle = (prime_i * prime_j) * 0.1

                    # Create rotation matrix for asymmetry
                    cos_a = np.cos(rot_angle)
                    sin_a = np.sin(rot_angle)

                    # Apply rotation and additional perturbations
                    x_rot = x * cos_a - y * sin_a
                    y_rot = x * sin_a + y * cos_a

                    # Add unique displacement based on prime factors and position
                    x_offset = 0.005 * np.sin(prime_i * 0.3) * np.cos(prime_j * 0.7) + \
                              0.003 * np.sin(i * 0.5 + j * 0.2)
                    y_offset = 0.005 * np.cos(prime_i * 0.4) * np.sin(prime_j * 0.6) + \
                              0.003 * np.cos(i * 0.3 + j * 0.8)

                    points.append([x_rot + x_offset, y_rot + y_offset])

        # Ensure we have exactly n_points
        points = np.array(points[:self.n_points])

        # Add small random noise to further break remaining symmetries
        points += np.random.normal(0, 0.002, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_random_points(self):
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)

    def _generate_initial_population(self, pop_size=50):
        """Generate diverse initial population."""
        population = []

        # Add Fibonacci spiral points
        for _ in range(pop_size // 4):
            points = self._generate_fibonacci_spiral_points()
            population.append(points.copy())

        # Add hexagonal grid points
        for _ in range(pop_size // 4):
            points = self._generate_hexagonal_grid()
            population.append(points.copy())

        # Add random points
        for _ in range(pop_size // 2):
            points = self._generate_random_points()
            population.append(points.copy())

        return population

    def _evolutionary_search(self):
        """Perform evolutionary optimization to find good starting points."""
        population = self._generate_initial_population(50)
        best_fitness = -np.inf
        best_individual = None

        # Evolution parameters
        generations = 100
        elite_size = 5
        tournament_size = 5

        for generation in range(generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness = self._compute_fitness_with_voronoi_and_penalty(individual)
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
                    mutation_strength = 0.015
                    for i in range(len(child)):
                        child[i] += np.random.normal(0, mutation_strength, 2)

                # Clamp to bounds
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

    def _adaptive_local_optimization(self, points, max_iter=200):
        """Optimize using adaptive local search with multiple strategies."""
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

        # Try multiple optimization methods
        best_points = points.copy()
        best_ratio = self._compute_ratio(best_points)

        # Try SLSQP first
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6, 'eps': 1e-4}
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

        # Try L-BFGS-B second
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
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

    def _simulated_annealing_refinement(self, points, max_iter=5000):
        """Refine solution using adaptive simulated annealing."""
        # Initialize parameters
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Adaptive cooling schedule
        temperature = 0.1
        cooling_rate = 0.9995
        min_temperature = 1e-6

        # Track improvements for adaptive cooling
        last_improvement = 0
        improvement_threshold = 0.0001

        # Main optimization loop
        for iteration in range(max_iter):
            # Adjust cooling rate based on progress
            if iteration % 100 == 0 and iteration > 0:
                # Check recent improvement
                if abs(current_ratio - best_ratio) < improvement_threshold:
                    last_improvement += 1
                    if last_improvement > 3:
                        cooling_rate *= 0.99  # Slow cooling when stagnant
                else:
                    last_improvement = 0
                    # Speed up cooling when making progress
                    if cooling_rate < 0.9995:
                        cooling_rate = min(cooling_rate * 1.01, 0.9995)

            # Generate neighbor solution - use neighborhood moves
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

    def _smart_gradient_refinement(self, points, max_iter=500):
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
        """Main optimization routine with multi-start approach."""
        best_solution = None
        best_ratio = -np.inf

        # Run multiple optimization attempts from different starting points
        start_points_strategies = [
            self._generate_fibonacci_spiral_points,
            self._generate_hexagonal_grid,
            self._generate_random_points
        ]

        results = []

        for i, strategy in enumerate(start_points_strategies):
            print(f"Starting optimization attempt {i+1}...")

            # Generate initial points
            initial_points = strategy()

            # Stage 1: Evolutionary search with enhanced parameters
            try:
                evolved_points = self._evolutionary_search()

                # Stage 2: Adaptive local optimization
                optimized_points = self._adaptive_local_optimization(evolved_points)

                # Stage 3: Simulated annealing refinement
                sa_points = self._simulated_annealing_refinement(optimized_points)

                # Stage 4: Smart gradient refinement
                final_points = self._smart_gradient_refinement(sa_points)

                # Evaluate final result
                final_ratio = self._compute_ratio(final_points)
                results.append((final_points, final_ratio))

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