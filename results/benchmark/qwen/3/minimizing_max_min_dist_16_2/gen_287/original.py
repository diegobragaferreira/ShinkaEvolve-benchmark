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

    def _compute_voronoi_uniformity(self, points):
        """Compute Voronoi cell uniformity factor."""
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
            return avg_area / (1.0/self.n_points) if avg_area > 0 else 0
        except:
            return 0

    def _compute_fitness(self, points):
        """Compute fitness with Voronoi uniformity enhancement."""
        ratio = self._compute_ratio(points)
        uniformity = self._compute_voronoi_uniformity(points)
        return ratio * (1 + 0.5 * uniformity)

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
        points += np.random.normal(0, 0.008, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _generate_symmetric_hexagonal_grid(self):
        """Generate points in a hexagonal grid with enhanced symmetry breaking."""
        points = []
        # Create a 4x4 grid with alternating rows offset
        for i in range(4):
            for j in range(4):
                if len(points) < self.n_points:
                    x = j * 0.25 + (i % 2) * 0.125
                    y = i * 0.25
                    points.append([x, y])

        # Apply sophisticated symmetry breaking based on hexagonal lattice structure
        points_array = np.array(points[:self.n_points])

        # Apply more effective perturbations that respect hexagonal symmetry while breaking it
        for i in range(len(points_array)):
            if i < len(points_array):
                # Create systematic perturbations based on position in hexagonal grid
                row = i // 4
                col = i % 4

                # Position-dependent perturbations that break hexagonal symmetry
                # Use higher frequency trigonometric functions to create more varied asymmetry
                row_perturbation = 0.008 * np.sin(row * 2.1 + col * 1.3) * np.cos(row * 0.7 + col * 1.9)
                col_perturbation = 0.008 * np.cos(row * 1.5 + col * 2.7) * np.sin(row * 1.1 + col * 0.9)

                # Apply perturbations with carefully chosen magnitudes
                points_array[i][0] += row_perturbation
                points_array[i][1] += col_perturbation

                # Add small random component for additional asymmetry
                points_array[i] += np.random.normal(0, 0.002, 2)

        points_array = np.clip(points_array, 0, 1)
        return points_array

    def _generate_triangular_lattice(self):
        """Generate points in a triangular lattice pattern."""
        points = []
        # Create triangular lattice pattern
        spacing = 0.25
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    # Apply triangular lattice offset
                    x = j * spacing + (i % 2) * spacing/2
                    y = i * spacing * np.sqrt(3)/2

                    # Add noise for symmetry breaking
                    x += np.random.normal(0, 0.004, 1)[0]
                    y += np.random.normal(0, 0.004, 1)[0]

                    points.append([x, y])

        points = np.array(points[:self.n_points])
        points = np.clip(points, 0, 1)
        return points

    def _generate_perturbed_grid(self):
        """Generate perturbed regular grid."""
        points = []
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]

        # Add systematic perturbations
        for i in range(len(points)):
            if i % 3 == 0:
                points[i] += np.random.normal(0, 0.008, 2)
            elif i % 3 == 1:
                points[i] += np.random.normal(0, 0.005, 2)
            else:
                points[i] += np.random.normal(0, 0.002, 2)

        points = np.clip(points, 0, 1)
        return points

    def _generate_random_points(self):
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)

    def _generate_multiple_initializations(self):
        """Generate diverse initial population."""
        initial_configs = []

        # Add different initialization strategies
        initial_configs.append(self._generate_fibonacci_spiral_points())
        initial_configs.append(self._generate_symmetric_hexagonal_grid())
        initial_configs.append(self._generate_triangular_lattice())
        initial_configs.append(self._generate_perturbed_grid())
        initial_configs.append(self._generate_random_points())

        return initial_configs

    def _evolutionary_search(self):
        """Perform evolutionary optimization with improved parameters."""
        population = self._generate_multiple_initializations()
        best_fitness = -np.inf
        best_individual = None

        # Evolution parameters with adaptive elements
        generations = 80
        elite_size = 4
        tournament_size = 4

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

            # Create new population with adaptive elitism
            new_population = []

            # Elitism: keep best individuals
            for i in range(elite_size):
                new_population.append(population[i].copy())

            # Generate offspring through tournament selection and crossover
            while len(new_population) < len(population):
                # Tournament selection with adaptive tournament size
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

                # Adaptive mutation rate based on generation
                mutation_rate = max(0.1, 0.3 - (generation / generations) * 0.2)
                if np.random.rand() < mutation_rate:  # Variable mutation rate
                    mutation_strength = 0.02 * (1 - generation / generations)
                    for i in range(len(child)):
                        child[i] += np.random.normal(0, mutation_strength, 2)

                # Clamp to bounds
                child = np.clip(child, 0, 1)
                new_population.append(child)

            population = new_population[:len(population)]

        return best_individual if best_individual is not None else self._generate_fibonacci_spiral_points()

    def _neighborhood_based_move(self, points, neighbor_size=3):
        """Apply neighborhood-based perturbation for better exploration."""
        new_points = points.copy()

        # Select random subset of points to move together
        indices = np.random.choice(len(points), min(neighbor_size, len(points)), replace=False)

        # Move selected points by small amounts
        for idx in indices:
            # Add larger perturbations for better exploration
            new_points[idx] += np.random.normal(0, 0.015, 2)

        # Clamp to bounds
        new_points = np.clip(new_points, 0, 1)
        return new_points

    def _adaptive_simulated_annealing(self, points, max_iter=3000):
        """Refine solution using adaptive simulated annealing with better cooling."""
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Initial temperature and adaptive cooling schedule
        temperature = 0.15
        cooling_rate = 0.9995
        min_temperature = 1e-8

        # Track improvements for adaptive cooling
        last_improvement = 0
        improvement_threshold = 0.00005

        convergence_counter = 0
        max_convergence = 200

        # Main optimization loop
        for iteration in range(max_iter):
            # Adjust cooling rate based on progress
            if iteration % 100 == 0 and iteration > 0:
                # Check recent improvement
                if abs(current_ratio - best_ratio) < improvement_threshold:
                    last_improvement += 1
                    if last_improvement > 2:
                        # Slow cooling when stagnant
                        cooling_rate = max(0.999, cooling_rate * 0.98)
                else:
                    last_improvement = 0
                    # Speed up cooling when making progress
                    if cooling_rate < 0.9995:
                        cooling_rate = min(cooling_rate * 1.02, 0.9995)

            # Generate neighbor solution using neighborhood moves
            new_points = self._neighborhood_based_move(current_points, neighbor_size=2)

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
                    convergence_counter = 0  # Reset convergence counter on improvement
                else:
                    convergence_counter += 1  # Increment counter for no improvement
            else:
                convergence_counter += 1  # Increment counter for rejected move

            # Early stopping if no improvement for many iterations
            if convergence_counter >= max_convergence:
                break

            # Cool down temperature
            temperature *= cooling_rate

            # Early stopping if temperature gets too low
            if temperature < min_temperature:
                break

        return best_points

    def _smart_gradient_refinement(self, points, max_iter=300):
        """Refine solution using smart gradient-based approach with adaptive steps."""
        final_points = points.copy()
        best_ratio = self._compute_ratio(final_points)

        # Dynamic step size adaptation
        initial_step_size = 0.02
        decay_factor = 0.99

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

            # Adaptive step size
            step_size = initial_step_size * (decay_factor ** iteration)

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
        """Main optimization routine with multi-start approach and adaptive refinement."""
        best_solution = None
        best_ratio = -np.inf

        # Run multiple optimization attempts from different starting points
        start_point_strategies = [
            self._generate_fibonacci_spiral_points,
            self._generate_symmetric_hexagonal_grid,
            self._generate_triangular_lattice,
            self._generate_perturbed_grid,
            self._generate_random_points
        ]

        results = []

        for i, strategy in enumerate(start_point_strategies):
            print(f"Starting optimization attempt {i+1}...")

            try:
                # Generate initial points
                initial_points = strategy()

                # Stage 1: Evolutionary search with enhanced parameters
                evolved_points = self._evolutionary_search()

                # Stage 2: Adaptive simulated annealing refinement
                sa_points = self._adaptive_simulated_annealing(evolved_points)

                # Stage 3: Smart gradient refinement
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