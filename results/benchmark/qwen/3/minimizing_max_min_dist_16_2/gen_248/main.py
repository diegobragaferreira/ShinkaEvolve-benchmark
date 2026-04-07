# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
from scipy.optimize import minimize
import time
from typing import Tuple, List

class GeometricParticleSwarmOptimizer:
    """Geometric Particle Swarm Optimization for point dispersion maximization."""

    def __init__(self, n_particles: int = 30, n_dimensions: int = 2, n_points: int = 16,
                 max_iterations: int = 1000, inertia: float = 0.7, cognitive: float = 1.5,
                 social: float = 1.5, seed: int = 42):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.n_points = n_points
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.seed = seed
        np.random.seed(seed)

        # Swarm parameters
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = 0.0

        # Adaptive cooling parameters
        self.initial_inertia = inertia
        self.final_inertia = 0.2
        self.cooling_rate = 0.999

        # Force parameters
        self.repulsion_strength = 0.1
        self.attraction_strength = 0.05
        self.boundary_repulsion = 1.0

    def compute_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix efficiently."""
        return squareform(pdist(points))

    def calculate_min_max_ratio(self, distance_matrix: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distances."""
        off_diagonal = distance_matrix[distance_matrix > 0]
        if len(off_diagonal) == 0:
            return 0.0
        d_min = np.min(off_diagonal)
        d_max = np.max(off_diagonal)
        return d_min / d_max if d_max > 0 else 0.0

    def _generate_hexagonal_lattice(self) -> np.ndarray:
        """Generate an optimized hexagonal lattice with better packing and symmetry breaking."""
        # Use a more sophisticated approach for 16 points in hexagonal arrangement
        # Create a true hexagonal tiling pattern for 16 points
        points = []

        # Generate points on a hexagonal lattice with more careful spacing
        # For 16 points, we can use 4 rows with 4 points each in a hexagonal pattern
        sqrt3 = np.sqrt(3)
        # Calculate spacing to achieve good distribution
        row_spacing = sqrt3 / 2
        col_spacing = 1.0

        # Create hexagonal grid with more precise mathematical positioning
        for i in range(4):
            for j in range(4):
                if len(points) >= self.n_points:
                    break
                # Proper hexagonal offset
                x = j * col_spacing + (i % 2) * (col_spacing / 2)
                y = i * row_spacing
                points.append([x, y])

        points = np.array(points[:self.n_points])

        # Normalize properly with better scaling
        if len(points) > 0:
            # Calculate ranges properly
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

            # Avoid division by zero
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0

            # Scale to fit nicely in [0,1] while preserving hexagonal shape
            scale_x = 0.8 / x_range if x_range > 0 else 0.8
            scale_y = 0.8 / y_range if y_range > 0 else 0.8
            scale = min(scale_x, scale_y)

            points[:, 0] = (points[:, 0] - x_min) * scale
            points[:, 1] = (points[:, 1] - y_min) * scale

            # Center the points
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])

            points[:, 0] = points[:, 0] - center_x + 0.5
            points[:, 1] = points[:, 1] - center_y + 0.5

            # Ensure all points are within bounds
            points[:, 0] = np.clip(points[:, 0], 0, 1)
            points[:, 1] = np.clip(points[:, 1], 0, 1)

        # Apply advanced symmetry breaking with mathematical patterns
        # Add carefully designed perturbations with specific mathematical relationships
        if len(points) > 0:
            # Generate a sequence of perturbations that break symmetries effectively
            for i in range(len(points)):
                # Use prime-based perturbation pattern to avoid periodic symmetries
                base_magnitude = 0.008
                # Different perturbation for different positions using primes
                prime_factor = (i * 7 + 3) % 11  # Prime numbers
                magnitude = base_magnitude * (1 + 0.1 * (prime_factor / 11.0))

                # Add perturbations with directional bias based on position
                noise_x = np.random.normal(0, magnitude * 0.3)
                noise_y = np.random.normal(0, magnitude * 0.3)

                # Apply additional directional bias to break rotational symmetry
                if i < 4:  # Corner points
                    noise_x *= 1.5
                    noise_y *= 1.5
                elif i % 4 == 0 or i % 4 == 3:  # Edge points
                    noise_x *= 1.2
                    noise_y *= 1.2

                points[i, 0] += noise_x
                points[i, 1] += noise_y

            # Final clipping to ensure bounds are respected
            points = np.clip(points, 0, 1)

        return points

    def _generate_grid_points(self) -> np.ndarray:
        """Generate grid-based points with structured perturbations."""
        grid_size = int(np.ceil(np.sqrt(self.n_points)))
        points = []

        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= self.n_points:
                    break
                x = j / (grid_size - 1) if grid_size > 1 else 0.5
                y = i / (grid_size - 1) if grid_size > 1 else 0.5
                points.append([x, y])

        return np.array(points[:self.n_points])

    def initialize_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particles with diverse geometric strategies."""
        particles = []

        # Strategy 1: Hexagonal lattice with perturbations (from geometric approach)
        hex_points = self._generate_hexagonal_lattice()
        for _ in range(self.n_particles // 3):
            particle = hex_points + np.random.normal(0, 0.02, hex_points.shape)
            particle = np.clip(particle, 0, 1)
            particles.append(particle.flatten())

        # Strategy 2: Random distribution
        for _ in range(self.n_particles // 3):
            particle = np.random.rand(self.n_points * self.n_dimensions)
            particle = np.clip(particle, 0, 1)
            particles.append(particle)

        # Strategy 3: Grid-based with structured perturbations (from geometric approach)
        grid_points = self._generate_grid_points()
        for _ in range(self.n_particles - len(particles)):
            particle = grid_points + np.random.normal(0, 0.03, grid_points.shape)
            particle = np.clip(particle, 0, 1)
            particles.append(particle.flatten())

        return np.array(particles), np.zeros_like(particles)

    def evaluate_fitness(self, positions: np.ndarray) -> np.ndarray:
        """Calculate fitness (min/max ratio) for all particles."""
        fitness = np.zeros(positions.shape[0])

        for i, pos in enumerate(positions):
            points = pos.reshape(self.n_points, self.n_dimensions)
            points = np.clip(points, 0, 1)

            try:
                dist_matrix = self.compute_distance_matrix(points)
                fitness[i] = self.calculate_min_max_ratio(dist_matrix)
            except Exception:
                fitness[i] = 0.0

        return fitness

    def apply_forces(self, positions: np.ndarray, velocities: np.ndarray,
                     fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric forces to update particle positions and velocities."""
        new_velocities = velocities.copy()
        new_positions = positions.copy()

        # Update global best
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.global_best_score:
            self.global_best_score = fitness[best_idx]
            self.global_best_position = positions[best_idx].copy()

        # Update personal bests
        for i in range(self.n_particles):
            if fitness[i] > self.personal_best_scores[i]:
                self.personal_best_scores[i] = fitness[i]
                self.personal_best_positions[i] = positions[i].copy()

        # Apply forces for each particle
        for i in range(self.n_particles):
            # Initialize forces
            force = np.zeros(self.n_points * self.n_dimensions)

            # Cognitive component (move towards personal best)
            if self.personal_best_scores[i] > 0:
                cognitive_force = self.cognitive * np.random.random(self.n_points * self.n_dimensions) * \
                                (self.personal_best_positions[i] - positions[i])
                force += cognitive_force

            # Social component (move towards global best)
            if self.global_best_score > 0:
                social_force = self.social * np.random.random(self.n_points * self.n_dimensions) * \
                             (self.global_best_position - positions[i])
                force += social_force

            # Geometric forces based on current configuration (enhanced from geometric approach)
            current_points = positions[i].reshape(self.n_points, self.n_dimensions)
            current_points = np.clip(current_points, 0, 1)

            try:
                dist_matrix = self.compute_distance_matrix(current_points)

                # Enhanced force calculations
                for j in range(self.n_points):
                    for k in range(j+1, self.n_points):
                        dist = dist_matrix[j, k]
                        if dist < 0.1:  # Only consider very close points
                            diff = current_points[j] - current_points[k]
                            if np.linalg.norm(diff) > 1e-10:
                                force_dir = diff / np.linalg.norm(diff)
                                force[j*self.n_dimensions:(j+1)*self.n_dimensions] += \
                                    self.repulsion_strength * (0.1 - dist) * force_dir
                                force[k*self.n_dimensions:(k+1)*self.n_dimensions] -= \
                                    self.repulsion_strength * (0.1 - dist) * force_dir

                # Add attractive forces for distant points (from geometric approach)
                for j in range(self.n_points):
                    for k in range(j+1, self.n_points):
                        dist = dist_matrix[j, k]
                        if 0.1 < dist < 0.5:  # Medium distances
                            diff = current_points[k] - current_points[j]
                            if np.linalg.norm(diff) > 1e-10:
                                force_dir = diff / np.linalg.norm(diff)
                                force[j*self.n_dimensions:(j+1)*self.n_dimensions] -= \
                                    self.attraction_strength * (0.5 - dist) * force_dir
                                force[k*self.n_dimensions:(k+1)*self.n_dimensions] += \
                                    self.attraction_strength * (0.5 - dist) * force_dir

                # Boundary repulsion (enhanced from geometric approach)
                for j in range(self.n_points):
                    point = current_points[j]
                    boundary_force = np.zeros(2)

                    if point[0] < 0.02:
                        boundary_force[0] += self.boundary_repulsion * (0.02 - point[0])
                    elif point[0] > 0.98:
                        boundary_force[0] -= self.boundary_repulsion * (point[0] - 0.98)

                    if point[1] < 0.02:
                        boundary_force[1] += self.boundary_repulsion * (0.02 - point[1])
                    elif point[1] > 0.98:
                        boundary_force[1] -= self.boundary_repulsion * (point[1] - 0.98)

                    force[j*2:(j+1)*2] += boundary_force

            except Exception:
                pass

            # Update velocity and position
            new_velocities[i] = self.inertia * velocities[i] + force
            new_positions[i] = positions[i] + new_velocities[i]

            # Enhanced boundary handling from geometric approach
            for j in range(self.n_points):
                point = new_positions[i][j*self.n_dimensions:(j+1)*self.n_dimensions]
                # Apply boundary repulsion forces for boundary points
                boundary_repulsion_magnitude = 0.0
                if point[0] < 0.01:
                    boundary_repulsion_magnitude = (0.01 - point[0]) * self.boundary_repulsion * 2.0
                    new_positions[i][j*self.n_dimensions] = 0.01
                elif point[0] > 0.99:
                    boundary_repulsion_magnitude = (point[0] - 0.99) * self.boundary_repulsion * 2.0
                    new_positions[i][j*self.n_dimensions] = 0.99

                if point[1] < 0.01:
                    boundary_repulsion_magnitude = (0.01 - point[1]) * self.boundary_repulsion * 2.0
                    new_positions[i][j*self.n_dimensions + 1] = 0.01
                elif point[1] > 0.99:
                    boundary_repulsion_magnitude = (point[1] - 0.99) * self.boundary_repulsion * 2.0
                    new_positions[i][j*self.n_dimensions + 1] = 0.99

                # Apply small random adjustments for boundary points
                if boundary_repulsion_magnitude > 0:
                    new_positions[i][j*self.n_dimensions:(j+1)*self.n_dimensions] += \
                        np.random.normal(0, 0.001, 2)

            # Final clip to ensure bounds
            new_positions[i] = np.clip(new_positions[i], 0, 1)

        return new_positions, new_velocities

    def optimize(self) -> np.ndarray:
        """Main optimization loop."""
        # Initialize swarm
        self.particles, self.velocities = self.initialize_particles()
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = self.evaluate_fitness(self.particles)

        # Initialize global best
        best_idx = np.argmax(self.personal_best_scores)
        self.global_best_score = self.personal_best_scores[best_idx]
        self.global_best_position = self.personal_best_positions[best_idx].copy()

        # Progress tracking
        self.best_score_history = []
        self.stagnation_counter = 0
        self.max_stagnation = 50

        # Evolution loop
        for iteration in range(self.max_iterations):
            # Evaluate fitness
            fitness = self.evaluate_fitness(self.particles)

            # Apply adaptive cooling and parameter updates
            if iteration > 0 and iteration % 10 == 0:
                # Track best score history for stagnation detection
                current_best = np.max(fitness)
                self.best_score_history.append(current_best)
                if len(self.best_score_history) > 10:
                    self.best_score_history.pop(0)

                # Check for stagnation
                if len(self.best_score_history) >= 10:
                    improvement = current_best - self.best_score_history[0]
                    if improvement < 1e-8:
                        self.stagnation_counter += 1
                        # Reset poor performers if stagnating
                        if self.stagnation_counter > self.max_stagnation:
                            worst_idx = np.argmin(fitness)
                            self.particles[worst_idx] = self._generate_hexagonal_lattice().flatten()
                            self.velocities[worst_idx] = np.zeros(self.n_points * self.n_dimensions)
                            self.personal_best_positions[worst_idx] = self.particles[worst_idx].copy()
                            self.personal_best_scores[worst_idx] = self.evaluate_fitness(
                                self.particles[worst_idx].reshape(1, -1)
                            )[0]
                            self.stagnation_counter = 0
                    else:
                        self.stagnation_counter = 0

                # Apply adaptive cooling
                if iteration > 100:
                    # Gradually decrease inertia
                    self.inertia = max(self.final_inertia,
                                     self.initial_inertia * (self.cooling_rate ** (iteration // 20)))

            # Apply forces
            self.particles, self.velocities = self.apply_forces(
                self.particles, self.velocities, fitness
            )

            # Periodically reinitialize poor performers (from geometric approach)
            if iteration % 100 == 0 and iteration > 0:
                worst_idx = np.argmin(fitness)
                if fitness[worst_idx] < self.global_best_score * 0.5:
                    self.particles[worst_idx] = self._generate_hexagonal_lattice().flatten()
                    self.velocities[worst_idx] = np.zeros(self.n_points * self.n_dimensions)
                    self.personal_best_positions[worst_idx] = self.particles[worst_idx].copy()
                    self.personal_best_scores[worst_idx] = self.evaluate_fitness(
                        self.particles[worst_idx].reshape(1, -1)
                    )[0]

        # Apply multi-scale geometric optimization to final solution
        try:
            final_points = self.global_best_position.reshape(self.n_points, self.n_dimensions)
            final_points = np.clip(final_points, 0, 1)

            # Apply geometric local optimization
            from scipy.spatial.distance import pdist, squareform
            def compute_distance_matrix(points):
                return squareform(pdist(points))

            def calculate_min_max_ratio(distance_matrix):
                off_diagonal = distance_matrix[distance_matrix > 0]
                if len(off_diagonal) == 0:
                    return 0.0
                d_min = np.min(off_diagonal)
                d_max = np.max(off_diagonal)
                return d_min / d_max if d_max > 0 else 0.0

            def geometric_local_optimization(points, max_iter=100):
                current_points = points.copy()
                for iteration in range(max_iter):
                    try:
                        dist_matrix = compute_distance_matrix(current_points)
                        ratio = calculate_min_max_ratio(dist_matrix)
                        if ratio < 1e-10:
                            break
                    except Exception:
                        break

                    new_points = current_points.copy()
                    updated = False

                    for i in range(len(current_points)):
                        original_point = current_points[i].copy()
                        best_direction = None
                        best_improvement = 0

                        directions = [
                            [0.001, 0], [0, 0.001], [-0.001, 0], [0, -0.001],
                            [0.000707, 0.000707], [-0.000707, 0.000707],
                            [0.000707, -0.000707], [-0.000707, -0.000707]
                        ]

                        for dx, dy in directions:
                            test_point = original_point + [dx, dy]
                            test_point = np.clip(test_point, 0, 1)

                            test_points = current_points.copy()
                            test_points[i] = test_point

                            try:
                                test_dist_matrix = compute_distance_matrix(test_points)
                                test_ratio = calculate_min_max_ratio(test_dist_matrix)

                                if test_ratio > ratio + best_improvement:
                                    best_improvement = test_ratio - ratio
                                    best_direction = [dx, dy]

                            except Exception:
                                continue

                        if best_direction is not None and best_improvement > 1e-12:
                            new_points[i] = original_point + best_direction
                            updated = True

                    if updated:
                        current_points = new_points.copy()
                    else:
                        break

                return current_points

            # Apply the geometric refinement
            refined_points = geometric_local_optimization(final_points, max_iter=50)
            final_points = np.clip(refined_points, 0, 1)

        except Exception:
            # Fallback to base solution
            final_points = self.global_best_position.reshape(self.n_points, self.n_dimensions)

        return final_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()

    # Create optimizer with tuned parameters for this specific problem
    optimizer = GeometricParticleSwarmOptimizer(
        n_particles=30,
        n_points=16,
        max_iterations=800,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=42
    )

    # Run optimization
    best_points = optimizer.optimize()

    end_time = time.time()
    eval_time = end_time - start_time

    return best_points

# EVOLVE-BLOCK-END