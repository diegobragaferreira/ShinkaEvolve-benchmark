# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Optional
import math

class CirclePackingOptimizer:
    """Main optimizer class for 26-circle packing problem."""

    def __init__(self, n_circles: int = 26, seed: int = 42):
        self.n_circles = n_circles
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def validate_placement(self, circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and don't overlap."""
        n = len(circles)

        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

        # Check non-overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = cKDTree(points)

        # Find all pairs within distance 2*r (minimum separation needed to avoid overlap)
        pairs = tree.query_pairs(2 * min(circles[:, 2]), output_type='ndarray')

        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_distance_sq = (r1 + r2)**2
            if distance_sq < min_distance_sq:
                return False

        return True

    def generate_voronoi_initialization(self) -> np.ndarray:
        """Generate initial circle positions using a Voronoi-inspired spreading mechanism."""
        # Create a grid of candidate positions
        grid_size = max(3, int(np.ceil(np.sqrt(self.n_circles))))
        x_coords = np.linspace(0.05, 0.95, grid_size)
        y_coords = np.linspace(0.05, 0.95, grid_size)

        # Generate all grid points
        grid_points = []
        for x in x_coords:
            for y in y_coords:
                grid_points.append([x, y])

        # If we have more circles than grid points, add some random points
        if len(grid_points) < self.n_circles:
            extra_points = self.n_circles - len(grid_points)
            for _ in range(extra_points):
                grid_points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

        # Shuffle the points to avoid systematic bias
        random.shuffle(grid_points)

        # Take the first n_circles points
        points = np.array(grid_points[:self.n_circles])

        # Initialize circles with small radii
        circles = np.zeros((self.n_circles, 3))
        circles[:, 0] = points[:, 0]  # x coordinates
        circles[:, 1] = points[:, 1]  # y coordinates
        circles[:, 2] = 0.01         # initial small radii

        return circles

    def generate_greedy_fallback(self) -> np.ndarray:
        """Fallback method to generate a feasible configuration."""
        # Simple greedy approach: place circles in order of decreasing radius
        circles = np.zeros((self.n_circles, 3))

        # Start with small radii and gradually increase
        # Place in a way that they don't overlap initially
        positions = []
        radii = []

        # Try to place circles greedily by spacing them out
        placed = 0
        radius = 0.05
        while placed < self.n_circles and radius > 0.005:
            # Try placing circles in a spiral pattern or grid
            attempt = 0
            while attempt < 100 and placed < self.n_circles:
                # Place in grid-like fashion
                rows = int(np.sqrt(self.n_circles)) + 1
                cols = self.n_circles // rows + 1

                for i in range(rows):
                    for j in range(cols):
                        if placed >= self.n_circles:
                            break
                        x = 0.1 + j * 0.8 / cols
                        y = 0.1 + i * 0.8 / rows

                        # Check if this position is valid
                        valid = True
                        for pos, rad in zip(positions, radii):
                            dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                            if dist_sq < (rad + radius)**2:
                                valid = False
                                break

                        if valid:
                            positions.append([x, y])
                            radii.append(radius)
                            placed += 1
                attempt += 1

            radius *= 0.9  # Decrease radius slightly

        # Fill remaining circles
        while placed < self.n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
            radii.append(0.01)
            placed += 1

        circles[:, 0] = [pos[0] for pos in positions]
        circles[:, 1] = [pos[1] for pos in positions]
        circles[:, 2] = radii

        return circles

    def compute_forces(self, circles: np.ndarray,
                      repulsion_strength: float = 100.0,
                      boundary_strength: float = 50.0) -> Tuple[np.ndarray, float]:
        """
        Compute forces acting on each circle including repulsion and boundary attraction.

        Returns:
            forces: array of shape (n_circles, 2) representing force vectors
            potential_energy: total potential energy of the system
        """
        n = len(circles)
        forces = np.zeros((n, 2))
        potential_energy = 0.0

        # Repulsion forces between circles
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]

                dx = x1 - x2
                dy = y1 - y2
                distance_sq = dx*dx + dy*dy

                # Skip if too far apart
                if distance_sq > 4*(r1+r2)**2:  # Approximate cutoff
                    continue

                distance = np.sqrt(distance_sq)

                # Avoid division by zero
                if distance < 1e-10:
                    continue

                # Repulsive force (inverse square law)
                force_magnitude = repulsion_strength / (distance_sq + 1e-10)

                # Apply force direction
                fx = force_magnitude * dx / distance
                fy = force_magnitude * dy / distance

                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy

                # Energy contribution (potential energy)
                potential_energy += force_magnitude * (1.0/distance)

        # Boundary forces (spring-like attraction to boundaries)
        for i in range(n):
            x, y, r = circles[i]

            # Force components towards boundaries
            fx = 0.0
            fy = 0.0

            # Left boundary
            if x - r < 0:
                fx += boundary_strength * (0 - (x - r))
            # Right boundary
            if x + r > 1:
                fx += boundary_strength * (1 - (x + r))

            # Bottom boundary
            if y - r < 0:
                fy += boundary_strength * (0 - (y - r))
            # Top boundary
            if y + r > 1:
                fy += boundary_strength * (1 - (y + r))

            forces[i, 0] += fx
            forces[i, 1] += fy

        return forces, potential_energy

    def update_circles(self, circles: np.ndarray, forces: np.ndarray,
                      dt: float = 0.01, max_velocity: float = 0.05) -> np.ndarray:
        """
        Update circle positions based on forces and velocity limits.
        """
        updated_circles = circles.copy()

        # Update velocities and positions
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Update velocity (force gives acceleration)
            vx = forces[i, 0] * dt
            vy = forces[i, 1] * dt

            # Apply velocity limits
            vel_norm = np.sqrt(vx*vx + vy*vy)
            if vel_norm > max_velocity:
                vx = vx * max_velocity / vel_norm
                vy = vy * max_velocity / vel_norm

            # Update position
            new_x = x + vx
            new_y = y + vy

            # Ensure new positions respect boundaries
            new_x = np.clip(new_x, r, 1-r)
            new_y = np.clip(new_y, r, 1-r)

            updated_circles[i, 0] = new_x
            updated_circles[i, 1] = new_y

        return updated_circles

    def adaptive_simulation_step(self, circles: np.ndarray,
                                target_energy: float = 1e-3,
                                max_steps: int = 1000) -> np.ndarray:
        """
        Perform adaptive simulation to reach equilibrium.
        """
        # Parameters for adaptive dynamics
        dt = 0.01
        damping_factor = 0.95
        energy_threshold = target_energy

        # Track energy evolution
        prev_energy = float('inf')
        energy_history = []

        for step in range(max_steps):
            # Compute forces
            forces, current_energy = self.compute_forces(circles)

            # Check for convergence
            if len(energy_history) > 10:
                recent_avg = np.mean(energy_history[-10:])
                if abs(recent_avg - current_energy) < energy_threshold:
                    break

            # Store energy for tracking
            energy_history.append(current_energy)

            # Update circles
            circles = self.update_circles(circles, forces, dt)

            # Apply damping to prevent oscillation
            if step % 10 == 0:
                dt *= damping_factor

            # Occasionally reset time step
            if step % 50 == 0:
                dt = max(0.001, dt * 0.95)

        return circles

    def multi_scale_refinement(self, initial_circles: np.ndarray,
                              max_iterations: int = 200) -> np.ndarray:
        """
        Refine the solution using multi-scale approach:
        1. Coarse-grained optimization
        2. Fine-grained refinement
        3. Local search improvement
        """
        circles = initial_circles.copy()

        # Phase 1: Coarse optimization using large time steps and strong forces
        circles = self.adaptive_simulation_step(circles, target_energy=1e-2, max_steps=300)

        # Phase 2: Medium optimization with moderate forces
        circles = self.adaptive_simulation_step(circles, target_energy=1e-3, max_steps=200)

        # Phase 3: Fine-tuning with very small forces and high precision
        circles = self.adaptive_simulation_step(circles, target_energy=1e-4, max_steps=150)

        return circles

    def constraint_aware_local_search(self, circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """
        Apply constraint-aware local search to fine-tune the solution.
        This version focuses on finding local optima while respecting constraints.
        """
        improved_circles = circles.copy()
        n = len(improved_circles)

        for iteration in range(max_iterations):
            improved = False

            # Try to increase radii while maintaining constraints
            for i in range(n):
                x, y, r = improved_circles[i]

                # Calculate maximum possible radius at this position
                max_radius = min(x, 1-x, y, 1-y)

                # Find neighboring circles to check constraints
                neighbors = []
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        min_dist_sq = (r + r2)**2
                        neighbors.append((j, dist_sq, min_dist_sq))

                # Calculate maximum increase in radius
                max_incr = max_radius - r

                # Check overlap constraints more rigorously
                min_dist_to_neighbor = float('inf')
                for _, dist_sq, min_dist_sq in neighbors:
                    if dist_sq < min_dist_sq:
                        min_dist_to_neighbor = min(min_dist_to_neighbor, dist_sq)

                # If we have space to grow and can still satisfy constraints
                if max_incr > 0.001:
                    # Binary search for maximum possible radius
                    low, high = r, max_radius
                    best_radius = r

                    # Binary search for maximum safe radius
                    for _ in range(20):  # Limit iterations
                        mid = (low + high) / 2
                        valid = True

                        # Check if this radius creates overlap issues
                        for j, dist_sq, min_dist_sq in neighbors:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (x - x2)**2 + (y - y2)**2
                            if dist_sq < (mid + r2)**2:
                                valid = False
                                break

                        if valid:
                            best_radius = mid
                            low = mid
                        else:
                            high = mid

                    # Apply the found safe radius if it's an improvement
                    if best_radius > r + 1e-6:
                        improved_circles[i, 2] = best_radius
                        improved = True

            # If no improvement from radius increases, try position adjustments
            if not improved:
                for i in range(n):
                    x, y, r = improved_circles[i]

                    # Try small movements in 8 directions
                    movements = [(-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                               (0, -0.005),              (0, 0.005),
                               (0.005, -0.005), (0.005, 0), (0.005, 0.005)]

                    best_x, best_y = x, y
                    best_score = -float('inf')
                    best_radius = r

                    for dx, dy in movements:
                        new_x, new_y = x + dx, y + dy

                        # Check bounds
                        if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                            continue

                        # Check overlap with neighbors
                        overlap_penalty = 0
                        valid = True
                        for j in range(n):
                            if i != j:
                                x2, y2, r2 = improved_circles[j]
                                dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                                min_dist_sq = (r + r2)**2
                                if dist_sq < min_dist_sq:
                                    overlap_penalty += (min_dist_sq - dist_sq) * 1000
                                    valid = False

                        if valid:
                            # Score based on overlap reduction and radius preservation
                            score = -overlap_penalty + r
                            if score > best_score:
                                best_score = score
                                best_x, best_y = new_x, new_y

                    # Apply the best movement if it helps
                    if best_x != x or best_y != y:
                        improved_circles[i, 0] = best_x
                        improved_circles[i, 1] = best_y
                        improved = True

            # If no improvement made, exit loop
            if not improved:
                break

        return improved_circles

    def improve_with_gradient_descent(self, circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """
        Apply gradient-based improvement focusing on increasing radii while respecting constraints.
        """
        improved_circles = circles.copy()
        n = len(improved_circles)

        for iteration in range(max_iter):
            # For each circle, compute how much we can increase radius
            updated = False

            for i in range(n):
                x, y, r = improved_circles[i]
                max_radius = min(x, 1-x, y, 1-y)

                # Find minimum distance to neighbors for overlap constraints
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        if dist_sq < min_dist:
                            min_dist = dist_sq

                # If we can increase radius
                current_max_radius = max_radius
                if min_dist < (r + 0.001)**2:
                    # We're currently overlapping, reduce radius
                    # Or find maximum safe radius
                    safe_radius = min(r, max_radius)
                    if safe_radius > r + 1e-6:
                        improved_circles[i, 2] = safe_radius
                        updated = True
                else:
                    # Try to increase radius up to boundary
                    if r < max_radius - 1e-6:
                        # Test if we can safely increase
                        test_radius = min(max_radius, r + 0.005)
                        valid = True
                        for j in range(n):
                            if i != j:
                                x2, y2, r2 = improved_circles[j]
                                dist_sq = (x - x2)**2 + (y - y2)**2
                                if dist_sq < (test_radius + r2)**2:
                                    valid = False
                                    break

                        if valid:
                            improved_circles[i, 2] = test_radius
                            updated = True

            if not updated:
                break

        return improved_circles

    def calculate_geometric_similarity(self, circles1: np.ndarray, circles2: np.ndarray) -> float:
        """Calculate geometric similarity between two circle configurations."""
        # Normalize positions to [0,1] range
        pos1 = circles1[:, :2]
        pos2 = circles2[:, :2]

        # Calculate average Euclidean distance between corresponding circles
        distances = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
        avg_distance = np.mean(distances)

        # Similarity is inversely related to average distance (normalized)
        similarity = 1.0 / (1.0 + avg_distance)
        return similarity

    def cluster_similar_individuals(self, population: List[np.ndarray],
                                  threshold: float = 0.8) -> List[List[int]]:
        """Cluster individuals based on geometric similarity."""
        n = len(population)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            # Start new cluster
            cluster = [i]
            visited[i] = True

            # Find similar individuals
            for j in range(i+1, n):
                if visited[j]:
                    continue

                similarity = self.calculate_geometric_similarity(population[i], population[j])
                if similarity >= threshold:
                    cluster.append(j)
                    visited[j] = True

            clusters.append(cluster)

        return clusters

    def speciated_evolution(self, n_generations: int = 50,
                           population_size: int = 50,
                           speciation_threshold: float = 0.8) -> np.ndarray:
        """Perform evolution with speciation to maintain diversity."""
        # Generate initial population
        population = []
        for _ in range(population_size):
            circles = self.generate_voronoi_initialization()
            population.append(circles)

        best_fitness = -float('inf')
        best_individual = None

        # Evolution loop
        for generation in range(n_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for circles in population:
                if self.validate_placement(circles):
                    fitness = np.sum(circles[:, 2])
                else:
                    fitness = -float('inf')
                fitness_scores.append(fitness)

            # Update best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()

            # Create clusters of similar individuals
            clusters = self.cluster_similar_individuals(population, speciation_threshold)

            # Create new population by evolving each cluster separately
            new_population = []

            for cluster in clusters:
                if len(cluster) == 0:
                    continue

                # Select individuals from cluster
                cluster_individuals = [population[i] for i in cluster]
                cluster_fitness = [fitness_scores[i] for i in cluster]

                # Sort by fitness (descending)
                sorted_indices = np.argsort(cluster_fitness)[::-1]
                top_individuals = [cluster_individuals[i] for i in sorted_indices[:len(cluster)//2]]

                # Elitism: keep best individuals
                new_population.extend(top_individuals)

                # Generate offspring through crossover and mutation for the rest
                while len(new_population) < population_size:
                    if len(top_individuals) >= 2:
                        # Tournament selection
                        parent1 = top_individuals[random.randint(0, len(top_individuals)-1)]
                        parent2 = top_individuals[random.randint(0, len(top_individuals)-1)]

                        # Crossover (uniform)
                        child = parent1.copy()
                        for i in range(len(parent1)):
                            if random.random() < 0.5:
                                child[i] = parent2[i]

                        # Mutation
                        for i in range(len(child)):
                            if random.random() < 0.1:
                                if i % 3 == 0:  # x coordinate
                                    child[i] = np.clip(child[i] + np.random.normal(0, 0.01), 0.05, 0.95)
                                elif i % 3 == 1:  # y coordinate
                                    child[i] = np.clip(child[i] + np.random.normal(0, 0.01), 0.05, 0.95)
                                else:  # radius
                                    child[i] = np.clip(child[i] + np.random.normal(0, 0.005), 0.005, 0.45)

                        # Ensure valid configuration
                        child = child.reshape(-1, 3)
                        if self.validate_placement(child):
                            new_population.append(child)
                        else:
                            # Repair if needed
                            repaired = self.repair_circles(child)
                            if self.validate_placement(repaired):
                                new_population.append(repaired)
                    else:
                        # If insufficient individuals, generate random
                        circles = self.generate_voronoi_initialization()
                        new_population.append(circles)

            # Trim to exact population size
            population = new_population[:population_size]

        return best_individual if best_individual is not None else population[0]

    def repair_circles(self, circles: np.ndarray) -> np.ndarray:
        """Repair invalid circle configurations."""
        repaired = circles.copy()
        n = len(repaired)

        # Repair each circle individually to satisfy constraints
        for i in range(n):
            x, y, r = repaired[i]

            # Fix containment constraints
            r = min(r, x, 1-x, y, 1-y)
            r = max(r, 0.005)  # Minimum radius

            # Adjust position to maintain containment
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)

            repaired[i] = [x, y, r]

        # Resolve overlaps through iterative adjustment
        for _ in range(100):  # Limited iterations to prevent infinite loop
            any_changed = False

            # Check all pairs of circles for overlap
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = repaired[i]
                    x2, y2, r2 = repaired[j]

                    distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_distance_sq = (r1 + r2)**2

                    if distance_sq < min_distance_sq:
                        # Circles overlap, resolve by moving them apart
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = np.sqrt(distance_sq)

                        if distance > 1e-10:  # Avoid division by zero
                            # Move circles apart along the vector connecting their centers
                            move_distance = (min_distance_sq**0.5 - distance) / 2.0

                            # Normalize direction vector
                            nx = dx / distance
                            ny = dy / distance

                            # Apply movement (split equally between both circles)
                            move_x = nx * move_distance
                            move_y = ny * move_distance

                            # Adjust positions
                            repaired[i][0] -= move_x * 0.5
                            repaired[i][1] -= move_y * 0.5
                            repaired[j][0] += move_x * 0.5
                            repaired[j][1] += move_y * 0.5

                            # Keep within bounds
                            repaired[i][0] = np.clip(repaired[i][0], repaired[i][2], 1-repaired[i][2])
                            repaired[i][1] = np.clip(repaired[i][1], repaired[i][2], 1-repaired[i][2])
                            repaired[j][0] = np.clip(repaired[j][0], repaired[j][2], 1-repaired[j][2])
                            repaired[j][1] = np.clip(repaired[j][1], repaired[j][2], 1-repaired[j][2])

                            any_changed = True

            if not any_changed:
                break

        return repaired

    def optimize(self) -> np.ndarray:
        """Main optimization pipeline with speciation."""
        # Try speciated evolution first
        try:
            circles = self.speciated_evolution(n_generations=50, population_size=50)

            # Apply final refinement
            circles = self.multi_scale_refinement(circles, max_iterations=100)
            circles = self.constraint_aware_local_search(circles, max_iterations=100)
            circles = self.improve_with_gradient_descent(circles, max_iter=50)

            # Final validation
            if not self.validate_placement(circles):
                circles = self.generate_greedy_fallback()

            return circles
        except Exception as e:
            # Fall back to regular approach
            pass

        # Regular optimization as fallback
        # Step 1: Generate initial configuration using Voronoi-inspired method
        circles = self.generate_voronoi_initialization()

        # Step 2: Multi-scale refinement using physical simulation
        circles = self.multi_scale_refinement(circles, max_iterations=200)

        # Step 3: Apply constraint-aware local search for fine-tuning
        circles = self.constraint_aware_local_search(circles, max_iterations=200)

        # Step 4: Additional gradient-based refinement to push for optimal radii
        circles = self.improve_with_gradient_descent(circles, max_iter=100)

        # Step 5: Validate result and apply fallback if needed
        if not self.validate_placement(circles):
            # Try greedy fallback
            circles = self.generate_greedy_fallback()

            # If still invalid, use Voronoi init
            if not self.validate_placement(circles):
                circles = self.generate_voronoi_initialization()

        return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a physics-inspired optimization approach combined with constraint-aware local search.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(n_circles=26, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END