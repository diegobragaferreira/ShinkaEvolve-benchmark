# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
import math
import time
import random

class HexagonalVoronoiEvolution:
    """Novel point dispersion optimizer combining hexagonal geometry, Voronoi analysis, and evolutionary techniques."""
    
    def __init__(self, n_points=16, dimensions=2, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _initialize_hexagonal_with_prime_breaking(self) -> np.ndarray:
        """Create hexagonal pattern with prime-based symmetry breaking."""
        # Mathematical constants for optimal hexagonal packing
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2
        col_spacing = 1.0

        points = []
        rows = 4
        cols = 4

        # Generate hexagonal lattice
        for i in range(rows):
            for j in range(cols):
                x = j * col_spacing + (i % 2) * 0.5
                y = i * row_spacing
                points.append([x, y])

        # Convert to numpy array and normalize
        points = np.array(points[:self.n_points])
        
        # Normalize properly to [0,1] bounds
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Apply prime-based symmetry breaking
        for i in range(self.n_points):
            # Use prime-related patterns for perturbations
            prime_factor = 1 + (i * 17) % 7  # Prime-inspired variation
            angle = i * 0.785398  # pi/4 increments
            noise_intensity = 0.005 + 0.003 * math.sin(prime_factor * 0.5)
            
            # Apply asymmetric perturbations
            noise_x = np.random.normal(0, noise_intensity, 1)[0] * (1 + math.sin(i * 3.14159))
            noise_y = np.random.normal(0, noise_intensity, 1)[0] * (1 + math.cos(i * 2.345))
            
            points[i] += [noise_x, noise_y]

        # Clip to ensure bounds
        points = np.clip(points, 0, 1)
        return points

    def _initialize_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral for good distribution."""
        points = np.zeros((self.n_points, self.dimensions))
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.n_points):
            # Map to sphere then to square using stereographic projection
            z = 1 - (i / (self.n_points - 1)) * 2
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)

            # Convert to Cartesian coordinates
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)

            # Project from sphere to square [0,1] x [0,1]
            points[i] = [(x + 1) / 2, (y + 1) / 2]

        # Add small perturbations for symmetry breaking
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _initialize_grid_pattern(self) -> np.ndarray:
        """Create structured grid pattern."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        
        # Add structured noise for symmetry breaking
        for i in range(self.n_points):
            noise_x = 0.002 * math.sin(i * 0.7) * math.cos(i * 0.3)
            noise_y = 0.002 * math.cos(i * 0.5) * math.sin(i * 0.9)
            points[i] += [noise_x, noise_y]
            
        points = np.clip(points, 0, 1)
        return points

    def _initialize_random_with_bias(self) -> np.ndarray:
        """Random initialization with bias towards inner region."""
        points = np.random.rand(self.n_points, self.dimensions) * 0.8 + 0.1
        return points

    def _calculate_voronoi_uniformity(self, points: np.ndarray) -> float:
        """Calculate uniformity of Voronoi cells."""
        try:
            vor = Voronoi(points)
            areas = []
            
            # Calculate areas of finite Voronoi regions
            for region in vor.regions:
                if not any(v == -1 for v in region) and len(region) >= 3:
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) >= 3:
                        # Shoelace formula for polygon area
                        area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] - 
                                           polygon[(i+1)%len(polygon)][0] * polygon[i][1] 
                                           for i in range(len(polygon))))
                        areas.append(area)
            
            if len(areas) == 0:
                return 0.0
                
            avg_area = np.mean(areas)
            expected_area = 1.0 / self.n_points  # Expected area per cell
            if expected_area > 0:
                uniformity = 1.0 / (1.0 + abs(avg_area - expected_area) / expected_area)
                return uniformity
            return 0.0
            
        except Exception:
            return 0.0

    def _calculate_distances(self, points: np.ndarray) -> tuple:
        """Calculate minimum and maximum distances efficiently."""
        if len(points) < 2:
            return 0, 0

        try:
            distances = pdist(points)
            
            if len(distances) == 0:
                return 0, 0

            min_distance = np.min(distances)
            max_distance = np.max(distances)

            return min_distance, max_distance
        except Exception:
            return 0, 0

    def _evaluate_fitness(self, points: np.ndarray) -> float:
        """Evaluate fitness combining distance ratio and Voronoi uniformity."""
        min_d, max_d = self._calculate_distances(points)
        
        if max_d <= 1e-12:
            return 0

        distance_ratio = min_d / max_d
        
        # Include Voronoi uniformity as additional factor
        voronoi_uniformity = self._calculate_voronoi_uniformity(points)
        
        # Weighted combination
        return distance_ratio * (1.0 + 0.5 * voronoi_uniformity)

    def _perturb_single(self, points: np.ndarray, idx: int, step_size: float = 0.005) -> np.ndarray:
        """Perturb a single point with boundary handling."""
        new_points = points.copy()
        delta = np.random.uniform(-step_size, step_size, self.dimensions)
        new_points[idx] = points[idx] + delta
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points

    def _perturb_neighborhood(self, points: np.ndarray, indices: list, step_size: float = 0.005) -> np.ndarray:
        """Perturb a group of points together while preserving structure."""
        new_points = points.copy()
        
        # Move all points relative to centroid to maintain local structure
        centroid = np.mean(points[indices], axis=0)
        move_vector = np.random.uniform(-step_size, step_size, self.dimensions)
        new_centroid = np.clip(centroid + move_vector, 0, 1)
        delta = new_centroid - centroid
        
        for idx in indices:
            new_points[idx] = points[idx] + delta
            new_points[idx] = np.clip(new_points[idx], 0, 1)

        return new_points

    def _adaptive_simulated_annealing(self, initial_points: np.ndarray, max_iterations: int = 5000) -> np.ndarray:
        """Enhanced adaptive simulated annealing with convergence monitoring."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_fitness = self._evaluate_fitness(current_points)

        # Adaptive cooling schedule
        temperature = 1.0
        cooling_rate = 0.9995
        stagnation_counter = 0
        previous_best = best_fitness

        # Track recent fitness improvements for early stopping
        recent_improvements = []
        max_recent = 50

        for iteration in range(max_iterations):
            # Determine perturbation strategy adaptively
            if np.random.random() < 0.7:  # Neighborhood moves
                # Vary neighborhood size based on iteration stage
                if iteration < 1000:
                    neighborhood_size = 2
                elif iteration < 3000:
                    neighborhood_size = np.random.randint(2, 4)
                else:
                    neighborhood_size = np.random.randint(2, min(5, self.n_points))

                indices = np.random.choice(self.n_points, neighborhood_size, replace=False).tolist()
                neighbor_points = self._perturb_neighborhood(current_points, indices, step_size=temperature * 0.05)
            else:  # Single point moves
                point_idx = np.random.randint(self.n_points)
                neighbor_points = self._perturb_single(current_points, point_idx, step_size=temperature * 0.05)

            # Evaluate neighbor solution
            neighbor_fitness = self._evaluate_fitness(neighbor_points)

            # Accept/reject based on Metropolis criterion
            if neighbor_fitness > best_fitness:
                current_points = neighbor_points
                best_points = neighbor_points
                best_fitness = neighbor_fitness
                stagnation_counter = 0
                recent_improvements.append(best_fitness)
            elif np.random.rand() < math.exp((neighbor_fitness - best_fitness) / temperature):
                current_points = neighbor_points
                stagnation_counter = 0
                recent_improvements.append(neighbor_fitness)
            else:
                stagnation_counter += 1

            # Adaptive cooling based on stagnation
            if stagnation_counter > 50:
                temperature *= 0.995
                stagnation_counter = 0
            else:
                # Phase-dependent cooling - slower initially
                phase_cooling = cooling_rate * (0.9 if iteration < 2000 else 1.0)
                temperature *= phase_cooling

            # Maintain recent improvements history
            if len(recent_improvements) > max_recent:
                recent_improvements.pop(0)

            # Convergence check
            if iteration % 100 == 0 and iteration > 0:
                current_fitness = self._evaluate_fitness(best_points)
                if abs(previous_best - current_fitness) < 1e-8:
                    break
                previous_best = current_fitness

            # Early termination for very slow progress
            if iteration > 1000 and temperature < 0.001:
                break

        return best_points

    def _gradient_refinement(self, points: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply gradient-based refinement to optimize final solution."""
        refined_points = points.copy()
        
        for iteration in range(max_iter):
            # Estimate gradient via finite differences
            eps = 1e-5
            gradient = np.zeros_like(refined_points)
            
            base_fitness = self._evaluate_fitness(refined_points)
            
            for i in range(len(refined_points)):
                for j in range(self.dimensions):
                    # Perturb point coordinate
                    points_plus = refined_points.copy()
                    points_plus[i, j] += eps
                    points_plus = np.clip(points_plus, 0, 1)
                    
                    points_minus = refined_points.copy()
                    points_minus[i, j] -= eps
                    points_minus = np.clip(points_minus, 0, 1)
                    
                    fitness_plus = self._evaluate_fitness(points_plus)
                    fitness_minus = self._evaluate_fitness(points_minus)
                    
                    gradient[i, j] = (fitness_plus - fitness_minus) / (2 * eps)
            
            # Apply update with adaptive learning rate
            learning_rate = 0.01 * (1.0 - iteration / max_iter)
            refined_points = refined_points + learning_rate * gradient
            refined_points = np.clip(refined_points, 0, 1)
            
            # Check for convergence
            if np.all(np.abs(gradient) < 1e-6):
                break
                
        return refined_points

    def optimize_multiple_starts(self, max_iterations: int = 5000) -> np.ndarray:
        """Run optimization from multiple diverse starting points."""
        # Multiple initialization strategies
        initial_configs = [
            self._initialize_hexagonal_with_prime_breaking(),
            self._initialize_fibonacci_spiral(), 
            self._initialize_grid_pattern(),
            self._initialize_random_with_bias()
        ]

        best_points = None
        best_fitness = -float('inf')

        for i, initial_config in enumerate(initial_configs):
            try:
                # Stage 1: Adaptive simulated annealing
                optimized_points = self._adaptive_simulated_annealing(initial_config, max_iterations)
                
                # Stage 2: Gradient-based refinement
                refined_points = self._gradient_refinement(optimized_points)
                
                # Evaluate final result
                final_fitness = self._evaluate_fitness(refined_points)
                
                if final_fitness > best_fitness:
                    best_fitness = final_fitness
                    best_points = refined_points.copy()

            except Exception as e:
                print(f"Warning: Optimization from start {i} failed: {e}")
                continue

        # Fallback to best initial configuration if nothing worked
        if best_points is None:
            # Return the configuration with highest initial fitness
            best_initial_fitness = -float('inf')
            for config in initial_configs:
                fitness = self._evaluate_fitness(config)
                if fitness > best_initial_fitness:
                    best_initial_fitness = fitness
                    best_points = config.copy()
                    
        return best_points if best_points is not None else initial_configs[0]


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Create optimizer instance
    optimizer = HexagonalVoronoiEvolution(n_points=16, dimensions=2, seed=42)

    # Run optimization with multiple starts
    start_time = time.time()
    optimized_points = optimizer.optimize_multiple_starts(max_iterations=5000)
    end_time = time.time()

    # Final validation
    final_fitness = optimizer._evaluate_fitness(optimized_points)
    
    return optimized_points

# EVOLVE-BLOCK-END