# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
import random
from copy import deepcopy

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    class PointDispersionOptimizer:
        def __init__(self, n_points=16, max_time=175):
            self.n_points = n_points
            self.max_time = max_time
            self.best_ratio = -np.inf
            self.best_points = None
            self.start_time = time.time()
            
        def compute_min_max_ratio(self, points):
            """Compute the ratio of minimum to maximum pairwise distances"""
            if len(points) < 2:
                return 0.0
            
            distances = pdist(points)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 0.0
            return np.min(distances) / max_dist
            
        def is_time_exceeded(self):
            """Check if maximum time limit has been reached"""
            return time.time() - self.start_time > self.max_time
            
        def initialize_hexagonal_grid(self):
            """Create initial configuration using a precise hexagonal grid pattern with symmetry breaking"""
            # Create a proper hexagonal lattice pattern
            points = []
            
            # Parameters for hexagonal lattice
            # Using triangular packing where points form equilateral triangles
            spacing = 0.25  # Adjusted spacing for 16 points in unit square
            row_spacing = spacing * np.sqrt(3) / 2.0
            col_spacing = spacing
            
            # Generate a 4x4 hexagonal grid (16 points total)
            for i in range(4):
                for j in range(4):
                    if len(points) >= self.n_points:
                        break
                    # Calculate position in hexagonal lattice
                    x = j * col_spacing
                    # Offset odd rows for proper hexagonal packing
                    if i % 2 == 1:
                        x += col_spacing / 2.0
                    y = i * row_spacing
                    points.append([x, y])
            
            # Convert to numpy array
            points = np.array(points[:self.n_points])
            
            # Scale to fit within unit square [0,1] x [0,1] while preserving hexagonal properties
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            
            # Avoid division by zero
            if max_x <= min_x:
                max_x = min_x + 1.0
            if max_y <= min_y:
                max_y = min_y + 1.0
                
            # Scale factors
            scale_x = 1.0 / (max_x - min_x)
            scale_y = 1.0 / (max_y - min_y)
            scale = min(scale_x, scale_y) * 0.8  # Leave padding
            
            # Apply scaling and centering
            points[:, 0] = (points[:, 0] - min_x) * scale
            points[:, 1] = (points[:, 1] - min_y) * scale
            
            # Center in unit square
            center_shift_x = 0.5 - (np.max(points[:, 0]) + np.min(points[:, 0])) / 2.0
            center_shift_y = 0.5 - (np.max(points[:, 1]) + np.min(points[:, 1])) / 2.0
            
            points[:, 0] += center_shift_x
            points[:, 1] += center_shift_y
            
            # Ensure all points are within bounds
            points = np.clip(points, 0, 1)
            
            # Apply symmetric perturbations to break symmetries
            np.random.seed(42)
            noise_magnitude = 0.015
            
            # Add different noise patterns based on position for better symmetry breaking
            for i in range(len(points)):
                row = i // 4
                col = i % 4
                
                # Corner points get stronger perturbations
                if (row == 0 or row == 3) and (col == 0 or col == 3):
                    pert_mag = noise_magnitude * 1.5
                # Edge points get medium perturbations
                elif row == 0 or row == 3 or col == 0 or col == 3:
                    pert_mag = noise_magnitude * 1.0
                else:
                    pert_mag = noise_magnitude * 0.5
                
                # Add perturbation
                points[i] += np.random.normal(0, pert_mag, 2)
            
            # Clip to ensure bounds
            points = np.clip(points, 0, 1)
            
            return points
            
        def initialize_random(self):
            """Create random initial configuration with clustering avoidance"""
            np.random.seed(42)
            points = np.random.rand(self.n_points, 2)
            
            # Simple clustering avoidance by ensuring minimum distance
            for i in range(self.n_points):
                for j in range(i):
                    # Check distance to existing points
                    dist = np.linalg.norm(points[i] - points[j])
                    # If too close, move point away
                    if dist < 0.05:
                        # Move point away from cluster
                        direction = points[i] - points[j]
                        if np.linalg.norm(direction) > 1e-8:
                            direction = direction / np.linalg.norm(direction)
                            points[i] += direction * 0.02
                        
            return points
            
        def initialize_structured(self):
            """Create structured initialization with mix of approaches"""
            # Create hexagonal grid as base
            base_points = self.initialize_hexagonal_grid()
            
            # Apply more structured perturbation
            np.random.seed(43)
            perturbations = np.random.normal(0, 0.01, (self.n_points, 2))
            
            # Apply position-based perturbation magnitudes
            for i in range(self.n_points):
                row = i // 4
                col = i % 4
                # Make center points more sensitive to perturbations
                if 1 <= row <= 2 and 1 <= col <= 2:
                    perturbations[i] *= 1.5  # Stronger perturbations for center
                else:
                    perturbations[i] *= 0.7  # Weaker for edges
                
            points = np.clip(base_points + perturbations, 0, 1)
            return points
            
        def generate_neighbor(self, points, temperature):
            """Generate neighbor solution by coordinated perturbations"""
            new_points = deepcopy(points)
            
            # Select a few points to perturb together (coordinated moves)
            num_to_move = min(3, self.n_points // 4)  # Move 1-4 points at a time
            indices_to_move = random.sample(range(self.n_points), num_to_move)
            
            # Perturbation magnitude decreases with temperature
            pert_magnitude = temperature * 0.05
            if pert_magnitude < 1e-8:
                pert_magnitude = 1e-8
                
            # Apply coordinated perturbations
            for idx in indices_to_move:
                new_points[idx] += np.random.normal(0, pert_magnitude, 2)
            
            # Ensure all points stay within bounds
            for i in range(self.n_points):
                new_points[i] = np.clip(new_points[i], 0, 1)
                
            return new_points
            
        def adaptive_simulated_annealing(self, initial_points, max_iterations=8000):
            """Run adaptive simulated annealing optimization"""
            current_points = deepcopy(initial_points)
            current_ratio = self.compute_min_max_ratio(current_points)
            
            best_points = deepcopy(current_points)
            best_ratio = current_ratio
            
            # Adaptive cooling schedule
            temperature = 0.2
            cooling_rate = 0.9995
            min_temperature = 1e-6
            
            # Track convergence
            last_improvement = 0
            patience = 200
            
            iteration = 0
            while iteration < max_iterations and not self.is_time_exceeded():
                # Generate neighbor solution
                new_points = self.generate_neighbor(current_points, temperature)
                new_ratio = self.compute_min_max_ratio(new_points)
                
                # Accept or reject the new solution
                if new_ratio > current_ratio:
                    # Always accept better solutions
                    current_points = new_points
                    current_ratio = new_ratio
                    
                    # Update best solution if needed
                    if new_ratio > best_ratio:
                        best_points = deepcopy(new_points)
                        best_ratio = new_ratio
                        last_improvement = iteration
                else:
                    # Accept worse solutions with probability
                    delta = new_ratio - current_ratio
                    acceptance_prob = np.exp(delta / temperature)
                    if random.random() < acceptance_prob:
                        current_points = new_points
                        current_ratio = new_ratio
                
                # Adaptive cooling based on convergence
                if iteration - last_improvement > patience:
                    # Stagnation detected - slow down cooling slightly
                    cooling_rate = 0.9990
                else:
                    # Normal cooling
                    cooling_rate = 0.9995
                
                # Cool down
                temperature *= cooling_rate
                
                # Prevent temperature from becoming too small
                if temperature < min_temperature:
                    temperature = min_temperature
                    
                iteration += 1
                
                # Occasionally check for time limit
                if iteration % 100 == 0 and self.is_time_exceeded():
                    break
                    
            return best_points, best_ratio
            
        def optimize(self):
            """Main optimization routine with multiple restarts"""
            # Try several initialization strategies with varying approaches
            initial_strategies = [
                self.initialize_hexagonal_grid,
                self.initialize_random,
                self.initialize_structured
            ]
            
            # Multiple restarts with different strategies
            restart_count = 0
            max_restarts = 5
            
            while restart_count < max_restarts and not self.is_time_exceeded():
                # Select initialization strategy
                init_func = initial_strategies[restart_count % len(initial_strategies)]
                initial_points = init_func()
                
                # Run adaptive simulated annealing optimization
                optimized_points, ratio = self.adaptive_simulated_annealing(initial_points)
                
                # Update global best
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = optimized_points.copy()
                
                restart_count += 1
                
                # Early exit if we're very close to target
                if self.best_ratio > 0.27:
                    break
                    
            # Fallback if no optimization was successful
            if self.best_points is None:
                # Try final optimization with random initialization
                fallback_points = self.initialize_random()
                optimized_points, ratio = self.adaptive_simulated_annealing(fallback_points, max_iterations=2000)
                self.best_points = optimized_points
                self.best_ratio = ratio
                
            return self.best_points
    
    # Initialize and run optimization
    optimizer = PointDispersionOptimizer(max_time=175)
    result = optimizer.optimize()
    
    return result

# EVOLVE-BLOCK-END