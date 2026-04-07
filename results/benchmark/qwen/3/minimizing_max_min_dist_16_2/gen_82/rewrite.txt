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
            return time.time() - self.start_time > self.max_time - 2  # Leave 2 seconds buffer
            
        def initialize_hexagonal_grid(self):
            """Create initial configuration using a refined hexagonal grid pattern"""
            points = []
            
            # Create a 4x4 hexagonal grid
            rows = 4
            cols = 4
            spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
            spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0

            for i in range(rows):
                for j in range(cols):
                    if len(points) < self.n_points:
                        x = j * spacing_x
                        y = i * spacing_y
                        # Add hexagonal offset to create true hexagonal pattern (alternating rows)
                        if i % 2 == 1:
                            x += spacing_x * 0.5
                        points.append([x, y])
            
            # Convert to numpy array
            points = np.array(points[:self.n_points])
            
            # Normalize to fit within unit square [0,1] x [0,1]
            if len(points) > 1:
                min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
                min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
                
                # Avoid division by zero
                if max_x > min_x and max_y > min_y:
                    scale_x = 1.0 / (max_x - min_x)
                    scale_y = 1.0 / (max_y - min_y)
                    scale = min(scale_x, scale_y, 1.0)
                    
                    points[:, 0] = (points[:, 0] - min_x) * scale
                    points[:, 1] = (points[:, 1] - min_y) * scale
            
            # Center the points in the unit square
            center_shift = 0.5 - np.mean(points, axis=0)
            points = points + center_shift
            
            # Ensure points are within bounds
            points = np.clip(points, 0, 1)
            
            # Apply systematic symmetry breaking with deterministic rotations
            np.random.seed(42)
            for i in range(len(points)):
                # Apply deterministic rotation to break symmetry
                if i % 3 == 0:  # Every 3rd point gets a rotation
                    angle = np.pi / 24  # 7.5 degrees
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    center = np.mean(points, axis=0)
                    rotated_point = rotation_matrix @ (points[i] - center) + center
                    points[i] = np.clip(rotated_point, 0, 1)
            
            # Add small Gaussian noise to further break symmetries
            noise_magnitude = 0.005
            noise = np.random.normal(0, noise_magnitude, points.shape)
            points += noise
            points = np.clip(points, 0, 1)
            
            return points
            
        def initialize_random(self):
            """Create random initial configuration"""
            np.random.seed(42)
            points = np.random.rand(self.n_points, 2)
            return points
            
        def initialize_structured(self):
            """Create structured initialization combining multiple approaches"""
            # Mix of different structured approaches
            configs = []
            
            # Hexagonal grid
            configs.append(self.initialize_hexagonal_grid())
            
            # Random with clustering avoidance
            configs.append(self.initialize_random())
            
            # Perturbed grid
            grid_points = self.initialize_hexagonal_grid()
            np.random.seed(43)
            perturbations = np.random.normal(0, 0.01, (self.n_points, 2))
            configs.append(np.clip(grid_points + perturbations, 0, 1))
            
            # Return a mixed configuration
            return configs[random.randint(0, len(configs)-1)]
            
        def generate_neighbor(self, points, temperature):
            """Generate neighbor solution by perturbing one point"""
            new_points = deepcopy(points)
            # Choose random point to perturb
            idx = random.randint(0, self.n_points - 1)
            
            # Perturbation magnitude decreases with temperature
            pert_magnitude = min(temperature * 0.05, 0.05)  # Cap at 0.05
            if pert_magnitude < 1e-8:
                pert_magnitude = 1e-8
                
            # Add perturbation
            new_points[idx] += np.random.normal(0, pert_magnitude, 2)
            
            # Ensure point stays within bounds
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
            return new_points
            
        def simulated_annealing(self, initial_points, max_iterations=3000):
            """Run simulated annealing optimization with adaptive cooling"""
            current_points = deepcopy(initial_points)
            current_ratio = self.compute_min_max_ratio(current_points)
            
            best_points = deepcopy(current_points)
            best_ratio = current_ratio
            
            # Initial temperature and cooling schedule
            temperature = 0.1
            cooling_rate = 0.9995
            min_temperature = 1e-8
            improvement_count = 0
            last_improvement_iteration = 0
            
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
                        improvement_count += 1
                        last_improvement_iteration = iteration
                        
                        # If we have made significant improvements recently, 
                        # reduce cooling rate to fine-tune
                        if improvement_count > 3 and iteration - last_improvement_iteration < 500:
                            cooling_rate = max(cooling_rate, 0.9998)
                else:
                    # Accept worse solutions with probability
                    delta = new_ratio - current_ratio
                    acceptance_prob = np.exp(delta / temperature)
                    if random.random() < acceptance_prob:
                        current_points = new_points
                        current_ratio = new_ratio
                
                # Cool down
                temperature *= cooling_rate
                
                # Prevent temperature from becoming too small
                if temperature < min_temperature:
                    temperature = min_temperature
                    
                iteration += 1
                
                # Occasionally check for time limit
                if iteration % 100 == 0 and self.is_time_exceeded():
                    break
                    
                # Adaptive stopping if no improvement for a long time
                if iteration - last_improvement_iteration > 1000:
                    break
                    
            return best_points, best_ratio
            
        def optimize(self):
            """Main optimization routine with multiple restarts"""
            # Try several initialization strategies
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
                
                # Run simulated annealing optimization
                optimized_points, ratio = self.simulated_annealing(initial_points)
                
                # Update global best
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = optimized_points.copy()
                
                restart_count += 1
                
                # Early exit if we're close to target
                if self.best_ratio > 0.275:  # Beat the benchmark by margin
                    break
                    
            # Fallback if no optimization was successful
            if self.best_points is None:
                # Try final optimization with random initialization
                fallback_points = self.initialize_random()
                optimized_points, ratio = self.simulated_annealing(fallback_points, max_iterations=1000)
                self.best_points = optimized_points
                self.best_ratio = ratio
                
            return self.best_points
    
    # Initialize and run optimization
    optimizer = PointDispersionOptimizer(max_time=175)
    result = optimizer.optimize()
    
    return result

# EVOLVE-BLOCK-END