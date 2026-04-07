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
            """Create initial configuration using a hexagonal grid pattern with symmetry breaking"""
            points = []
            rows = 4
            cols = 4
            spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
            spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0

            for i in range(rows):
                for j in range(cols):
                    if len(points) < self.n_points:
                        x = j * spacing_x
                        y = i * spacing_y
                        # Add hexagonal offset to create true hexagonal pattern
                        if i % 2 == 1:
                            x += spacing_x * 0.5
                        points.append([x, y])
            
            # Apply systematic symmetry breaking
            points = np.array(points[:self.n_points])
            
            # Add deterministic perturbations to break symmetries
            np.random.seed(42)
            noise_magnitude = 0.02
            
            # Apply different perturbations based on position
            for i in range(len(points)):
                # Add position-dependent perturbations
                row = i // 4
                col = i % 4
                
                # Different noise magnitude for corners, edges, center
                if (row == 0 or row == 3) and (col == 0 or col == 3):
                    pert_mag = noise_magnitude * 0.5  # Less noise for corners
                elif row == 0 or row == 3 or col == 0 or col == 3:
                    pert_mag = noise_magnitude * 0.75  # Medium noise for edges
                else:
                    pert_mag = noise_magnitude  # Full noise for center
                
                # Add noise
                points[i] += np.random.normal(0, pert_mag, 2)
            
            # Ensure all points are within bounds
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
            perturbations = np.random.normal(0, 0.03, (self.n_points, 2))
            configs.append(np.clip(grid_points + perturbations, 0, 1))
            
            # Return a mixed configuration
            return configs[random.randint(0, len(configs)-1)]
            
        def generate_neighbor(self, points, temperature):
            """Generate neighbor solution by perturbing one point"""
            new_points = deepcopy(points)
            # Choose random point to perturb
            idx = random.randint(0, self.n_points - 1)
            
            # Perturbation magnitude decreases with temperature
            pert_magnitude = temperature * 0.1
            if pert_magnitude < 1e-8:
                pert_magnitude = 1e-8
                
            # Add perturbation
            new_points[idx] += np.random.normal(0, pert_magnitude, 2)
            
            # Ensure point stays within bounds
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
            return new_points
            
        def simulated_annealing(self, initial_points, max_iterations=5000):
            """Run simulated annealing optimization"""
            current_points = deepcopy(initial_points)
            current_ratio = self.compute_min_max_ratio(current_points)
            
            best_points = deepcopy(current_points)
            best_ratio = current_ratio
            
            # Initial temperature and cooling schedule
            temperature = 0.1
            cooling_rate = 0.9995
            min_temperature = 1e-8
            
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
                
                # Early exit if we're very close to target
                if self.best_ratio > 0.27:
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