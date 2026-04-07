# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import math
import time
from typing import List, Tuple

class HexagonalSAOptimizer:
    """Optimizes point distribution using hybrid hexagonal + simulated annealing approach."""
    
    def __init__(self, n_points=16, dimensions=2, seed=42, max_time_seconds=180):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        np.random.seed(seed)

    def _compute_ratio(self, points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
            
        # Use efficient distance computation
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 1e-12:
            return 0.0
            
        return d_min / d_max

    def _generate_hexagonal_initialization(self):
        """Create hexagonal lattice with enhanced asymmetry."""
        points = []
        sqrt3 = np.sqrt(3)
        
        # Create 4x4 hexagonal grid
        spacing_x = 1.0 / 3.0
        spacing_y = sqrt3 / 4.0
        
        for i in range(4):
            for j in range(4):
                if len(points) >= self.n_points:
                    break
                    
                x = j * spacing_x
                y = i * spacing_y
                
                # Offset odd rows for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Add systematic asymmetry based on position
                position_factor = (i * 7 + j * 3) % 10
                noise_scale = 0.01 + position_factor * 0.001
                
                # Apply noise with directional bias
                x += np.random.normal(0, noise_scale * 0.5)
                y += np.random.normal(0, noise_scale * 0.5)
                
                # Directional bias for better distribution
                if i % 3 == 0:
                    x += np.random.normal(0, noise_scale * 0.1)
                if j % 3 == 0:
                    y += np.random.normal(0, noise_scale * 0.1)
                
                points.append([x, y])
        
        points = np.array(points[:self.n_points])
        
        # Normalize to [0,1] range
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        if x_max > x_min and x_max != x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min and y_max != y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
            
        # Ensure proper bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points

    def _generate_fibonacci_initialization(self):
        """Generate Fibonacci-inspired point distribution."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Generate points in Fibonacci spiral pattern
        for i in range(self.n_points):
            if i >= self.n_points:
                break
                
            # Normalize index to [0, 1]
            t = i / (self.n_points - 1) if self.n_points > 1 else 0.5
            
            # Use Fibonacci-like distribution
            radius = np.sqrt(t)
            angle = 2 * np.pi * i * phi
            
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            
            # Add noise for symmetry breaking
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            # Clip to bounds
            points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
            
        return np.array(points)

    def _generate_spiral_initialization(self):
        """Generate spiral-based initialization for even distribution."""
        points = []
        
        # Spiral pattern with radial distribution
        angle_step = 2 * np.pi / 16
        radius_scale = 0.4
        
        for i in range(self.n_points):
            if i >= self.n_points:
                break
                
            angle = i * angle_step
            radius = (i / (self.n_points - 1)) * radius_scale if self.n_points > 1 else 0.2
            
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Add noise for symmetry breaking
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            # Clip to bounds
            points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
            
        return np.array(points)

    def _generate_grid_initialization(self):
        """Generate structured grid with asymmetric perturbations."""
        points = []
        grid_size = 4
        
        # Create 4x4 grid
        x_coords = np.linspace(0.1, 0.9, grid_size)
        y_coords = np.linspace(0.1, 0.9, grid_size)
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                if len(points) >= self.n_points:
                    break
                    
                # Add positional noise
                noise_x = np.random.normal(0, 0.01)
                noise_y = np.random.normal(0, 0.01)
                
                # Different noise levels based on position
                if i % 2 == 0 and j % 2 == 0:
                    noise_x *= 1.5
                    noise_y *= 1.5
                    
                points.append([np.clip(x + noise_x, 0, 1), np.clip(y + noise_y, 0, 1)])
        
        return np.array(points[:self.n_points])

    def _generate_random_initialization(self):
        """Generate random point configuration."""
        return np.random.rand(self.n_points, self.dimensions)

    def generate_initial_points(self) -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        configurations = []
        
        # 1. Enhanced hexagonal initialization
        configurations.append(self._generate_hexagonal_initialization())
        
        # 2. Fibonacci-inspired initialization
        configurations.append(self._generate_fibonacci_initialization())
        
        # 3. Spiral initialization
        configurations.append(self._generate_spiral_initialization())
        
        # 4. Grid initialization
        configurations.append(self._generate_grid_initialization())
        
        # 5. Random initialization
        configurations.append(self._generate_random_initialization())
        
        # 6. Slightly perturbed hexagonal
        hex_points = self._generate_hexagonal_initialization()
        hex_points += np.random.normal(0, 0.02, hex_points.shape)
        hex_points[:, 0] = np.clip(hex_points[:, 0], 0, 1)
        hex_points[:, 1] = np.clip(hex_points[:, 1], 0, 1)
        configurations.append(hex_points)
        
        return configurations

    def _lbfgs_optimization(self, initial_points: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Perform L-BFGS optimization on the initial configuration."""
        # Flatten points for optimization
        flat_initial = initial_points.flatten()
        
        # Define bounds for each coordinate [0,1]
        bounds = [(0, 1) for _ in range(len(flat_initial))]
        
        # Optimize using L-BFGS-B with tight tolerances
        try:
            result = minimize(
                lambda x: -self._compute_ratio(x.reshape(-1, self.dimensions)),  # Negative because we maximize
                flat_initial,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                return optimized_points, self._compute_ratio(optimized_points)
            else:
                warnings.warn(f"L-BFGS-B optimization failed: {result.message}")
                return initial_points, self._compute_ratio(initial_points)
                
        except Exception as e:
            warnings.warn(f"L-BFGS-B optimization error: {str(e)}")
            return initial_points, self._compute_ratio(initial_points)

    def _adaptive_simulated_annealing(self, points: np.ndarray, max_iter: int = 5000) -> Tuple[np.ndarray, float]:
        """
        Adaptive Simulated Annealing with intelligent cooling schedule
        """
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Adaptive cooling parameters
        temp = 0.5  # Initial temperature
        cooling_rate = 0.9995
        min_temp = 1e-8
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 50
        
        # Progress tracking
        last_improvement_iter = 0
        stagnation_counter = 0
        
        for iteration in range(max_iter):
            # Create neighbor by perturbing one random point
            neighbor_points = current_points.copy()
            idx = np.random.randint(0, len(neighbor_points))
            
            # Adaptive step size based on iteration progress
            if iteration < max_iter // 3:
                step_size = 0.03
            elif iteration < 2 * max_iter // 3:
                step_size = 0.015
            else:
                step_size = 0.005
            
            # Apply perturbation
            neighbor_points[idx, 0] += np.random.normal(0, step_size)
            neighbor_points[idx, 1] += np.random.normal(0, step_size)
            
            # Boundary handling with clipping
            neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
            neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)
            
            # Calculate neighbor ratio
            neighbor_ratio = self._compute_ratio(neighbor_points)
            
            # Track improvements
            if neighbor_ratio > current_ratio:
                recent_improvements.append(1)
                stagnation_counter = 0
            else:
                recent_improvements.append(0)
                stagnation_counter += 1
            
            # Maintain window size
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)
            
            # Accept or reject the neighbor
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
                    last_improvement_iter = iteration
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                if delta < 0:
                    acceptance_prob = math.exp(delta / temp)
                    if np.random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio
            
            # Adaptive cooling schedule
            if iteration % 50 == 0 and len(recent_improvements) > 0:
                recent_avg = sum(recent_improvements) / len(recent_improvements)
                if recent_avg < 0.1:  # Low improvement rate
                    temp *= 0.99  # Cool faster
                elif recent_avg > 0.3:  # High improvement rate  
                    temp *= 1.005  # Warm up slightly
                else:
                    temp *= cooling_rate  # Normal cooling
            
            # Early stopping for stagnation
            if stagnation_counter > 500:
                break
                
            # Temperature bounds checking
            if temp < min_temp:
                break

        return best_points, best_ratio

    def optimize(self):
        """Main optimization routine with multi-start approach."""
        best_points = None
        best_ratio = -np.inf
        start_time = time.time()
        
        # Generate multiple initial configurations
        initial_configs = self.generate_initial_points()
        
        # Try all initial configurations
        for i, initial_config in enumerate(initial_configs):
            if time.time() - start_time > self.max_time_seconds - 5:
                break
                
            try:
                # Stage 1: L-BFGS optimization
                lbfgsb_points, lbfgsb_ratio = self._lbfgs_optimization(initial_config.copy())
                
                # Stage 2: Simulated Annealing refinement (if time allows and solution is reasonable)
                if (time.time() - start_time < self.max_time_seconds - 10 and 
                    lbfgsb_ratio > 0.05):  # Only refine if solution is decent
                    
                    sa_points, sa_ratio = self._adaptive_simulated_annealing(lbfgsb_points.copy())
                    final_points = sa_points if sa_ratio > lbfgsb_ratio else lbfgsb_points
                    final_ratio = sa_ratio if sa_ratio > lbfgsb_ratio else lbfgsb_ratio
                else:
                    final_points = lbfgsb_points
                    final_ratio = lbfgsb_ratio
                
                # Update best solution
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Error in optimization stage {i}: {str(e)}")
                continue
        
        # Final validation and fallback
        if best_points is not None:
            final_ratio = self._compute_ratio(best_points)
            print(f"Final optimized ratio: {final_ratio:.6f}")
            return best_points
        else:
            # Fallback to hexagonal initialization if all methods fail
            return self._generate_hexagonal_initialization()

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = HexagonalSAOptimizer(n_points=16, dimensions=2, seed=42, max_time_seconds=180)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END