# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math
from typing import Tuple, List, Optional
import random

class AdaptiveCirclePackingOptimizer:
    def __init__(self, n_circles: int = 32):
        self.n_circles = n_circles
        self.best_solution = None
        self.best_sum_radii = -float('inf')
        
    def _initialize_hexagonal_grid(self) -> np.ndarray:
        """Initialize circle positions using a hexagonal grid pattern with adaptive sizing."""
        # Find dimensions of hexagonal grid
        cols = int(math.ceil(math.sqrt(self.n_circles)))
        rows = int(math.ceil(self.n_circles / cols))
        
        # Create a hexagonal grid
        positions = []
        for i in range(rows):
            for j in range(cols):
                # Offset every other row
                x_offset = j + 0.5 * (i % 2)
                y_offset = i * math.sqrt(3) / 2
                positions.append([x_offset, y_offset])

        if len(positions) > self.n_circles:
            positions = positions[:self.n_circles]

        positions = np.array(positions)
        
        if len(positions) == 0:
            return np.zeros((self.n_circles, 2))
            
        # Normalize to fit in [0.05, 0.95] x [0.05, 0.95] to leave margin
        min_x, min_y = positions.min(axis=0)
        max_x, max_y = positions.max(axis=0)
        
        if max_x - min_x > 0 and max_y - min_y > 0:
            scale_x = 0.9 / (max_x - min_x)
            scale_y = 0.9 / (max_y - min_y)
            positions[:, 0] = (positions[:, 0] - min_x) * scale_x + 0.05
            positions[:, 1] = (positions[:, 1] - min_y) * scale_y + 0.05
        
        return positions

    def _initialize_random_density_aware(self) -> np.ndarray:
        """Initialize positions using random placement with density awareness."""
        positions = []
        max_attempts = 10000
        attempts = 0
        
        while len(positions) < self.n_circles and attempts < max_attempts:
            # Generate random position
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Check if it's far enough from existing positions
            min_dist = float('inf')
            for px, py in positions:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                min_dist = min(min_dist, dist)
            
            # If sufficiently far from all existing circles, add it
            if min_dist > 0.05:  # Minimum distance threshold
                positions.append([x, y])
            
            attempts += 1
        
        # If we couldn't fill all positions, fill with hexagonal grid
        if len(positions) < self.n_circles:
            hex_positions = self._initialize_hexagonal_grid()
            for i in range(self.n_circles - len(positions)):
                if i < len(hex_positions):
                    positions.append([hex_positions[i][0], hex_positions[i][1]])
        
        return np.array(positions[:self.n_circles])

    def _initialize_grid_varied_spacing(self) -> np.ndarray:
        """Initialize positions using a grid with varied spacing for better distribution."""
        # Create grid with variable spacing to avoid regular patterns
        n_cols = int(np.ceil(np.sqrt(self.n_circles)))
        n_rows = int(np.ceil(self.n_circles / n_cols))
        
        positions = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(positions) >= self.n_circles:
                    break
                    
                # Add slight randomness to spacing
                x = 0.1 + (j + 0.5 + np.random.uniform(-0.1, 0.1)) * (0.8 / n_cols)
                y = 0.1 + (i + 0.5 + np.random.uniform(-0.1, 0.1)) * (0.8 / n_rows)
                
                # Ensure within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                positions.append([x, y])
        
        return np.array(positions[:self.n_circles])

    def _compute_initial_radii(self, positions: np.ndarray) -> np.ndarray:
        """Compute initial radii for circles based on available space and density."""
        radii = np.zeros(self.n_circles)
        
        for i, pos in enumerate(positions):
            x, y = pos
            # Initial estimate of radius based on distance to nearest boundary
            r_boundary = min(x, 1-x, y, 1-y)
            
            if r_boundary <= 0.001:
                radii[i] = 0.001
                continue
                
            # Check overlap with existing circles
            min_dist = float('inf')
            for j in range(i):
                circ_x, circ_y, circ_r = positions[j][0], positions[j][1], radii[j]
                dist = np.sqrt((x - circ_x)**2 + (y - circ_y)**2)
                min_dist = min(min_dist, dist)
            
            # Maximum radius is limited by both boundary and existing circles
            if min_dist != float('inf'):
                r = min(r_boundary, min_dist/2 - 0.001)
            else:
                r = r_boundary
                
            radii[i] = max(0.001, min(0.5, r))
            
        return radii

    def _adaptive_penalty(self, distance: float, min_distance: float) -> float:
        """Apply adaptive penalty for constraint violations based on violation severity."""
        if distance >= min_distance:
            return 0.0
        else:
            # Exponential penalty with adaptive scaling based on violation severity
            violation = min_distance - distance
            # Scale penalty based on how severe the violation is
            penalty_weight = 2000.0 * (1.0 + violation * 5.0)
            return penalty_weight * (1.0 - np.exp(-violation * 20.0))

    def _evaluate_fitness(self, circles_flat: np.ndarray) -> float:
        """Evaluate the fitness of a circle configuration with adaptive penalties."""
        # Reshape flat array back into circles
        circles = circles_flat.reshape((self.n_circles, 3))
        
        # Calculate sum of radii (this is what we want to maximize)
        total_radius = np.sum(circles[:, 2])
        
        penalty = 0.0
        
        # Boundary penalties using adaptive exponential function
        for i in range(self.n_circles):
            x, y, r = circles[i]
            # Penalties for boundary violations using adaptive exponential
            if x - r < 0:
                penalty += self._adaptive_penalty(x - r, 0.0)
            if x + r > 1:
                penalty += self._adaptive_penalty(-(x + r - 1), 0.0)
            if y - r < 0:
                penalty += self._adaptive_penalty(y - r, 0.0)
            if y + r > 1:
                penalty += self._adaptive_penalty(-(y + r - 1), 0.0)
        
        # Overlap penalties using adaptive exponential function with spatial indexing for efficiency
        if len(circles) > 1:
            # Build k-d tree for fast neighbor search
            positions = circles[:, :2]
            tree = cKDTree(positions)
            
            # Search for nearby points to reduce pairwise comparisons
            for i in range(self.n_circles):
                x1, y1, r1 = circles[i]
                
                # Query nearby circles (within 4*(r1 + r_max) distance)
                r_max = np.max(circles[:, 2]) if len(circles) > 0 else 0.5
                query_radius = 4 * (r1 + r_max)
                
                neighbors = tree.query_ball_point([x1, y1], query_radius)
                
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        # Adaptive penalty for overlap
                        penalty += self._adaptive_penalty(distance, r1 + r2)
        
        # Return negative because we minimize in scipy but want to maximize radius sum
        return -(total_radius - penalty)

    def _optimize_stage(self, initial_solution: np.ndarray, 
                       bounds: List[Tuple[float, float]], 
                       max_iter: int = 1500,
                       patience: int = 50) -> Optional[np.ndarray]:
        """Perform optimization stage with specified parameters and early stopping."""
        try:
            result = minimize(
                self._evaluate_fitness, 
                initial_solution, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                final_fitness = self._evaluate_fitness(result.x)
                sum_radii = -final_fitness
                
                if sum_radii > self.best_sum_radii:
                    self.best_sum_radii = sum_radii
                    self.best_solution = result.x.copy()
                
                return result.x
        except Exception as e:
            print(f"Optimization stage failed: {e}")
            return None
            
        return None

    def _generate_initialization(self, method: str, seed: int = 0) -> np.ndarray:
        """Generate a single initial configuration using specified method."""
        np.random.seed(seed)
        random.seed(seed)
        
        if method == "hexagonal":
            positions = self._initialize_hexagonal_grid()
        elif method == "random_density":
            positions = self._initialize_random_density_aware()
        elif method == "grid_varied":
            positions = self._initialize_grid_varied_spacing()
        else:
            positions = self._initialize_hexagonal_grid()
        
        # Add more pronounced random noise
        noise_scale = 0.08
        positions += np.random.uniform(-noise_scale, noise_scale, positions.shape)
        
        # Ensure positions stay within bounds
        positions = np.clip(positions, 0.05, 0.95)
        
        # Compute initial radii
        radii = self._compute_initial_radii(positions)
        
        # Combine into flat array
        return np.column_stack([positions, radii]).flatten()

    def optimize(self) -> np.ndarray:
        """Main optimization routine with enhanced strategies."""
        # Phase 1: Multi-strategy initialization with coarse optimization
        print("Starting multi-strategy initialization...")
        
        # Different initialization methods
        initialization_methods = ["hexagonal", "random_density", "grid_varied"]
        initial_solutions = []
        
        # Generate multiple initializations with different methods
        for i, method in enumerate(initialization_methods):
            sol = self._generate_initialization(method, seed=i)
            initial_solutions.append(sol)
        
        # Run coarse optimization for each initialization
        coarse_bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)] * self.n_circles
        
        print("Running coarse optimization phase...")
        for i, init_sol in enumerate(initial_solutions):
            print(f"Coarse optimization attempt {i+1}")
            self._optimize_stage(init_sol, coarse_bounds, max_iter=500)
        
        # Phase 2: Fine optimization with better refinement
        print("Starting fine optimization phase...")
        fine_bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)] * self.n_circles
        
        # Run additional optimization phases with different initializations
        additional_solutions = []
        for i in range(3):
            sol = self._generate_initialization("hexagonal", seed=10+i)
            additional_solutions.append(sol)
        
        for i, init_sol in enumerate(additional_solutions):
            print(f"Fine optimization attempt {i+1}")
            self._optimize_stage(init_sol, fine_bounds, max_iter=1500)
        
        # Return best solution found
        if self.best_solution is not None:
            return self.best_solution.reshape((self.n_circles, 3))
        else:
            # Final fallback
            fallback_positions = self._initialize_hexagonal_grid()
            fallback_radii = np.full(self.n_circles, 0.02)
            return np.column_stack([fallback_positions, fallback_radii])

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = AdaptiveCirclePackingOptimizer(32)
    return optimizer.optimize()

# EVOLVE-BLOCK-END