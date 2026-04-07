# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math
import time

class HexagonalVoronoiEvolutionOptimizer:
    """Hybrid hexagonal-voronoi evolution optimizer for point dispersion optimization."""
    
    def __init__(self):
        self.best_points = None
        self.best_ratio = 0.0
        self.eval_time = 0.0
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        
    def compute_distance_matrix(self, points):
        """Compute pairwise distance matrix for given points."""
        return squareform(pdist(points))

    def calculate_min_max_ratio(self, distance_matrix):
        """Calculate the ratio of minimum to maximum distances."""
        # Exclude diagonal (distance to self)
        off_diagonal = distance_matrix[distance_matrix > 0]
        if len(off_diagonal) == 0:
            return 0.0
        d_min = np.min(off_diagonal)
        d_max = np.max(off_diagonal)
        return d_min / d_max if d_max > 0 else 0.0

    def fibonacci_symmetry_breaking(self, index, total_points):
        """Use Fibonacci sequence for controlled symmetry breaking."""
        if index < len(self.fibonacci_sequence):
            return self.fibonacci_sequence[index] % 1000 / 10000.0
        else:
            return (index % 1000) / 10000.0

    def initialize_hybrid_hexagonal_voronoi(self):
        """
        Create a hybrid hexagonal-voronoi initialization pattern.
        Combines mathematical precision with geometric intuition.
        """
        # Layer 1: Core hexagonal structure
        points = []
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2
        col_spacing = 1.0
        rows = 4
        cols = 4

        # Create base hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                x = j * col_spacing + (i % 2) * col_spacing / 2
                y = i * row_spacing
                points.append([x, y])

        points = np.array(points[:16])

        # Layer 2: Normalize and scale to unit square
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Scale and center properly
        scale_factor = 0.9
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        points[:, 0] = 0.05 + scale_factor * (points[:, 0] - center_x) + 0.5
        points[:, 1] = 0.05 + scale_factor * (points[:, 1] - center_y) + 0.5

        # Layer 3: Pattern-based Voronoi-inspired perturbations
        np.random.seed(42)
        for i in range(len(points)):
            # Fibonacci-based perturbation magnitude
            fib_factor = self.fibonacci_symmetry_breaking(i, 16)
            base_magnitude = 0.01 * (0.5 + 0.5 * fib_factor)
            
            # Apply controlled Voronoi-style perturbations
            perturbation_x = np.random.normal(0, base_magnitude * 0.7, 1)[0]
            perturbation_y = np.random.normal(0, base_magnitude * 0.7, 1)[0]
            
            # Add additional pattern-based offset
            pattern_offset_x = 0.005 * math.sin(i * 0.5) * fib_factor
            pattern_offset_y = 0.005 * math.cos(i * 0.5) * fib_factor
            
            points[i, 0] += perturbation_x + pattern_offset_x
            points[i, 1] += perturbation_y + pattern_offset_y

        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)

        # Layer 4: Adaptive boundary correction
        boundary_margin = 0.02
        for i in range(len(points)):
            if points[i, 0] < boundary_margin:
                points[i, 0] = boundary_margin + np.random.rand() * 0.01
            elif points[i, 0] > 1 - boundary_margin:
                points[i, 0] = 1 - boundary_margin - np.random.rand() * 0.01
                
            if points[i, 1] < boundary_margin:
                points[i, 1] = boundary_margin + np.random.rand() * 0.01
            elif points[i, 1] > 1 - boundary_margin:
                points[i, 1] = 1 - boundary_margin - np.random.rand() * 0.01

        return points

    def initialize_random_points(self):
        """Initialize points randomly with better distribution properties."""
        np.random.seed(42)
        return np.random.uniform(0, 1, (16, 2))

    def initialize_adaptive_grid(self):
        """Initialize using adaptive grid with structured spacing."""
        # Create more optimized grid with better point distribution
        points = []
        grid_size = 4
        
        # Create adaptive spacing based on golden ratio principles
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Apply golden ratio based spacing for better distribution
                x = (i + (j % 2) * 0.5) / (grid_size - 1) * 0.9 + 0.05
                y = j / (grid_size - 1) * 0.9 + 0.05
                
                # Add structured perturbations
                noise_x = 0.01 * math.sin(i * 0.785) * math.cos(j * 0.785)
                noise_y = 0.01 * math.cos(i * 0.785) * math.sin(j * 0.785)
                
                points.append([x + noise_x, y + noise_y])

        points = np.array(points[:16])
        
        # Normalize to ensure proper bounds
        points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
        points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
        
        return points

    def evaluate_solution_quality(self, points):
        """Quick assessment of solution quality."""
        try:
            dist_matrix = self.compute_distance_matrix(points)
            ratio = self.calculate_min_max_ratio(dist_matrix)
            return ratio
        except:
            return 0.0

    def adaptive_local_search(self, initial_points, max_iter=1000):
        """Adaptive local search with dynamic step sizing."""
        current_points = initial_points.copy()
        current_ratio = self.evaluate_solution_quality(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Dynamic cooling based on improvement rate
        temp = 0.1
        cooling_rate = 0.999
        min_temp = 1e-6
        
        progress_history = []
        max_history = 50
        
        for iteration in range(max_iter):
            # Adaptively adjust perturbation magnitude based on current state
            adaptive_step = temp
            
            # Select random point to perturb
            idx = np.random.randint(len(current_points))
            new_points = current_points.copy()
            
            # Apply perturbation with adaptive magnitude
            delta_x = np.random.normal(0, adaptive_step, 1)[0]
            delta_y = np.random.normal(0, adaptive_step, 1)[0]
            
            new_points[idx, 0] += delta_x
            new_points[idx, 1] += delta_y
            
            # Enforce bounds
            new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
            new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)
            
            # Boundary correction to prevent sticking
            if new_points[idx, 0] < 0.01:
                new_points[idx, 0] = 0.01 + np.random.rand() * 0.01
            elif new_points[idx, 0] > 0.99:
                new_points[idx, 0] = 0.99 - np.random.rand() * 0.01
                
            if new_points[idx, 1] < 0.01:
                new_points[idx, 1] = 0.01 + np.random.rand() * 0.01
            elif new_points[idx, 1] > 0.99:
                new_points[idx, 1] = 0.99 - np.random.rand() * 0.01

            # Evaluate new point
            new_ratio = self.evaluate_solution_quality(new_points)
            
            # Accept with Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < math.exp((new_ratio - current_ratio) / temp):
                current_points = new_points
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
            
            # Track progress
            progress_history.append(current_ratio)
            if len(progress_history) > max_history:
                progress_history.pop(0)
            
            # Update temperature
            temp = max(min_temp, temp * cooling_rate)
            
            # Early stopping based on stagnation
            if len(progress_history) >= 20:
                recent_change = abs(progress_history[-1] - progress_history[-20])
                if recent_change < 1e-8:
                    break

        return best_points

    def multi_phase_optimization(self, initial_points):
        """Multi-phase optimization with dynamic strategy switching."""
        # Phase 1: Global search with differential evolution
        def objective_function(points_flat):
            points = points_flat.reshape((16, 2))
            points = np.clip(points, 0, 1)

            try:
                distances = pdist(points)
                d_min = np.min(distances)
                d_max = np.max(distances)

                if d_max == 0:
                    return float('inf')

                return -(d_min / d_max)
            except Exception:
                return 1e6

        bounds = [(0, 1)] * 32
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            de_result = result.x.reshape((16, 2))
            de_result[:, 0] = np.clip(de_result[:, 0], 0, 1)
            de_result[:, 1] = np.clip(de_result[:, 1], 0, 1)
            phase1_points = de_result
        except:
            phase1_points = initial_points.copy()

        # Phase 2: Local refinement
        phase2_points = self.adaptive_local_search(phase1_points, max_iter=1000)
        
        # Phase 3: Enhanced refinement with multiple restarts
        best_final = phase2_points.copy()
        best_final_ratio = self.evaluate_solution_quality(best_final)
        
        for _ in range(3):  # Multiple restarts
            # Small random perturbation to start fresh
            restart_points = phase2_points.copy()
            for i in range(len(restart_points)):
                restart_points[i] += np.random.normal(0, 0.005, 2)
            
            restart_points[:, 0] = np.clip(restart_points[:, 0], 0, 1)
            restart_points[:, 1] = np.clip(restart_points[:, 1], 0, 1)
            
            refined = self.adaptive_local_search(restart_points, max_iter=500)
            refined_ratio = self.evaluate_solution_quality(refined)
            
            if refined_ratio > best_final_ratio:
                best_final_ratio = refined_ratio
                best_final = refined.copy()
        
        return best_final

    def find_optimal_configuration(self):
        """Find the optimal point configuration using hybrid approach."""
        # Strategy 1: Hybrid hexagonal-voronoi initialization
        strategy1_points = self.initialize_hybrid_hexagonal_voronoi()
        
        # Strategy 2: Adaptive grid initialization  
        strategy2_points = self.initialize_adaptive_grid()
        
        # Strategy 3: Random initialization
        strategy3_points = self.initialize_random_points()
        
        # Evaluate all strategies
        strategies = [
            ("hybrid_hexagonal", strategy1_points),
            ("adaptive_grid", strategy2_points), 
            ("random", strategy3_points)
        ]
        
        best_ratio = 0.0
        best_points = None
        
        for strategy_name, initial_points in strategies:
            try:
                # Apply multi-phase optimization
                optimized_points = self.multi_phase_optimization(initial_points)
                ratio = self.evaluate_solution_quality(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
            except Exception:
                continue

        # Fallback to the best of initial strategies if optimization failed
        if best_points is None:
            best_points = strategy1_points.copy()
            
        return best_points

    def run_optimization(self) -> np.ndarray:
        """Run the complete optimization process."""
        start_time = time.time()

        try:
            # Find optimal configuration
            best_points = self.find_optimal_configuration()

            # Final validation and quality check
            dist_matrix = self.compute_distance_matrix(best_points)
            final_ratio = self.calculate_min_max_ratio(dist_matrix)

            self.best_ratio = final_ratio
            self.eval_time = time.time() - start_time

            return best_points

        except Exception as e:
            # Fallback to simple random initialization
            np.random.seed(42)
            fallback_points = np.random.uniform(0, 1, (16, 2))
            self.eval_time = time.time() - start_time
            return fallback_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = HexagonalVoronoiEvolutionOptimizer()
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END