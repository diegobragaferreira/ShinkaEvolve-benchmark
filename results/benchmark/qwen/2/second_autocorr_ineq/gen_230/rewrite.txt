# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
import random
import time
from typing import List, Tuple
import math

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# JIT compile the core computation functions for speed
@numba.jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """Fast autoconvolution computation using Numba"""
    n = len(f_vals)
    g = np.zeros(2 * n - 1)
    
    # Manual convolution for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@numba.jit(nopython=True)
def compute_norms_piecewise(g_vals):
    """Compute norms using piecewise linear integration matching evaluator's method"""
    n = len(g_vals)
    
    if n <= 1:
        return 0.0, 0.0, 0.0
    
    # Compute L2 norm squared using trapezoidal-like integration
    # Formula: (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    norm_2_sq = 0.0
    dx = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.5
    
    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)
    
    # Compute L1 norm (sum of absolute values)
    norm_1 = 0.0
    for i in range(n):
        norm_1 += abs(g_vals[i])
    
    # Compute L-infinity norm (maximum absolute value)
    norm_inf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > norm_inf:
            norm_inf = abs_val
    
    return norm_2_sq, norm_1, norm_inf

class FunctionEvaluator:
    """Encapsulates all function evaluation logic for C2 computation."""
    
    @staticmethod
    def compute_autoconvolution_norms(f: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation using efficient piecewise integration.
        Returns (||g||₂², ||g||₁, ||g||∞)
        """
        # Convert to numpy array
        f_arr = np.array(f, dtype=np.float64)
        
        # Compute autoconvolution
        g = compute_autoconvolution_fast(f_arr)
        
        # Compute norms using piecewise integration
        norm_2_sq, norm_1, norm_inf = compute_norms_piecewise(g)
        
        return norm_2_sq, norm_1, norm_inf

    @staticmethod
    def compute_c2(f: List[float]) -> float:
        """Compute C2 value for given function"""
        norm_2_sq, norm_1, norm_inf = FunctionEvaluator.compute_autoconvolution_norms(f)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
        
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2

class StepFunctionGenerator:
    """Handles generation and initialization of step functions."""
    
    @staticmethod
    def create_structured_step_function(n_steps: int) -> List[float]:
        """Create a structured step function with Gaussian peaks and step patterns"""
        # Create base function with multiple Gaussian peaks
        f_vals = np.zeros(n_steps)
        
        # Add multiple Gaussian peaks with logarithmic spacing
        n_peaks = max(5, min(25, n_steps // 40))
        
        # Generate log-spaced peak positions across domain [-0.25, 0.25]
        positions = []
        for i in range(n_peaks):
            # Use log-uniform distribution to avoid clustering
            if i == 0:
                pos = -0.25 + random.uniform(0.01, 0.05)  # Near left edge
            elif i == n_peaks - 1:
                pos = 0.25 - random.uniform(0.01, 0.05)   # Near right edge
            else:
                # Logarithmic distribution in the middle
                log_min = np.log(0.02)
                log_max = np.log(0.2)
                log_pos = np.random.uniform(log_min, log_max)
                pos = np.exp(log_pos) * random.choice([-1, 1])  # Alternate sides
                pos = np.clip(pos, -0.24, 0.24)  # Keep within bounds
                
            positions.append(pos)
        
        # Sort positions
        positions.sort()
        
        # Ensure minimum separation between peaks to prevent narrow autoconvolution interference
        min_separation = max(20, n_steps // 50)
        adjusted_positions = []
        for i, pos in enumerate(positions):
            if i == 0:
                adjusted_positions.append(pos)
            else:
                # Ensure minimum gap from previous peak
                prev_pos = adjusted_positions[-1]
                if abs(pos - prev_pos) < min_separation * 0.1:
                    # Adjust to maintain gap
                    adjusted_positions.append(prev_pos + min_separation * 0.1 * np.sign(pos - prev_pos))
                else:
                    adjusted_positions.append(pos)
        
        # Generate peak parameters
        for i, center_pos in enumerate(adjusted_positions):
            # Convert position to step index
            step_index = int((center_pos + 0.25) / (0.5 / n_steps))
            step_index = max(0, min(n_steps - 1, step_index))
            
            # Adaptive width and height based on position and peak importance
            # Peaks near center get narrower and higher amplitude for sharper autoconvolution
            # Outer peaks get broader and more moderate to avoid boundary artifacts
            if i == 0 or i == len(adjusted_positions) - 1:
                # Boundary peaks: broader and lower
                width = max(15, min(60, n_steps // 8))
                height = random.uniform(1.2, 2.0)
            else:
                # Inner peaks: narrower and higher  
                width = max(10, min(40, n_steps // 15))
                height = random.uniform(0.8, 1.5)
                
            # Scale height to avoid extremely dominant peaks that hurt C2 ratio
            height *= min(1.0, 100.0 / (width * (i + 1) + 20.0))
            
            # Create Gaussian-like peak centered at step_index
            x = np.arange(n_steps)
            gaussian = height * np.exp(-0.5 * ((x - step_index) / width) ** 2)
            f_vals += gaussian
        
        # Apply smoothing to reduce extreme variations
        if n_steps > 50:
            window_size = min(51, n_steps // 5)
            if window_size % 2 == 0:
                window_size -= 1
            if window_size > 1:
                f_vals = signal.savgol_filter(f_vals, window_size, 3)
        
        # Ensure non-negativity and normalize
        f_vals = np.maximum(f_vals, 0)
        if np.max(f_vals) > 0:
            f_vals = f_vals / np.max(f_vals) * 2.0
        
        # Apply constraint-aware normalization to prevent autoconvolution spikes
        _, _, norm_inf = FunctionEvaluator.compute_autoconvolution_norms(f_vals.tolist())
        if norm_inf > 0:
            # Cap extreme values to keep autoconvolution manageable
            max_allowed = np.percentile(f_vals, 90) if len(f_vals) > 10 else 1.0
            if max_allowed > 0:
                f_vals = np.minimum(f_vals, max_allowed * 3.0)
        
        return f_vals.tolist()

    @staticmethod
    def create_simple_step_function(n_steps: int) -> List[float]:
        """Create a simple step function with random heights"""
        # Create step function with varying heights
        heights = []
        n_steps_per_region = max(1, n_steps // 20)
        
        for i in range(min(20, n_steps // n_steps_per_region)):
            region_height = random.uniform(0.5, 2.0)
            for _ in range(n_steps_per_region):
                if len(heights) < n_steps:
                    heights.append(region_height)
        
        # Pad or truncate to exact length
        if len(heights) < n_steps:
            heights.extend([random.uniform(0.5, 2.0)] * (n_steps - len(heights)))
        elif len(heights) > n_steps:
            heights = heights[:n_steps]
        
        return heights

    @classmethod
    def adaptive_step_function_initialization(cls, n_steps: int) -> List[float]:
        """
        Create initial step function with adaptive construction using multiple strategies
        """
        # Use different initialization strategies based on problem size
        if n_steps < 200:
            # For small functions, use simple approach
            return cls.create_simple_step_function(n_steps)
        else:
            # For larger functions, use structured approach
            return cls.create_structured_step_function(n_steps)

class OptimizerPipeline:
    """Main optimization pipeline orchestrating the evolutionary process."""
    
    def __init__(self, max_time_seconds: int = 85):
        self.max_time_seconds = max_time_seconds
        self.evaluator = FunctionEvaluator()
        self.generator = StepFunctionGenerator()
        
    def evaluate_candidate(self, individual: List[float]) -> float:
        """Evaluate a single candidate function"""
        return self.evaluator.compute_c2(individual)
    
    def local_search_refinement(self, initial_f: List[float], max_iter: int = 30) -> List[float]:
        """
        Apply local search to improve the function using multiplicative perturbations
        """
        f_current = np.array(initial_f, dtype=np.float64)
        best_c2 = self.evaluator.compute_c2(f_current.tolist())
        best_f = f_current.copy()
        
        # Simple local search with small multiplicative perturbations
        for iteration in range(max_iter):
            # Create neighbor by making small changes
            f_new = f_current.copy()
            
            # Choose random indices to modify
            indices_to_modify = np.random.choice(
                len(f_new), 
                size=max(1, min(len(f_new) // 10, 50)), 
                replace=False
            )
            
            for idx in indices_to_modify:
                # Small random multiplicative perturbation
                if f_new[idx] > 0:
                    factor = np.random.uniform(0.95, 1.05)
                    f_new[idx] = max(0, f_new[idx] * factor)
                else:
                    # For zero values, add small random amount
                    f_new[idx] = max(0, f_new[idx] + np.random.uniform(-0.1, 0.1))
            
            # Evaluate new function
            new_c2 = self.evaluator.compute_c2(f_new.tolist())
            
            # Accept improvement
            if new_c2 > best_c2:
                best_c2 = new_c2
                best_f = f_new.copy()
                
            f_current = f_new
        
        return best_f.tolist()

    def simulated_annealing_refinement(self, initial_f: List[float], max_iter: int = 300) -> List[float]:
        """
        Apply simulated annealing for more thorough refinement
        """
        try:
            current_solution = np.array(initial_f, dtype=np.float64)
            current_score = self.evaluator.compute_c2(current_solution.tolist())
            
            best_solution = current_solution.copy()
            best_score = current_score
            
            temperature = 1.0
            
            # Track recent improvements for early stopping
            recent_improvements = []
            
            for iteration in range(max_iter):
                # Create neighbor by perturbing some values
                neighbor = current_solution.copy()
                
                # Select which elements to modify
                n_modify = max(1, min(len(neighbor) // 20, 20))
                indices_to_modify = np.random.choice(
                    len(neighbor), 
                    size=n_modify, 
                    replace=False
                )
                
                for idx in indices_to_modify:
                    # Small random perturbation with adaptive scaling
                    if random.random() < 0.7:
                        # Multiplicative perturbation
                        if neighbor[idx] > 0:
                            factor = random.uniform(0.9, 1.1)
                            neighbor[idx] = max(0, neighbor[idx] * factor)
                        else:
                            neighbor[idx] = max(0, neighbor[idx] + random.uniform(-0.1, 0.1))
                    else:
                        # Additive perturbation
                        delta = random.gauss(0, 0.05 * max(1, neighbor[idx]))
                        neighbor[idx] = max(0, neighbor[idx] + delta)
                
                # Evaluate neighbor
                neighbor_score = self.evaluator.compute_c2(neighbor.tolist())
                
                # Accept or reject based on simulated annealing criteria
                if neighbor_score > current_score:
                    current_solution = neighbor
                    current_score = neighbor_score
                    recent_improvements.append(neighbor_score)
                else:
                    # Accept with probability based on temperature
                    if temperature > 1e-10:
                        prob_accept = math.exp((neighbor_score - current_score) / temperature)
                        if random.random() < prob_accept:
                            current_solution = neighbor
                            current_score = neighbor_score
                            recent_improvements.append(neighbor_score)
                
                # Update best if improved
                if current_score > best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
                
                # Cool down
                temperature *= 0.995
                
                # Early stopping if no recent improvement
                if len(recent_improvements) > 20:
                    recent_avg = sum(recent_improvements[-20:]) / 20
                    if recent_avg < best_score * 0.9999:
                        recent_improvements.clear()
            
            return best_solution.tolist()
            
        except Exception as e:
            return initial_f

    def generate_initial_population(self, pop_size: int) -> List[List[float]]:
        """Generate diverse initial population"""
        population = []
        for i in range(pop_size):
            # Create function with adaptive initialization
            n_steps = max(800, min(2000, 1000 + i * 50))  # Vary number of steps
            
            # Create initial function
            f_init = self.generator.adaptive_step_function_initialization(n_steps)
            
            # Add slight randomization to break symmetry
            f_init = [val * (0.9 + random.random() * 0.2) for val in f_init]
            
            population.append(f_init)
            
        return population

    def run_evolutionary_optimization(self, initial_population: List[List[float]]) -> List[float]:
        """Run evolutionary optimization on initial population"""
        if not initial_population:
            return []
            
        # Sort population by fitness and keep top performers
        try:
            fitness_scores = []
            for candidate in initial_population:
                score = self.evaluate_candidate(candidate)
                fitness_scores.append(score)
            
            # Combine and sort by fitness
            combined = list(zip(initial_population, fitness_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 30% as elite
            elite_size = max(1, len(combined) // 3)
            elite_population = [ind for ind, _ in combined[:elite_size]]
            
            # If we have elite, refine them further
            if elite_population:
                # Apply local search to top individuals
                refined_elite = []
                for individual in elite_population:
                    refined = self.local_search_refinement(individual, max_iter=20)
                    refined_elite.append(refined)
                
                # Evaluate refined elite
                refined_scores = []
                for candidate in refined_elite:
                    score = self.evaluate_candidate(candidate)
                    refined_scores.append(score)
                
                # Find best among refined
                best_idx = np.argmax(refined_scores)
                return refined_elite[best_idx]
                
        except Exception as e:
            # Fallback to simple approach
            pass
            
        # Fallback to first individual if nothing works
        return initial_population[0] if initial_population else []

    def optimize_single_function(self) -> List[float]:
        """Single function optimization attempt"""
        start_time = time.time()
        
        # Phase 1: Generate diverse initial population
        initial_pop = self.generate_initial_population(30)
        
        # Phase 2: Evolutionary optimization
        best_function = self.run_evolutionary_optimization(initial_pop)
        
        # Phase 3: Additional refinement if time permits
        if best_function and time.time() - start_time < self.max_time_seconds - 10:
            # Apply local search refinement
            refined_local = self.local_search_refinement(best_function, max_iter=25)
            local_c2 = self.evaluator.compute_c2(refined_local)
            
            # Apply simulated annealing refinement (more intensive)
            if time.time() - start_time < self.max_time_seconds - 5:
                refined_sa = self.simulated_annealing_refinement(best_function, max_iter=150)
                sa_c2 = self.evaluator.compute_c2(refined_sa)
                
                if sa_c2 > local_c2:
                    best_function = refined_sa
                else:
                    best_function = refined_local
        
        return best_function

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses modularized evolutionary optimization approach with advanced refinement.
    """
    try:
        # Create optimizer pipeline
        optimizer = OptimizerPipeline(max_time_seconds=85)
        
        # Perform optimization
        f_values = optimizer.optimize_single_function()
        
        # Ensure we return at least some function
        if not f_values:
            # Fallback to simple construction
            f_values = [1.0] * 1000
            
        return f_values
        
    except Exception as e:
        # Fallback to random generation if anything fails
        print(f"Error in optimization: {e}")
        f_values = [np.random.random()] * np.random.randint(500, 2000)
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")