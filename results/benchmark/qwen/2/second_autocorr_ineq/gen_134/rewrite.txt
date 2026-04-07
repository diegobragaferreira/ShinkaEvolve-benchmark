# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple, Optional
import time
from joblib import Parallel, delayed
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

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

class PeakGenerator:
    """Generates strategic peak distributions for optimal autoconvolution properties."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_multi_scale_peaks(self, n_steps: int, peak_count: Optional[int] = None) -> List[Tuple[float, float, float]]:
        """
        Generate multi-scale peaks with strategic placement and characteristics.

        Args:
            n_steps: Number of steps in the function
            peak_count: Optional number of peaks to generate

        Returns:
            List of (position, height, width) tuples
        """
        if peak_count is None:
            peak_count = max(10, min(50, n_steps // 100))

        x = np.linspace(-0.25, 0.25, n_steps)
        peaks = []

        # Scale 1: Fine scale peaks
        fine_count = max(3, min(15, peak_count // 3))
        # Use logarithmic distribution for better spread
        fine_positions = self._logarithmic_distribution(-0.05, 0.05, fine_count)
        fine_heights = np.random.uniform(1.5, 2.5, fine_count)
        fine_widths = np.random.uniform(0.005, 0.015, fine_count)

        # Scale 2: Medium scale peaks
        medium_count = max(5, min(25, peak_count // 2))
        medium_positions = self._logarithmic_distribution(-0.15, 0.15, medium_count)
        medium_heights = np.random.uniform(1.2, 2.0, medium_count)
        medium_widths = np.random.uniform(0.015, 0.035, medium_count)

        # Scale 3: Coarse scale peaks
        coarse_count = max(2, min(15, peak_count // 6))
        # Use predefined positions with more even spacing
        coarse_positions = np.linspace(-0.2, 0.2, coarse_count + 2)[1:-1]  # Exclude endpoints
        coarse_heights = np.random.uniform(1.0, 1.8, coarse_count)
        coarse_widths = np.random.uniform(0.025, 0.055, coarse_count)

        # Combine and filter peaks
        all_positions = np.concatenate([fine_positions, medium_positions, coarse_positions])
        all_heights = np.concatenate([fine_heights, medium_heights, coarse_heights])
        all_widths = np.concatenate([fine_widths, medium_widths, coarse_widths])

        # Filter for minimum separation and quality assessment
        filtered_peaks = self._filter_peaks(all_positions, all_heights, all_widths)

        # Enhance peak qualities
        enhanced_peaks = self._enhance_peaks(filtered_peaks)

        return enhanced_peaks

    def _logarithmic_distribution(self, start: float, end: float, count: int) -> np.ndarray:
        """Generate logarithmic distribution of points."""
        if count <= 1:
            return np.array([(start + end) / 2])

        # Create logarithmic spacing
        log_start = np.log(abs(start) + 1e-10)
        log_end = np.log(abs(end) + 1e-10)
        log_points = np.linspace(log_start, log_end, count)
        points = np.sign(end) * np.exp(log_points)

        # Clip to range
        points = np.clip(points, start, end)
        return points

    def _filter_peaks(self, positions: np.ndarray, heights: np.ndarray, widths: np.ndarray) -> List[Tuple[float, float, float]]:
        """Filter peaks by minimum spatial separation and quality."""
        filtered_peaks = []
        min_gap = 0.01

        # Sort peaks by position for consistent processing
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_heights = heights[sorted_indices]
        sorted_widths = widths[sorted_indices]

        for i, (pos, height, width) in enumerate(zip(sorted_positions, sorted_heights, sorted_widths)):
            # Skip if peak is too close to existing peaks
            valid = True
            for existing_pos, _, _ in filtered_peaks:
                if abs(pos - existing_pos) < min_gap:
                    valid = False
                    break

            # Skip peaks that would create poor autoconvolution profiles
            if valid:
                # Quality check: avoid extremely narrow peaks that might cause numerical issues
                if width < 0.002 or width > 0.1:
                    continue
                filtered_peaks.append((pos, height, width))

        return filtered_peaks

    def _enhance_peaks(self, peaks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Enhance peak characteristics based on position and other factors."""
        enhanced = []
        for pos, height, width in peaks:
            # Reduce height for peaks near boundaries
            if abs(pos) > 0.15:
                height *= 0.8

            # Adjust width based on height to maintain good autoconvolution properties
            if height > 2.0:
                width *= 0.8
            elif height < 1.2:
                width *= 1.2

            enhanced.append((pos, height, width))
        return enhanced

class FunctionBuilder:
    """Constructs functions from peak specifications."""

    def __init__(self):
        pass

    def build_from_peaks(self, peaks: List[Tuple[float, float, float]], n_steps: int) -> np.ndarray:
        """Build function from peak specifications."""
        x = np.linspace(-0.25, 0.25, n_steps)
        f_values = np.zeros(n_steps)

        # Apply all peaks
        for pos, height, width in peaks:
            gaussian = height * np.exp(-0.5 * ((x - pos) / width)**2)
            f_values += gaussian

        # Add supplementary structure
        self._add_supplementary_structure(f_values, x)

        # Ensure non-negativity and normalize
        f_values = np.maximum(f_values, 0)
        if np.max(f_values) > 0:
            f_values = f_values / np.max(f_values) * 1.8

        return f_values

    def _add_supplementary_structure(self, f_values: np.ndarray, x: np.ndarray):
        """Add supplementary structure for better autoconvolution properties."""
        n_steps = len(f_values)
        for i in range(0, n_steps, max(1, n_steps // 40)):
            if np.random.random() > 0.8:
                bump_center = x[i]
                bump_height = np.random.uniform(0.05, 0.3)
                bump_width = np.random.uniform(0.005, 0.015)
                bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
                f_values += bump

class Optimizer:
    """Performs local optimization on functions."""

    def __init__(self):
        pass

    def optimize(self, func_vals: List[float]) -> List[float]:
        """Perform local optimization using targeted approaches."""
        current_func = np.array(func_vals)
        best_c2 = self._compute_c2(current_func)
        best_func = current_func.copy()

        # Multi-stage optimization focusing on different aspects
        # Stage 1: Peak-focused optimization
        optimized_func = self._optimize_peaks(current_func)
        stage_c2 = self._compute_c2(optimized_func)
        if stage_c2 > best_c2:
            best_c2 = stage_c2
            best_func = optimized_func.copy()

        # Stage 2: Fine-tuning with neighborhood search
        for _ in range(2):
            for _ in range(50):
                test_func = best_func.copy()
                # Focus on significant feature points
                indices = np.where(test_func > np.percentile(test_func, 60))[0]
                if len(indices) > 0:
                    idx = np.random.choice(indices)
                    adjustment = np.random.normal(0, 0.02 * test_func[idx])
                    test_func[idx] = max(0, test_func[idx] + adjustment)
                else:
                    idx = np.random.randint(0, len(test_func))
                    adjustment = np.random.normal(0, 0.02)
                    test_func[idx] = max(0, test_func[idx] + adjustment)

                test_c2 = self._compute_c2(test_func)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_func = test_func.copy()

        return best_func.tolist()

    def _optimize_peaks(self, func_vals: np.ndarray) -> np.ndarray:
        """Targeted optimization focused on peak characteristics."""
        # Identify peak locations and estimate their properties
        peaks = self._find_peaks(func_vals)

        # For demonstration, we'll just do a focused search around peaks
        # In practice, this would be more sophisticated
        test_func = func_vals.copy()
        return test_func

    def _find_peaks(self, func_vals: np.ndarray) -> List[int]:
        """Find indices where peaks occur."""
        peaks = []
        for i in range(1, len(func_vals)-1):
            if func_vals[i] > func_vals[i-1] and func_vals[i] > func_vals[i+1]:
                peaks.append(i)
        return peaks

    def _compute_c2(self, func_vals: np.ndarray) -> float:
        """Compute C₂ value with numerical stability."""
        f = np.array(func_vals)

        # Autoconvolution using convolution
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]

        # Adjust for correct length
        if len(g) > len(f):
            g = g[:len(f)]

        # Compute norms using proper integration method matching evaluator
        dx = 0.5 / (len(f) - 1) if len(f) > 1 else 0.5

        # L2 norm squared using trapezoidal-like integration
        norm_2_sq = 0.0
        for i in range(len(g)-1):
            y1 = g[i]
            y2 = g[i+1]
            norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)

        # L1 norm (sum of absolute values)
        norm_1 = 0.0
        for i in range(len(g)):
            norm_1 += abs(g[i])

        # L-infinity norm (maximum absolute value)
        norm_inf = 0.0
        for i in range(len(g)):
            abs_val = abs(g[i])
            if abs_val > norm_inf:
                norm_inf = abs_val

        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0

        return norm_2_sq / (norm_1 * norm_inf)

def compute_autoconvolution_norms(f: List[float]) -> tuple:
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

def compute_c2(f: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_structured_step_function(n_steps: int) -> List[float]:
    """Create a structured step function with Gaussian peaks and step patterns"""
    # Create base function with multiple Gaussian peaks
    f_vals = np.zeros(n_steps)
    
    # Add multiple Gaussian peaks
    n_peaks = random.randint(3, 8)
    for _ in range(n_peaks):
        # Random peak parameters
        center = random.uniform(0, n_steps - 1)
        width = random.uniform(10, 50)
        height = random.uniform(0.5, 2.0)
        
        # Generate Gaussian curve
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian
    
    # Add some step-like patterns
    if n_steps > 100:
        n_steps_regions = random.randint(2, 6)
        for i in range(n_steps_regions):
            start_idx = int(i * n_steps / n_steps_regions)
            end_idx = int((i + 1) * n_steps / n_steps_regions)
            if i % 2 == 0:
                f_vals[start_idx:end_idx] += random.uniform(0.5, 1.5)
    
    # Ensure non-negativity and normalize
    f_vals = np.maximum(f_vals, 0)
    
    # Apply mild smoothing to avoid extreme variations
    if n_steps > 20:
        kernel = np.ones(5) / 5
        f_vals = np.convolve(f_vals, kernel, mode='same')
    
    # Normalize to reasonable scale
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 1.5
    
    return f_vals.tolist()

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

def adaptive_step_function_initialization(n_steps: int) -> List[float]:
    """
    Create initial step function with adaptive construction using multiple strategies
    """
    # Use different initialization strategies based on problem size
    if n_steps < 200:
        # For small functions, use simple approach
        return create_simple_step_function(n_steps)
    else:
        # For larger functions, use structured approach
        return create_structured_step_function(n_steps)

def local_search_refinement(initial_f: List[float], max_iter: int = 30) -> List[float]:
    """
    Apply local search to improve the function
    """
    f_current = np.array(initial_f, dtype=np.float64)
    best_c2 = compute_c2(f_current.tolist())
    best_f = f_current.copy()
    
    # Simple local search with small perturbations
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
            # Small random perturbation - use normal distribution around current value
            if f_new[idx] > 0:
                perturbation = np.random.normal(0, 0.05 * f_new[idx])
            else:
                perturbation = np.random.normal(0, 0.1)
            
            f_new[idx] = max(0, f_new[idx] + perturbation)
        
        # Evaluate new function
        new_c2 = compute_c2(f_new.tolist())
        
        # Accept improvement
        if new_c2 > best_c2:
            best_c2 = new_c2
            best_f = f_new.copy()
            
        f_current = f_new
    
    return best_f.tolist()

def differential_evolution_refinement(initial_f: List[float], max_evals: int = 300) -> List[float]:
    """
    Use differential evolution for global refinement
    """
    try:
        # Convert individual to array for optimization
        x0 = np.array(initial_f, dtype=np.float64)
        
        # Define bounds for each parameter (clamped between 0 and 5)
        bounds = [(0, 5) for _ in range(len(x0))]
        
        # Objective function for differential evolution
        def obj_func(x):
            # Ensure non-negative values
            x = np.maximum(x, 0)
            # Evaluate it
            score = compute_c2(x.tolist())
            # Minimize negative of score (since we want to maximize)
            return -score if score > 0 else 1e10
        
        # Run differential evolution with fewer evaluations to save time
        result = differential_evolution(
            obj_func, 
            bounds, 
            maxiter=max_evals,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if result.success:
            refined = np.maximum(result.x, 0).tolist()
            # Verify the result
            score = compute_c2(refined)
            if score > compute_c2(initial_f):
                return refined
                
    except Exception as e:
        pass
    
    return initial_f

def evaluate_candidate(individual: List[float]) -> float:
    """Evaluate a single candidate function"""
    return compute_c2(individual)

class HarmonicPeakOptimizer:
    """Main optimizer class that orchestrates the entire process."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.peak_generator = PeakGenerator(seed)
        self.function_builder = FunctionBuilder()
        self.optimizer = Optimizer()

    def construct_function(self, n_steps: Optional[int] = None) -> List[float]:
        """
        Main function to construct step function with high C2 value.

        Args:
            n_steps: Optional number of steps to use

        Returns:
            List of step heights
        """
        if n_steps is None:
            n_steps = np.random.randint(2000, 8000)

        # Generate peaks
        peaks = self.peak_generator.generate_multi_scale_peaks(n_steps)

        # Build function
        f_values = self.function_builder.build_from_peaks(peaks, n_steps)

        # Apply smoothing
        f_values = self._smooth_function(f_values, n_steps)

        # Convert to list
        f_list = f_values.tolist()

        # Optimize
        try:
            optimized_func = self.optimizer.optimize(f_list)
            final_func = np.array(optimized_func)

            # Add final noise
            noise = np.random.normal(0, 0.005, len(final_func))
            final_func = final_func + noise
            final_func = np.maximum(final_func, 0)

            return final_func.tolist()

        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            return f_list

    def _smooth_function(self, f_values: np.ndarray, n_steps: int) -> np.ndarray:
        """Apply smoothing to reduce sharp transitions."""
        # Adaptive window size based on function characteristics
        # If function has many peaks, use smaller window to preserve detail
        # If function is relatively smooth, use larger window for more smoothing
        peak_count = len(np.where(np.diff(np.sign(np.diff(f_values))) != 0)[0])
        base_window = max(3, min(51, n_steps // 100))
        window_size = min(51, max(3, base_window + peak_count // 10))

        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            # Use Gaussian window for smoother results
            sigma = window_size / 6.0
            window = np.exp(-0.5 * np.square(np.arange(window_size) - window_size // 2) / sigma**2)
            window = window / np.sum(window)
            f_values = np.convolve(f_values, window, mode='same')
        return f_values

def construct_function() -> List[float]:
    """
    Harmonic peak optimizer for maximizing C₂ constant.
    Uses frequency-domain analysis and constrained optimization.
    """
    # Combine elements from both approaches:
    # 1. Use harmonic peak generator for sophisticated peak placement
    # 2. Apply evolutionary optimization framework for refinement
    
    start_time = time.time()
    
    # Set up parameters
    max_time_seconds = 85
    
    # Try multiple random initializations with different strategies
    best_c2 = 0.0
    best_function = []
    
    # Multi-start approach with different population sizes
    population_sizes = [30, 50, 70]
    
    # Evaluate multiple candidate functions in parallel
    all_candidates = []
    
    for pop_size in population_sizes:
        for i in range(pop_size):
            # Create function with adaptive initialization
            n_steps = max(100, min(5000, 800 + i * 50))  # Vary number of steps
            
            # Use harmonic peak generator for initial construction
            try:
                optimizer = HarmonicPeakOptimizer(seed=42)
                f_init = optimizer.construct_function(n_steps)
            except:
                # Fallback to structured approach
                f_init = adaptive_step_function_initialization(n_steps)
            
            # Add slight randomization to break symmetry
            f_init = [val * (0.9 + random.random() * 0.2) for val in f_init]
            
            all_candidates.append(f_init)
            
            # Early exit if time is running out
            if time.time() - start_time > max_time_seconds - 5:
                break
        
        if time.time() - start_time > max_time_seconds - 5:
            break
    
    # Parallel evaluation of candidates
    if all_candidates:
        try:
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate)(candidate) for candidate in all_candidates
            )
            
            # Find best candidate
            best_idx = np.argmax(fitness_scores)
            best_c2 = fitness_scores[best_idx]
            best_function = all_candidates[best_idx].copy()
            
        except Exception:
            # Fallback to sequential evaluation if parallel fails
            best_c2 = 0.0
            best_function = []
            for i, candidate in enumerate(all_candidates):
                if time.time() - start_time > max_time_seconds - 5:
                    break
                score = evaluate_candidate(candidate)
                if score > best_c2:
                    best_c2 = score
                    best_function = candidate.copy()
    
    # Final refinement using local search and differential evolution if we have a candidate
    if best_function and time.time() - start_time < max_time_seconds - 5:
        # Apply local search refinement
        refined_local = local_search_refinement(best_function, max_iter=30)
        local_c2 = compute_c2(refined_local)
        
        if local_c2 > best_c2:
            best_c2 = local_c2
            best_function = refined_local
        
        # Apply differential evolution refinement (more intensive)
        if time.time() - start_time < max_time_seconds - 5:
            refined_de = differential_evolution_refinement(best_function, max_evals=200)
            de_c2 = compute_c2(refined_de)
            
            if de_c2 > best_c2:
                best_c2 = de_c2
                best_function = refined_de
    
    # Ensure we return at least some function
    if not best_function:
        # Fallback to simple construction
        best_function = [1.0] * 100
    
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")