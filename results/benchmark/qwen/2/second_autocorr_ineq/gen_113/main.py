# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from numba import jit

class AutoconvolutionEvaluator:
    """Handles all autoconvolution norm computations with numerical stability."""
    
    @staticmethod
    @jit(nopython=True)
    def compute_norms_fast(f_values):
        """Fast computation of autoconvolution norms using numba optimization"""
        if len(f_values) < 1:
            return 0.0, 1e-12, 1e-12
            
        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)
        
        # Compute autoconvolution using fast convolution
        g = signal.convolve(f, f, mode='full')
        g = g[len(f)-1:]  # Keep only the relevant part
        
        # Compute norms with numerical stability checks
        g_abs = np.abs(g)
        
        # ||g||₂² (L2 norm squared) - use trapezoidal rule properly
        g_sq = g_abs ** 2
        if len(g_sq) < 2:
            norm_2_sq = 0.0 if len(g_sq) == 0 else g_sq[0]
        else:
            # Trapezoidal integration: sum((y[i] + y[i+1])/2 * delta_x)
            norm_2_sq = np.sum((g_sq[:-1] + g_sq[1:]) / 2.0)
        
        # ||g||₁ (L1 norm) - sum of absolute values
        norm_1 = np.sum(g_abs)
        
        # ||g||∞ (infinity norm)
        norm_inf = np.max(g_abs)
        
        # Numerical stability checks
        norm_2_sq = max(0.0, norm_2_sq)
        norm_1 = max(1e-12, norm_1)  # Avoid division by zero
        norm_inf = max(1e-12, norm_inf)  # Avoid division by zero
        
        return norm_2_sq, norm_1, norm_inf

class MultiScalePeakBuilder:
    """Handles construction of step functions with multi-scale peak optimization."""
    
    @staticmethod
    def generate_multiscale_peaks(n_points, n_peaks=None, resolution_level=1):
        """
        Generate step function using multi-scale approach with adaptive peak placement
        """
        if n_peaks is None:
            n_peaks = max(5, min(30, n_points // 50))
            
        # Create domain
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros_like(x)
        
        # Determine peak distribution based on resolution level
        if resolution_level == 1:  # Coarse resolution
            n_peaks = min(n_peaks, 15)
            # Sparse logarithmic distribution
            positions_log = np.logspace(np.log(0.02), np.log(0.12), max(1, n_peaks//3))
            positions = np.concatenate([positions_log, -positions_log[::-1], [0]])
            positions = positions[:n_peaks]
        elif resolution_level == 2:  # Medium resolution
            n_peaks = min(n_peaks, 25)
            # Balanced distribution
            positions = np.linspace(-0.23, 0.23, n_peaks)
        else:  # Fine resolution
            # Dense distribution with spatial awareness
            positions = np.linspace(-0.23, 0.23, n_peaks)
            # Add some variation to avoid regularity
            positions += np.random.uniform(-0.005, 0.005, n_peaks)
            
        # Remove duplicates and ensure valid range
        positions = np.unique(positions)
        positions = positions[np.abs(positions) <= 0.23]
        positions = positions[:n_peaks]
        
        # Generate peaks with adaptive characteristics
        peak_amplitudes = []
        peak_widths = []
        
        for i, pos in enumerate(positions):
            # Amplitude varies with position - more peaks at center for better autoconvolution
            center_distance = abs(pos)
            # Base amplitude with preference for center area
            base_amp = 1.0 - center_distance * 2.0  # Peak at center
            base_amp = max(0.1, base_amp)  # Minimum amplitude
            base_amp *= np.random.uniform(0.8, 1.2)  # Random variation
            
            # Adjust according to peak position
            if center_distance < 0.05:
                base_amp *= 1.5  # Emphasize center peaks
            elif center_distance > 0.15:
                base_amp *= 0.7  # Reduce edge peaks
            
            # Scale amplitude by resolution level
            base_amp *= (1.0 + (resolution_level - 1) * 0.2)
            
            amp = max(0.1, base_amp)
            peak_amplitudes.append(amp)
            
            # Width varies with position to create more interesting autoconvolution
            base_width = 0.01 + 0.04 * np.exp(-center_distance * 3.0)
            # Adjust by resolution level
            base_width *= (0.8 + (resolution_level - 1) * 0.2)
            # Add variation
            width = max(0.005, base_width * np.random.uniform(0.8, 1.2))
            peak_widths.append(width)
        
        # Create peaks with optimized parameters
        for i, (pos, amp, width) in enumerate(zip(positions, peak_amplitudes, peak_widths)):
            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width)**2)
            f_values += gaussian_peak
            
        # Apply smoothing based on resolution level
        if n_points > 100:
            # Resolution-dependent smoothing
            if resolution_level == 1:
                window_size = max(3, min(15, int(n_points / 100)))
            elif resolution_level == 2:
                window_size = max(3, min(21, int(n_points / 75)))
            else:
                window_size = max(3, min(31, int(n_points / 50)))
                
            if window_size % 2 == 0:
                window_size += 1
            try:
                f_values = gaussian_filter1d(f_values, window_size, mode='nearest')
            except:
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')
        
        # Ensure non-negativity and normalize
        f_values = np.maximum(f_values, 0)
        
        # Normalize for stability
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.5)  # Scale down gently
            
        return f_values.tolist()

class MultiObjectiveOptimizer:
    """Specialized multi-objective optimizer for C2 maximization"""
    
    def __init__(self):
        self.evaluator = AutoconvolutionEvaluator()
        
    def multi_objective_fitness(self, params, target_c2=None):
        """
        Fitness function that balances multiple objectives for C2 maximization
        """
        # Ensure non-negativity and normalize
        params = np.maximum(params, 0)
        
        # Compute norms
        norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms_fast(params)
        
        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 1e10
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        
        # Multi-objective balancing: prefer higher C2, but also consider
        # that we want well-distributed peaks (lower L1 and higher L2)
        # This encourages flatter autoconvolution profiles
        if c2 <= 0:
            return 1e10
            
        # Penalize if C2 is below a threshold (we want to maximize)
        return -c2  # Negative because we minimize in scipy
        
    def optimize_with_local_search(self, initial_params, max_evaluations=50):
        """
        Enhanced optimization using multiple strategies
        """
        try:
            # Strategy 1: Differential evolution on the full parameter space
            bounds = [(0, 1) for _ in range(len(initial_params))]
            
            # Run multiple DE optimizations with different settings
            best_result = None
            best_c2 = float('-inf')
            
            for strategy in ['best1bin', 'rand1bin', 'best2bin']:
                try:
                    result = differential_evolution(
                        self.multi_objective_fitness,
                        bounds,
                        args=(None,),
                        maxiter=max_evaluations // 3,
                        popsize=8,
                        seed=42,
                        strategy=strategy,
                        disp=False
                    )
                    
                    # Evaluate result
                    optimized_params = np.maximum(result.x, 0)
                    norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms_fast(optimized_params)
                    
                    if norm_1 > 1e-12 and norm_inf > 1e-12:
                        c2 = norm_2_sq / (norm_1 * norm_inf)
                        if c2 > best_c2:
                            best_c2 = c2
                            best_result = result
                            
                except Exception as e:
                    continue  # Skip failed optimization attempts
                    
            if best_result is not None:
                return np.maximum(best_result.x, 0).tolist()
                
        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")
            
        # Return unchanged if optimization fails
        return initial_params

class MultiScaleOptimizerPipeline:
    """Main pipeline that performs multi-scale optimization"""
    
    def __init__(self):
        self.evaluator = AutoconvolutionEvaluator()
        self.builder = MultiScalePeakBuilder()
        self.optimizer = MultiObjectiveOptimizer()
        
    def evaluate_candidate(self, f_values):
        """Evaluate a single candidate function"""
        try:
            norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms_fast(f_values)
            
            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return 0.0
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {str(e)}")
            return 0.0
            
    def construct_multiscale_candidates(self, n_candidates=20, n_points=2000, resolution_levels=[1,2,3]):
        """
        Construct candidates with multiple resolutions and peak configurations for better exploration
        """
        candidates = []
        
        def build_single_candidate(index):
            try:
                # Try different resolution levels
                resolution_level = resolution_levels[index % len(resolution_levels)]
                f_values = self.builder.generate_multiscale_peaks(n_points, None, resolution_level)
                c2 = self.evaluate_candidate(f_values)
                return (c2, f_values)
            except Exception as e:
                warnings.warn(f"Candidate {index} construction failed: {str(e)}")
                return (0.0, [])
        
        # Process candidates in parallel
        with ThreadPoolExecutor(max_workers=min(8, n_candidates)) as executor:
            futures = [executor.submit(build_single_candidate, i) for i in range(n_candidates)]
            results = [future.result() for future in as_completed(futures)]
            
        # Filter out invalid candidates
        valid_candidates = [(c2, f_values) for c2, f_values in results if f_values and len(f_values) > 0]
        
        # Sort by C2 score
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return valid_candidates
    
    def construct_optimized_function(self, max_time_seconds=85):
        """
        Main function construction routine with multi-scale optimization
        """
        start_time = time.time()
        best_c2 = 0.0
        best_f = None
        
        # Multi-scale approach: start coarse, progressively refine
        resolution_levels = [1, 2, 3]  # Fine, medium, coarse
        n_points_list = [1000, 1500, 2000]  # Different resolutions
        
        # Phase 1: Coarse exploration with minimal resolution
        for attempt in range(3):  # Multiple passes to avoid local optima
            if time.time() - start_time > max_time_seconds:
                break
                
            # Try different resolutions
            for res_idx, n_points in enumerate(n_points_list):
                if time.time() - start_time > max_time_seconds:
                    break
                    
                # Generate candidates with different resolution levels
                candidates = self.construct_multiscale_candidates(
                    n_candidates=10, 
                    n_points=n_points, 
                    resolution_levels=[resolution_levels[res_idx]]
                )
                
                # Select best among generated candidates
                for c2, f_values in candidates:
                    if c2 > best_c2:
                        best_c2 = c2
                        best_f = f_values
                        
                # Early termination if we already have good result
                if best_c2 > 0.95:
                    break
                    
        # Phase 2: Refine with higher resolution and better optimization
        if best_f is not None:
            try:
                # Increase resolution for better final refinement
                high_res_f = self.builder.generate_multiscale_peaks(3000, None, 3)
                high_res_c2 = self.evaluate_candidate(high_res_f)
                
                if high_res_c2 > best_c2:
                    best_c2 = high_res_c2
                    best_f = high_res_f
                    
                # Apply local optimization with enhanced search
                refined_f = self.optimizer.optimize_with_local_search(best_f, max_evaluations=30)
                refined_c2 = self.evaluate_candidate(refined_f)
                
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_f = refined_f
                    
            except Exception as e:
                warnings.warn(f"Final refinement failed: {str(e)}")
        
        # Phase 3: Additional refinement with different approaches
        if best_f is not None and time.time() - start_time < max_time_seconds * 0.8:
            try:
                # Try a completely different approach - adaptive peak tuning
                # Use a smaller subset for more focused optimization
                reduced_f = self.builder.generate_multiscale_peaks(1000, 15, 3)
                refined_f = self.optimizer.optimize_with_local_search(reduced_f, max_evaluations=20)
                refined_c2 = self.evaluate_candidate(refined_f)
                
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_f = refined_f
                    
            except Exception as e:
                warnings.warn(f"Second refinement failed: {str(e)}")
        
        # Fallback if nothing worked
        if best_f is None:
            # Use a more robust fallback with better distribution
            n_points = 1000
            x = np.linspace(-0.25, 0.25, n_points)
            # Create a more structured function - bell-shaped with multiple peaks
            f_values = np.zeros(n_points)
            peak_positions = np.linspace(-0.23, 0.23, 8)
            
            for pos in peak_positions:
                width = 0.02 + 0.03 * np.exp(-abs(pos)*2)
                height = 0.8 + 0.2 * np.sin(pos * 5)
                peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
                f_values += peak
                
            f_values = np.maximum(f_values, 0)
            max_val = np.max(f_values)
            if max_val > 0:
                f_values = f_values / (max_val * 1.5)
            best_f = f_values.tolist()
            
        return best_f

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Create pipeline instance and run optimization
    pipeline = MultiScaleOptimizerPipeline()
    return pipeline.construct_optimized_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")