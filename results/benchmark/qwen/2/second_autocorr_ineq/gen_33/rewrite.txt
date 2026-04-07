# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using adaptive Gaussian optimization."""
    np.random.seed(42)  # For reproducibility
    
    # Phase 1: Adaptive Gaussian Construction
    def adaptive_gaussian_construction():
        n_steps = np.random.randint(500, 3000)  # More stable range
        
        # Create domain
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Logarithmic peak positioning to ensure adequate spacing
        num_peaks = np.random.randint(5, 15)
        log_positions = np.logspace(np.log10(0.01), np.log10(0.24), num_peaks)
        peak_positions = np.concatenate([[-0.25 + p for p in log_positions], 
                                        [0.25 - p for p in log_positions]])
        # Remove duplicates and ensure within bounds
        peak_positions = np.unique([p for p in peak_positions if -0.25 <= p <= 0.25])
        peak_positions = peak_positions[:num_peaks]
        
        # Initialize function
        base_function = np.zeros_like(x)
        
        # Add peaks with adaptive amplitudes
        for i, pos in enumerate(peak_positions):
            # Peak height decreases with position to favor inner regions
            peak_height = np.random.uniform(1.0, 2.5) * (1.0 - abs(pos)/0.25)
            peak_width = np.random.uniform(0.02, 0.06)
            
            gaussian_peak = peak_height * np.exp(-0.5 * ((x - pos) / peak_width)**2)
            base_function += gaussian_peak
            
        # Add small bumps for complexity
        for _ in range(np.random.randint(2, 6)):
            bump_center = np.random.uniform(-0.25, 0.25)
            bump_height = np.random.uniform(0.2, 0.8)
            bump_width = np.random.uniform(0.005, 0.02)
            bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
            base_function += bump
            
        # Ensure non-negativity
        base_function = np.maximum(base_function, 0)
        
        # Normalize to reasonable range
        if np.max(base_function) > 0:
            base_function = base_function / np.max(base_function) * 1.5
            
        # Apply smoothing to reduce sharp transitions
        try:
            smoothed = savgol_filter(base_function, min(51, len(base_function)-1), 3)
            base_function = np.maximum(smoothed, 0)
        except:
            pass
            
        return base_function
    
    # Phase 2: Local Optimization of Peak Parameters
    def optimize_peaks(initial_func, n_steps):
        # Extract peak information
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Identify approximate peak locations
        peaks = []
        for i in range(1, len(initial_func)-1):
            if initial_func[i] > initial_func[i-1] and initial_func[i] > initial_func[i+1]:
                peaks.append((i, initial_func[i]))
                
        # Take top peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        selected_peaks = peaks[:min(8, len(peaks))]
        
        # Refine only peak positions and heights
        def objective(params):
            # Reconstruct function with given params
            temp_func = np.zeros_like(x)
            for i, (pos_idx, height) in enumerate(selected_peaks):
                center_pos = x[pos_idx] + (params[i*2] - 0.5) * 0.05  # Adjust by up to 0.05
                peak_height = height * (1.0 + params[i*2+1] * 0.5)  # Adjust by up to 50%
                width = np.random.uniform(0.02, 0.06)
                temp_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
            return -compute_c2(temp_func)  # Negative because we minimize
        
        # Initial parameter guess
        params0 = [0.0] * (len(selected_peaks) * 2)  # [delta_pos, delta_height] pairs
        
        # Optimize
        try:
            result = differential_evolution(objective, 
                                          bounds=[(-0.5, 0.5)] * (len(selected_peaks) * 2),
                                          maxiter=50, popsize=10, seed=42)
            optimized_params = result.x
        except:
            optimized_params = params0
            
        # Apply optimization results
        final_func = np.zeros_like(x)
        for i, (pos_idx, height) in enumerate(selected_peaks):
            center_pos = x[pos_idx] + (optimized_params[i*2] - 0.5) * 0.05
            peak_height = height * (1.0 + optimized_params[i*2+1] * 0.5)
            width = np.random.uniform(0.02, 0.06)
            final_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
            
        # Add remaining peaks from original
        for i in range(len(initial_func)):
            if not any(abs(x[i] - x[pos_idx]) < 0.01 for _, pos_idx in selected_peaks):
                final_func[i] += initial_func[i] * 0.5
                
        return final_func
    
    # Phase 3: Final C2 Computation and Return
    def compute_c2(func):
        # Compute autoconvolution g = f * f
        # Using discrete convolution
        g = np.convolve(func, func, mode='full')
        g = g[len(g)//2:]  # Take positive part
        
        # Truncate if necessary to match original length
        if len(g) > len(func):
            g = g[:len(func)]
            
        # Compute norms
        norm_2_sq = np.sum(g**2) * (0.5 / len(func))  # Approximate integral
        norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
            
        return norm_2_sq / (norm_1 * norm_inf)
    
    # Execute the phases
    try:
        # Construct initial function
        initial_func = adaptive_gaussian_construction()
        
        # Optimize peak parameters
        optimized_func = optimize_peaks(initial_func, len(initial_func))
        
        # Final check
        final_func = np.maximum(optimized_func, 0)
        
        # Add slight noise for robustness
        noise_level = 0.01
        noisy_func = final_func + np.random.normal(0, noise_level, len(final_func))
        noisy_func = np.maximum(noisy_func, 0)
        
        # Convert to step values
        step_values = noisy_func.tolist()
        
        return step_values
        
    except Exception as e:
        # Fallback to simple construction if anything fails
        fallback_func = np.ones(np.random.randint(300, 1000))
        return fallback_func.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")