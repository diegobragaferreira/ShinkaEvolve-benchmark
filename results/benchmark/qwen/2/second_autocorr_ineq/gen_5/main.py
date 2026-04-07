# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import time

def construct_function() -> list[float]:
    """
    Construct a step function that maximizes C2 = ||g||₂² / (||g||₁ · ||g||∞)
    where g = f*f (autoconvolution) and f is the step function.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    max_time = 90.0  # seconds
    start_time = time.time()
    
    # Initial coarse grid
    n_initial = 100
    f = np.random.rand(n_initial)
    f = np.maximum(f, 0)  # Ensure non-negative
    
    best_c2 = 0.0
    best_f = f.copy()
    
    # Multi-scale optimization
    scales = [1, 2, 4, 8, 16]
    scale_factors = [1.0, 0.7, 0.5, 0.3, 0.2]
    
    for scale_idx, (scale_factor, grid_scale) in enumerate(zip(scale_factors, scales)):
        if time.time() - start_time > max_time * 0.95:
            break
            
        # Determine grid size for this scale
        n_grid = max(100, int(n_initial * grid_scale))
        
        # Refine function at this scale
        current_f = np.interp(np.linspace(0, 1, n_grid), np.linspace(0, 1, len(f)), f)
        current_f = np.maximum(current_f, 0)
        
        # Local optimization at this scale
        for _ in range(50):
            if time.time() - start_time > max_time * 0.95:
                break
                
            # Compute current autoconvolution
            g = signal.convolve(current_f, current_f, mode='full')
            g = g[len(g)//2:]  # Take positive lags only
            
            # Compute norms
            g_abs = np.abs(g)
            norm_1 = np.sum(g_abs) / len(g)
            norm_2_sq = np.sum(g_abs**2) / len(g)
            norm_inf = np.max(g_abs)
            
            # Prevent division by zero
            if norm_1 < 1e-12 or norm_inf < 1e-12:
                continue
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            
            if c2 > best_c2:
                best_c2 = c2
                best_f = current_f.copy()
            
            # Gradient estimation using finite differences
            eps = 1e-4
            grad = np.zeros_like(current_f)
            
            for i in range(len(current_f)):
                # Perturb point i
                perturbed = current_f.copy()
                perturbed[i] += eps
                perturbed = np.maximum(perturbed, 0)
                
                # Compute convolutions
                g_pert = signal.convolve(perturbed, perturbed, mode='full')
                g_pert = g_pert[len(g_pert)//2:]
                
                # Compute gradient component
                g_pert_abs = np.abs(g_pert)
                norm_1_pert = np.sum(g_pert_abs) / len(g_pert)
                norm_2_sq_pert = np.sum(g_pert_abs**2) / len(g_pert)
                norm_inf_pert = np.max(g_pert_abs)
                
                if norm_1_pert < 1e-12 or norm_inf_pert < 1e-12:
                    continue
                    
                c2_pert = norm_2_sq_pert / (norm_1_pert * norm_inf_pert)
                grad[i] = (c2_pert - c2) / eps
            
            # Update function using gradient ascent
            learning_rate = scale_factor * 0.01
            current_f += learning_rate * grad
            current_f = np.maximum(current_f, 0)  # Keep non-negative
            
            # Normalize to prevent explosion
            current_f /= (np.mean(current_f) + 1e-6)
    
    # Final refinement with higher resolution
    if time.time() - start_time < max_time * 0.95:
        final_n = min(10000, max(5000, len(best_f) * 2))
        refined_f = np.interp(np.linspace(0, 1, final_n), np.linspace(0, 1, len(best_f)), best_f)
        refined_f = np.maximum(refined_f, 0)
        
        # Final optimization pass
        for _ in range(20):
            if time.time() - start_time > max_time * 0.95:
                break
                
            g = signal.convolve(refined_f, refined_f, mode='full')
            g = g[len(g)//2:]
            g_abs = np.abs(g)
            norm_1 = np.sum(g_abs) / len(g)
            norm_2_sq = np.sum(g_abs**2) / len(g)
            norm_inf = np.max(g_abs)
            
            if norm_1 < 1e-12 or norm_inf < 1e-12:
                break
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            
            # Gradient descent update
            eps = 1e-5
            grad = np.zeros_like(refined_f)
            
            for i in range(len(refined_f)):
                perturbed = refined_f.copy()
                perturbed[i] += eps
                perturbed = np.maximum(perturbed, 0)
                
                g_pert = signal.convolve(perturbed, perturbed, mode='full')
                g_pert = g_pert[len(g_pert)//2:]
                g_pert_abs = np.abs(g_pert)
                norm_1_pert = np.sum(g_pert_abs) / len(g_pert)
                norm_2_sq_pert = np.sum(g_pert_abs**2) / len(g_pert)
                norm_inf_pert = np.max(g_pert_abs)
                
                if norm_1_pert < 1e-12 or norm_inf_pert < 1e-12:
                    continue
                    
                c2_pert = norm_2_sq_pert / (norm_1_pert * norm_inf_pert)
                grad[i] = (c2_pert - c2) / eps
            
            learning_rate = 0.005
            refined_f -= learning_rate * grad
            refined_f = np.maximum(refined_f, 0)
            
            # Early stopping if improvement is minimal
            if abs(c2 - best_c2) < 1e-8:
                break
                
            best_c2 = c2
            best_f = refined_f.copy()
    
    # Convert to list format and ensure final quality
    final_result = best_f.tolist()
    
    # Post-process to ensure numerical stability and good metrics
    final_result = [max(0, x) for x in final_result]
    
    return final_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
