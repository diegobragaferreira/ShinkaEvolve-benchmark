# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import cvxpy as cp
from cvxpy import Variable, Minimize, Problem, norm, sum_entries
import random
from typing import List

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the autoconvolution g = f*f and return its norms.
    Uses fast numba-compiled operations.
    """
    n = len(f_vals)
    # Autoconvolution using direct computation
    g = np.zeros(2*n - 1)

    # Compute convolution manually for efficiency
    # Optimized loop order for better performance
    for i in range(n):
        f_i = f_vals[i]
        for j in range(n):
            g[i + j] += f_i * f_vals[j]

    # Compute norms
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared)
    norm_g1 = np.sum(np.abs(g))
    norm_g_inf = np.max(np.abs(g))

    return norm_g2_squared, norm_g1, norm_g_inf

@njit
def calculate_c2(f_vals):
    """
    Calculate C2 value for given step function values.
    """
    norm_g2_squared, norm_g1, norm_g_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_g1 < 1e-15 or norm_g_inf < 1e-15:
        return 0.0

    c2 = norm_g2_squared / (norm_g1 * norm_g_inf)
    return c2

def construct_sparse_geometric_pattern(n_steps: int, alpha: float = 0.8) -> List[float]:
    """
    Construct a sparse geometric pattern that combines power-law decay with strategic spikes.
    This is based on the mathematical insight that optimal solutions often have sparse support
    with specific geometric properties.
    """
    # Create base power-law decay
    base_vals = np.power(np.arange(1, n_steps + 1), -alpha)
    
    # Rescale to maintain reasonable magnitude
    base_vals = base_vals / np.sum(base_vals) * 100
    
    # Add strategic spikes to encourage convolution peaks
    # Place peaks at quarter positions to create symmetry
    spikes = np.zeros(n_steps)
    peak_positions = [n_steps // 4, n_steps // 2, 3 * n_steps // 4]
    
    for pos in peak_positions:
        if pos < n_steps:
            # Add spike with higher amplitude than regular values
            spikes[pos] = 3.0 * np.max(base_vals)
    
    # Combine base and spikes
    combined = base_vals + spikes
    
    # Ensure non-negativity and normalize
    combined = np.maximum(combined, 0.0)
    if np.sum(combined) > 0:
        combined = combined / np.sum(combined) * 50
    
    return combined.tolist()

def solve_convex_relaxation(f_vals: List[float]) -> List[float]:
    """
    Solve a convex relaxation of the optimization problem to get a better starting point.
    This approach reformulates the C2 maximization as a convex optimization problem.
    """
    n = len(f_vals)
    if n == 0:
        return f_vals
    
    # Create a convex approximation by working with log-transformed variables
    # This avoids the non-convex nature of the original problem while providing 
    # good heuristic guidance
    
    # We'll use a simpler approach: construct a structured convex problem
    # that approximates the behavior of C2 maximization
    
    try:
        # Define variables for the optimization
        x = Variable(n, nonneg=True)
        
        # Simplified convex approximation (not the exact C2 but a good heuristic)
        # We maximize the L2 norm squared subject to L1 norm constraints
        # This encourages spreading out the values to maximize squared norms
        
        # Constraint: sum of values equals total mass (normalized)
        total_mass = np.sum(f_vals)
        if total_mass <= 0:
            total_mass = 1.0
            
        # Scale initial values to unit sum for constraint
        normalized_f_vals = np.array(f_vals) / total_mass
        
        # Set up the convex problem with simplified objective
        # Maximize the L2 norm squared subject to constraints
        objective = Minimize(-cp.norm(x, 2)**2)  # Negative because we minimize
        constraints = [cp.sum(x) == 1.0]  # Unit sum constraint
        
        # Solve with CVXPY
        prob = Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status == cp.OPTIMAL:
            result = x.value
            # Scale back to original magnitude
            return (result * total_mass).tolist()
    except:
        # Fallback to original values if convex optimization fails
        return f_vals

def multi_resolution_optimization(initial_vals: List[float], max_levels: int = 3) -> List[float]:
    """
    Perform multi-resolution optimization by starting with coarse resolution 
    and progressively refining.
    """
    current_vals = initial_vals.copy()
    
    # Start with a coarse version
    n_orig = len(current_vals)
    
    for level in range(max_levels):
        # Reduce resolution by factor of 2^(level+1) for coarser approximation
        reduction_factor = 2**(level + 1)
        n_current = max(10, n_orig // reduction_factor)
        
        # Create reduced resolution version
        if n_current < len(current_vals):
            reduced_vals = []
            step_size = len(current_vals) // n_current
            
            for i in range(n_current):
                start_idx = i * step_size
                end_idx = min((i + 1) * step_size, len(current_vals))
                avg_val = np.mean(current_vals[start_idx:end_idx])
                reduced_vals.append(avg_val)
            
            # Optimize the reduced version
            try:
                # Use a simple local optimization approach for the reduced version
                reduced_vals = np.array(reduced_vals)
                # Apply convex relaxation to get better guidance
                refined_reduced = solve_convex_relaxation(reduced_vals.tolist())
                refined_reduced = np.array(refined_reduced)
                
                # Interpolate back to full resolution
                if len(refined_reduced) < len(current_vals):
                    # Interpolate from reduced to full resolution
                    full_res = []
                    step_size = len(current_vals) // len(refined_reduced)
                    
                    for i, val in enumerate(refined_reduced):
                        for j in range(step_size):
                            if len(full_res) < len(current_vals):
                                full_res.append(val)
                    
                    # Pad if necessary
                    while len(full_res) < len(current_vals):
                        full_res.append(val)
                    
                    current_vals = full_res[:len(current_vals)]
                else:
                    current_vals = refined_reduced.tolist()
                    
            except:
                pass  # Skip if interpolation fails
                
        # Fine tune the current solution
        try:
            # Apply local optimization
            def objective(f_vals):
                return -calculate_c2(f_vals)
            
            # Simple gradient-free optimization for refinement
            from scipy.optimize import differential_evolution
            bounds = [(0, max(1e-6, 1000)) for _ in range(len(current_vals))]
            result = differential_evolution(objective, bounds, maxiter=50, seed=42)
            if result.success:
                current_vals = result.x.tolist()
        except:
            pass  # Skip if optimization fails
    
    return current_vals

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using sparse convex optimization approach."""
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Sparse geometric pattern initialization with convex relaxation
    try:
        n_steps = np.random.randint(500, 2000)  # Wider range for better exploration
        
        # Construct sparse geometric pattern
        initial_f_vals = construct_sparse_geometric_pattern(n_steps, alpha=0.7)
        
        # Apply convex relaxation for better initial guidance
        relaxed_vals = solve_convex_relaxation(initial_f_vals)
        
        # Apply multi-resolution optimization
        optimized_vals = multi_resolution_optimization(relaxed_vals)
        
        # Ensure non-negativity
        optimized_vals = np.maximum(optimized_vals, 0.0).tolist()
        
        final_c2 = calculate_c2(optimized_vals)
        if final_c2 > best_c2:
            best_c2 = final_c2
            best_solution = optimized_vals
            
    except Exception as e:
        pass

    # Strategy 2: Alternative pattern-based approach
    try:
        if best_solution is None:
            n_steps = np.random.randint(500, 2000)
            
            # Create a different geometric pattern with different exponent
            base_vals = np.geomspace(1, 0.001, num=n_steps // 2)
            base_vals = np.pad(base_vals, (0, n_steps - len(base_vals)), 'constant')
            
            # Normalize and add structure
            if np.sum(base_vals) > 0:
                base_vals = base_vals / np.sum(base_vals) * 100
            
            # Add concentrated mass at center
            center_pos = n_steps // 2
            if center_pos < n_steps:
                base_vals[center_pos] += 5.0
            
            # Apply convex relaxation as pre-processing
            relaxed_vals = solve_convex_relaxation(base_vals.tolist())
            
            # Multi-resolution optimization
            optimized_vals = multi_resolution_optimization(relaxed_vals)
            
            # Ensure non-negativity
            optimized_vals = np.maximum(optimized_vals, 0.0).tolist()
            
            final_c2 = calculate_c2(optimized_vals)
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_solution = optimized_vals
                
    except Exception as e:
        pass

    # Strategy 3: Hybrid approach with structured sparsity
    try:
        if best_solution is None:
            n_steps = np.random.randint(500, 2000)
            
            # Create highly structured sparse pattern
            sparse_pattern = np.zeros(n_steps)
            
            # Place several peaks at strategic locations
            peak_positions = [n_steps // 8, 3 * n_steps // 8, 5 * n_steps // 8, 7 * n_steps // 8]
            
            for i, pos in enumerate(peak_positions):
                if pos < n_steps:
                    # Create peaks with decreasing height
                    sparse_pattern[pos] = 10.0 * (1 - i * 0.2)
            
            # Add base geometric decay
            for i in range(n_steps):
                if sparse_pattern[i] == 0:  # Only modify non-peak positions
                    # Base geometric decay modified by proximity to peaks
                    min_dist = min([abs(i - pos) for pos in peak_positions if pos < n_steps])
                    decay_factor = np.exp(-min_dist / (n_steps // 8))
                    sparse_pattern[i] = 0.5 * decay_factor
            
            # Ensure non-negativity and normalize
            sparse_pattern = np.maximum(sparse_pattern, 0.0)
            if np.sum(sparse_pattern) > 0:
                sparse_pattern = sparse_pattern / np.sum(sparse_pattern) * 50
            
            # Apply convex relaxation
            relaxed_vals = solve_convex_relaxation(sparse_pattern.tolist())
            
            # Multi-resolution optimization
            optimized_vals = multi_resolution_optimization(relaxed_vals)
            
            # Ensure non-negativity
            optimized_vals = np.maximum(optimized_vals, 0.0).tolist()
            
            final_c2 = calculate_c2(optimized_vals)
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_solution = optimized_vals
                
    except Exception as e:
        pass

    # Fallback to simple geometric pattern if all strategies fail
    if best_solution is None:
        n_steps = 1000
        best_solution = construct_sparse_geometric_pattern(n_steps, alpha=0.8)

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")