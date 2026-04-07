# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple, Optional
import warnings

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = convolve_fft(a, a)
    else:
        b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def compute_gradient_and_hessian_approx(sequence: List[float], eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute approximate gradient and Hessian for a sequence using finite differences.
    """
    a = np.array(sequence)
    n = len(a)
    grad = np.zeros(n)
    hess = np.zeros((n, n))
    
    # Compute gradient
    base_c1, _ = compute_c1_constant(sequence)
    
    for i in range(n):
        perturbed = a.copy()
        perturbed[i] += eps
        _, inv_c1_perturbed = compute_c1_constant(perturbed.tolist())
        grad[i] = (base_c1 - inv_c1_perturbed) / eps
    
    # Compute Hessian (approximate)
    for i in range(n):
        for j in range(i, n):
            perturbed = a.copy()
            perturbed[i] += eps
            if i != j:
                perturbed[j] += eps
                
            _, inv_c1_perturbed = compute_c1_constant(perturbed.tolist())
            
            # Second order finite difference
            hess[i, j] = (base_c1 - 2*inv_c1_perturbed + inv_c1_perturbed) / (eps ** 2)
            if i != j:
                hess[j, i] = hess[i, j]
    
    return grad, hess

def curvature_aware_update(sequence: List[float], lr: float = 1e-3) -> Optional[List[float]]:
    """
    Perform a curvature-aware update on the sequence.
    """
    try:
        n = len(sequence)
        a = np.array(sequence)
        
        # Compute gradient and Hessian
        grad, hess = compute_gradient_and_hessian_approx(sequence)
        
        # Regularize Hessian to make it positive definite if needed
        try:
            eigvals = np.linalg.eigvals(hess)
            if np.all(eigvals >= -1e-8):
                reg_hess = hess
            else:
                reg_hess = hess + np.eye(n) * (-min(eigvals) + 1e-8)
        except:
            reg_hess = hess + np.eye(n) * 1e-8
            
        # Compute update: -lr * (H + I)^{-1} * grad
        try:
            # Use pseudo-inverse for numerical stability
            hess_reg_inv = np.linalg.pinv(reg_hess + np.eye(n) * 1e-6)
            update = -lr * hess_reg_inv @ grad
        except:
            # Fallback to basic gradient descent if matrix inversion fails
            update = -lr * grad
        
        # Apply update
        new_sequence = a + update
        
        # Ensure non-negativity
        new_sequence = np.maximum(new_sequence, 0)
        
        return new_sequence.tolist()
        
    except Exception as e:
        # On failure, fall back to simple gradient descent
        try:
            grad, _ = compute_gradient_and_hessian_approx(sequence)
            new_sequence = a - lr * grad
            new_sequence = np.maximum(new_sequence, 0)
            return new_sequence.tolist()
        except:
            return None

def adaptive_gradient_optimization(
    initial_sequence: List[float],
    max_iterations: int = 500,
    min_improvement: float = 1e-6,
    patience: int = 10
) -> List[float]:
    """
    Perform adaptive gradient optimization with curvature-aware updates.
    """
    sequence = np.array(initial_sequence).copy()
    best_sequence = sequence.copy()
    best_inv_c1 = 0.0
    last_improvement = 0
    lr = 1e-3
    momentum = 0.9
    
    v = np.zeros_like(sequence)
    
    for i in range(max_iterations):
        # Compute current C1 and 1/C1
        _, current_inv_c1 = compute_c1_constant(sequence.tolist())
        
        if current_inv_c1 > best_inv_c1:
            best_inv_c1 = current_inv_c1
            best_sequence = sequence.copy()
            last_improvement = i
            
        # Adaptive learning rate
        if i > 0 and i - last_improvement > patience:
            lr *= 0.5
            last_improvement = i
            
        # Compute gradient
        try:
            grad, _ = compute_gradient_and_hessian_approx(sequence.tolist())
        except:
            # Fall back to simpler gradient computation
            grad = np.zeros_like(sequence)
            eps = 1e-5
            base_inv_c1 = current_inv_c1
            for j in range(len(sequence)):
                perturbed = sequence.copy()
                perturbed[j] += eps
                _, inv_c1_perturbed = compute_c1_constant(perturbed.tolist())
                grad[j] = (base_inv_c1 - inv_c1_perturbed) / eps
        
        # Update with momentum and curvature
        v = momentum * v + (1 - momentum) * grad
        update = -lr * v
        
        # Add curvature correction
        try:
            _, hess = compute_gradient_and_hessian_approx(sequence.tolist())
            try:
                hess_inv = np.linalg.pinv(hess + np.eye(len(sequence)) * 1e-8)
                curvature_correction = lr * hess_inv @ grad
                update -= curvature_correction
            except:
                pass  # Skip curvature correction if it fails
        except:
            pass  # Skip curvature correction if there are issues
            
        # Apply update
        sequence = sequence + update
        
        # Ensure non-negativity and clipping
        sequence = np.maximum(sequence, 0)
        sequence = np.minimum(sequence, 1000)
        
        # Early stopping if improvement is negligible
        if i > 0 and abs(current_inv_c1 - best_inv_c1) < min_improvement:
            break
            
    return best_sequence.tolist()

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence using gradient-based approach."""
    start_time = time.time()
    
    best_sequence = None
    best_inv_c1 = -float('inf')
    
    # Try several different initialization strategies
    init_strategies = [
        lambda n: [random.random() * 100 for _ in range(n)],
        lambda n: [0.0] * n,
        lambda n: [abs(random.gauss(0, 1)) * 10 for _ in range(n)]
    ]
    
    # Different sizes to explore
    sizes = [100, 200, 300, 500, 750, 1000]
    
    for size in sizes:
        # Multiple random starts with different strategies
        for strategy in init_strategies:
            if time.time() - start_time > 170:  # Leave some room for final steps
                break
                
            try:
                # Initialize sequence
                sequence = strategy(size)
                
                # Ensure positivity and reasonable scale
                sequence = [max(x, 0.01) for x in sequence]
                
                # Try gradient-based optimization
                optimized = adaptive_gradient_optimization(sequence, max_iterations=200)
                
                # Evaluate result
                _, inv_c1 = compute_c1_constant(optimized)
                
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = optimized
                    
            except Exception as e:
                continue
                
        if time.time() - start_time > 170:
            break
    
    # If nothing was found, return default
    if best_sequence is None:
        best_sequence = [1.0] * 100
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")