# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import convolve as fft_convolve
import random
import time

class AutocorrelationOptimizer:
    def __init__(self):
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        
    def compute_convolution(self, sequence):
        """Compute convolution using FFT for efficiency"""
        # Ensure sequence is numpy array
        seq = np.array(sequence)
        
        # For small sequences, use direct convolution to avoid FFT overhead
        if len(seq) < 100:
            conv_result = np.convolve(seq, seq, mode='full')
        else:
            # Use FFT-based convolution for larger sequences
            conv_result = fft_convolve(seq, seq, mode='full')
            
        # Return only the relevant part (middle section)
        middle = len(conv_result) // 2
        return conv_result[middle:]
    
    def compute_c1_constant(self, sequence):
        """Calculate C1 constant from sequence"""
        if len(sequence) == 0:
            return float('inf')
            
        sum_a = np.sum(sequence)
        if sum_a < 0.01:
            return float('inf')  # Reject sequences with very small sums
            
        conv_result = self.compute_convolution(sequence)
        max_conv = np.max(conv_result)
        
        if max_conv <= 0:
            return float('inf')
            
        n = len(sequence)
        c1 = 2 * n * max_conv / (sum_a ** 2)
        return c1
    
    def compute_inverse_c1(self, sequence):
        """Calculate inverse of C1 constant (what we want to maximize)"""
        c1 = self.compute_c1_constant(sequence)
        if c1 == float('inf'):
            return 0.0
        return 1.0 / c1
    
    def generate_random_sequence(self, min_length=100, max_length=1000):
        """Generate a random valid sequence"""
        length = random.randint(min_length, max_length)
        # Generate sequence with some randomness
        sequence = [random.uniform(0.1, 10.0) for _ in range(length)]
        return sequence
    
    def solve_convolution_lp(self, f_sequence, rhs):
        """Solve linear programming problem for convolution constraint"""
        try:
            n = len(f_sequence)
            c = -np.ones(n)
            
            # Build constraint matrix
            a_ub = []
            b_ub = []
            
            # Create convolution constraints
            for k in range(2 * n - 1):
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_sequence[i]
                a_ub.append(row)
                b_ub.append(rhs)
            
            # Add non-negativity constraints
            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)
            
            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])
            
            # Solve optimization
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
            
            if result.success:
                return result.x
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_good_direction_to_move_into(self, sequence):
        """Get optimized direction for improvement"""
        try:
            n = len(sequence)
            sum_sequence = np.sum(sequence)
            
            # Avoid division by zero
            if sum_sequence < 0.01:
                sum_sequence = 0.01
                
            # Normalize sequence
            normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
            
            # Compute RHS for LP
            conv_result = self.compute_convolution(normalized_sequence)
            rhs = np.max(conv_result)
            
            # Solve LP
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
            
            if g_fun is None or np.any(np.isnan(g_fun)):
                return None
                
            # Normalize result
            sum_g = np.sum(g_fun)
            if sum_g < 0.01:
                sum_g = 0.01
                
            normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
            
            # Mix original and new sequence
            t = 0.01
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
            
            return new_sequence
            
        except Exception as e:
            return None
    
    def optimize_step(self):
        """Perform one optimization step"""
        # Start with a random sequence
        if self.best_sequence is None:
            self.best_sequence = self.generate_random_sequence()
            
        # Try to improve current sequence
        improved_sequence = self.get_good_direction_to_move_into(self.best_sequence)
        
        if improved_sequence is not None:
            # Evaluate improvement
            old_inv_c1 = self.compute_inverse_c1(self.best_sequence)
            new_inv_c1 = self.compute_inverse_c1(improved_sequence)
            
            if new_inv_c1 > old_inv_c1:
                self.best_sequence = improved_sequence
                return True
                
        return False
    
    def run_optimization(self, max_iterations=1000, timeout_seconds=180):
        """Run the main optimization loop"""
        start_time = time.time()
        
        for iteration in range(max_iterations):
            if time.time() - start_time > timeout_seconds:
                break
                
            if not self.optimize_step():
                # If no improvement, generate a new random sequence occasionally
                if iteration % 50 == 0:
                    self.best_sequence = self.generate_random_sequence()
                    
        return self.best_sequence

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence"""
    optimizer = AutocorrelationOptimizer()
    best_sequence = optimizer.run_optimization()
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
