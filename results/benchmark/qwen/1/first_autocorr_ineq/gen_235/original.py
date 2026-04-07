# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from typing import List, Optional, Tuple

class AutocorrelationOptimizer:
    """
    An optimizer for finding step functions that maximize 1/C₁,
    thereby improving the upper bound on the first autocorrelation inequality constant.
    """
    
    def __init__(self, max_sequence_length: int = 1000, min_sequence_length: int = 100):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        
    def compute_convolution(self, sequence: np.ndarray) -> np.ndarray:
        """Compute the autoconvolution of the sequence efficiently using FFT."""
        # Use FFT-based convolution for better performance
        padded_seq = np.pad(sequence, (0, len(sequence) - 1), 'constant')
        conv_result = np.convolve(padded_seq, sequence, mode='valid')
        return conv_result
    
    def compute_c1(self, sequence: np.ndarray) -> float:
        """Compute C₁ constant for the given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')
        
        # Compute convolution
        conv = self.compute_convolution(sequence)
        
        # Calculate C₁
        max_conv = np.max(conv)
        sum_sq = np.sum(sequence) ** 2
        if sum_sq == 0:
            return float('inf')
            
        c1 = 2 * n * max_conv / sum_sq
        return c1
    
    def compute_inv_c1(self, sequence: np.ndarray) -> float:
        """Compute 1/C₁ for the given sequence."""
        c1 = self.compute_c1(sequence)
        if c1 == 0:
            return float('inf')
        return 1.0 / c1
    
    def solve_convolution_lp(self, f_sequence: np.ndarray, rhs: float) -> Optional[np.ndarray]:
        """Solve the convolution LP for a given sequence and RHS."""
        try:
            n = len(f_sequence)
            if n == 0:
                return None
                
            # Objective: minimize -sum(x) (i.e., maximize sum(x))
            c = -np.ones(n)
            
            # Build constraint matrix
            a_ub = []
            b_ub = []
            
            # Convolution constraints
            for k in range(2 * n - 1):
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_sequence[i]
                a_ub.append(row)
                b_ub.append(rhs)

            # Non-negativity constraints
            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)
            
            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])

            # Solve LP
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
            
            if result.success:
                return result.x
            else:
                return None
        except Exception as e:
            print(f'Error in LP solver: {e}')
            return None

    def get_good_direction_to_move_into(self, sequence: List[float]) -> Optional[List[float]]:
        """Get the direction to move into the sequence."""
        try:
            n = len(sequence)
            if n == 0:
                return None
                
            sequence = np.array(sequence)
            
            # Normalize for consistent scale
            sum_sequence = np.sum(sequence)
            if sum_sequence < 0.01:
                return None
                
            # Normalize sequence
            normalized_sequence = sequence * np.sqrt(2 * n) / sum_sequence
            
            # Compute the RHS for LP
            conv = self.compute_convolution(normalized_sequence)
            rhs = np.max(conv)
            
            # Solve LP
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
            
            if g_fun is None:
                return None
                
            # Normalize result back
            sum_g_fun = np.sum(g_fun)
            if sum_g_fun < 0.01:
                return None
            normalized_g_fun = g_fun * np.sqrt(2 * n) / sum_g_fun
            
            # Apply mixing
            t = 0.01
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
            
            return new_sequence
        except Exception as e:
            print(f'Error in get_good_direction_to_move_into: {e}')
            return None
            
    def generate_initial_sequence(self) -> List[float]:
        """Generate a good initial sequence."""
        n = np.random.randint(self.min_sequence_length, self.max_sequence_length)
        # Generate a sequence with some structure to avoid poor starts
        sequence = np.random.exponential(scale=1.0, size=n).tolist()
        # Normalize to have reasonable total weight
        total = sum(sequence)
        if total > 0:
            sequence = [x / total * 100 for x in sequence]
        return sequence

    def run_evolution(self, max_iterations: int = 100) -> Tuple[List[float], float]:
        """Run the evolutionary optimization process."""
        # Generate initial sequence
        best_sequence = self.generate_initial_sequence()
        best_inv_c1 = self.compute_inv_c1(best_sequence)
        
        print(f'Initial inv_C1: {best_inv_c1:.6f}')
        
        # Evolution loop
        for iteration in range(max_iterations):
            try:
                direction = self.get_good_direction_to_move_into(best_sequence)
                
                if direction is not None:
                    # Evaluate proposed direction
                    candidate_inv_c1 = self.compute_inv_c1(direction)
                    
                    if candidate_inv_c1 > best_inv_c1:
                        best_sequence = direction
                        best_inv_c1 = candidate_inv_c1
                        print(f'Iteration {iteration}: New best inv_C1: {best_inv_c1:.6f}')
                        
                        # Early exit if we beat the benchmark
                        if best_inv_c1 > 1.0 / 1.5031:
                            print(f'Beaten benchmark at iteration {iteration}')
                            break
                else:
                    # Fallback if direction is not computed
                    print(f'Iteration {iteration}: LP failed, using random perturbation')
                    new_sequence = self.generate_initial_sequence()
                    new_inv_c1 = self.compute_inv_c1(new_sequence)
                    if new_inv_c1 > best_inv_c1:
                        best_sequence = new_sequence
                        best_inv_c1 = new_inv_c1
                        
            except Exception as e:
                print(f'Error in evolution iteration {iteration}: {e}')
                # Perturb slightly
                if best_sequence:
                    index = np.random.randint(len(best_sequence))
                    best_sequence[index] = max(0, best_sequence[index] + np.random.normal(0, 0.5))
        
        return best_sequence, best_inv_c1

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer()
    best_sequence, _ = optimizer.run_evolution()
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
