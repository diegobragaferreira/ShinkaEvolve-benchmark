# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple, Optional

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

class SequenceEvaluator:
    """Handles sequence evaluation and C1 constant computation."""
    
    @staticmethod
    def compute_c1_constant(sequence: List[float]) -> float:
        """Computes the C1 constant for a given sequence."""
        a = np.array(sequence)
        sum_a = np.sum(a)
        if sum_a < 0.01:
            return float('inf')
        
        conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]
        max_conv = np.max(conv)
        n = len(a)
        c1 = (2 * n * max_conv) / (sum_a ** 2)
        return c1

    @classmethod
    def evaluate_sequence(cls, sequence: List[float]) -> float:
        """Evaluates a sequence and returns 1/C1 (the objective to maximize)."""
        try:
            c1 = cls.compute_c1_constant(sequence)
            if c1 == float('inf'):
                return 0.0
            return 1.0 / c1
        except Exception:
            return 0.0

class StepFunctionGenerator:
    """Generates step function sequences with various configurations."""
    
    @staticmethod
    def generate_step_function(n_steps: int, heights: Optional[List[float]] = None, 
                              max_height: float = 1000.0, total_length: int = 1000) -> List[float]:
        """Generate a step function with n_steps and optional specific heights."""
        if heights is None:
            heights = [random.uniform(0, max_height) for _ in range(n_steps)]
        else:
            heights = heights[:n_steps] + [0.0] * (n_steps - len(heights))
            heights = heights[:n_steps]

        # Create step function with equal-length steps
        step_sizes = [total_length // n_steps] * n_steps
        remainder = total_length % n_steps
        for i in range(remainder):
            step_sizes[i] += 1

        # Build the sequence
        sequence = []
        for i, (height, size) in enumerate(zip(heights, step_sizes)):
            sequence.extend([height] * size)

        # Trim or pad to exact length
        if len(sequence) > total_length:
            sequence = sequence[:total_length]
        elif len(sequence) < total_length:
            sequence.extend([0.0] * (total_length - len(sequence)))

        return sequence

    @classmethod
    def generate_predefined_patterns(cls, n_steps: int) -> List[List[float]]:
        """Generate predefined height patterns that often work well."""
        patterns = [
            [1000 * (0.9 ** i) for i in range(n_steps)],  # Exponential decay
            [1000 * (1.0 / (i + 1)) for i in range(n_steps)],  # Harmonic decay
            [1000] * n_steps,  # Constant
            [1000 * (i / (n_steps - 1)) if n_steps > 1 else 1000 for i in range(n_steps)]  # Linear increase
        ]
        return patterns

class AdaptiveOptimizer:
    """Performs adaptive optimization with dynamic parameters."""
    
    @staticmethod
    def quadratic_optimization_approach(initial_sequence: List[float], max_iter: int = 1000) -> List[float]:
        """Use quadratic programming to directly optimize the 1/C1 objective."""
        n = len(initial_sequence)
        x0 = np.array(initial_sequence)

        def objective(x):
            x = np.maximum(x, 0)
            conv = fftconvolve(x, x, mode='full')[:len(x)*2-1]
            max_conv = np.max(conv)
            sum_x = np.sum(x)
            if sum_x < 0.01:
                return 1e6
            c1 = (2 * len(x) * max_conv) / (sum_x ** 2)
            return -1.0 / c1  # Negative because we want to maximize 1/C1

        bounds = [(0, 1000) for _ in range(n)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-9})

        return result.x.tolist() if result.success else initial_sequence

    @classmethod
    def adaptive_local_search(cls, initial_sequence: List[float], max_iter: int = 200) -> Tuple[List[float], float]:
        """Enhanced local search with adaptive mutation rates and dynamic step sizes."""
        current_sequence = initial_sequence.copy()
        current_fitness = SequenceEvaluator.evaluate_sequence(current_sequence)

        # Adaptive parameters
        mutation_rate = 0.3
        max_mutation = 0.5
        cooling_factor = 0.99

        for i in range(max_iter):
            # Adaptively decrease mutation rate
            mutation_rate *= cooling_factor
            max_mutation *= cooling_factor

            # Mutate current sequence
            mutated = [x * random.uniform(1 - max_mutation, 1 + max_mutation) for x in current_sequence]
            mutated = [max(0, min(1000, x)) for x in mutated]

            mutated_fitness = SequenceEvaluator.evaluate_sequence(mutated)
            if mutated_fitness > current_fitness:
                current_sequence = mutated
                current_fitness = mutated_fitness

            # Occasionally introduce more drastic changes
            if random.random() < 0.1:
                # Large mutation
                mutated_large = [x * random.uniform(0.1, 10.0) for x in current_sequence]
                mutated_large = [max(0, min(1000, x)) for x in mutated_large]
                mutated_large_fitness = SequenceEvaluator.evaluate_sequence(mutated_large)
                if mutated_large_fitness > current_fitness:
                    current_sequence = mutated_large
                    current_fitness = mutated_large_fitness

        return current_sequence, current_fitness

class AdaptiveStepOptimizer:
    """Main optimization class that orchestrates the search process."""
    
    def __init__(self):
        self.evaluator = SequenceEvaluator()
        self.generator = StepFunctionGenerator()
        self.optimizer = AdaptiveOptimizer()
        
    def domain_specific_optimization(self, max_time_seconds: int = 180) -> Tuple[List[float], float]:
        """Specialized approach focusing on step function domains."""
        start_time = time.time()
        best_sequence = None
        best_inv_c1 = 0.0

        # Predefined step counts to explore
        step_counts = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
        
        # Try various combinations of step counts and height patterns
        for attempt in range(30):  # More attempts to ensure thorough search
            if time.time() - start_time > max_time_seconds:
                break

            n_steps = random.choice(step_counts)
            patterns = self.generator.generate_predefined_patterns(n_steps)
            heights = random.choice(patterns)

            # Generate step function using a pattern
            sequence = self.generator.generate_step_function(n_steps, heights=heights)

            # Optimize using quadratic programming
            optimized_sequence = self.optimizer.quadratic_optimization_approach(sequence, 150)

            # Evaluate the optimized sequence
            inv_c1 = self.evaluator.evaluate_sequence(optimized_sequence)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_sequence

        return best_sequence, best_inv_c1

    def search_for_best_sequence(self) -> List[float]:
        """Main search function implementing the new optimization strategy."""
        start_time = time.time()
        best_sequence = None
        best_inv_c1 = 0.0

        # Phase 1: Domain-specific step function optimization
        step_seq, step_fitness = self.domain_specific_optimization(150)
        if step_fitness > best_inv_c1:
            best_inv_c1 = step_fitness
            best_sequence = step_seq

        # Phase 2: Refinement using adaptive local search
        if best_sequence is not None:
            refined_seq, refined_fitness = self.optimizer.adaptive_local_search(best_sequence, 100)
            if refined_fitness > best_inv_c1:
                best_inv_c1 = refined_fitness
                best_sequence = refined_seq

        # Phase 3: Fallback to generic optimization if needed
        if best_sequence is None or time.time() - start_time > 170:
            initial_sequence = self.generator.generate_step_function(random.randint(10, 100))
            optimized_seq = self.optimizer.quadratic_optimization_approach(initial_sequence)
            optimized_fitness = self.evaluator.evaluate_sequence(optimized_seq)
            if optimized_fitness > best_inv_c1:
                best_inv_c1 = optimized_fitness
                best_sequence = optimized_seq

        # Final check with adaptive local search
        if best_sequence is not None:
            final_seq, final_fitness = self.optimizer.adaptive_local_search(best_sequence, 100)
            if final_fitness > best_inv_c1:
                best_inv_c1 = final_fitness
                best_sequence = final_seq

        return best_sequence if best_sequence is not None else self.generator.generate_step_function(50)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")