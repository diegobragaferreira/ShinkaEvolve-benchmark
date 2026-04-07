# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit, njit
import random
import time
from scipy.sparse import csr_matrix
import heapq
from collections import defaultdict
import math

# Global constants
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.8
CROSSOVER_PROB = 0.7
NUM_STARTS = 3
MAX_EVALUATIONS = 8000

class SparseAutoconvolutionComputer:
    """Efficient sparse representation of autoconvolution computations"""
    
    def __init__(self, n_points):
        self.n_points = n_points
        self.sparse_threshold = 0.01
        
    def compute_sparse_autoconvolution(self, f_vals):
        """Compute autoconvolution using sparse representation"""
        n = len(f_vals)
        if n == 0:
            return np.array([])
            
        # Create sparse representation focusing on significant contributions
        sparse_pairs = defaultdict(float)
        
        # Only compute non-trivial convolutions for significant values
        for i in range(n):
            if f_vals[i] > self.sparse_threshold:
                for j in range(n):
                    if f_vals[j] > self.sparse_threshold:
                        idx = i + j
                        if 0 <= idx < 2*n - 1:
                            sparse_pairs[idx] += f_vals[i] * f_vals[j]
        
        # Convert to dense array with only relevant entries
        g = np.zeros(2*n - 1)
        for idx, val in sparse_pairs.items():
            g[idx] = val
            
        return g
    
    def compute_autoconvolution_norms_sparse(self, f_vals):
        """Compute norms using sparse convolution with optimized integration"""
        g = self.compute_sparse_autoconvolution(f_vals)
        
        if len(g) == 0:
            return 0.0, 0.0, 0.0
        
        # Optimize L2^2 using trapezoidal integration on sparse data
        g_abs = np.abs(g)
        norm_l2_sq = 0.0
        
        # For sparse data, compute meaningful segments
        nonzero_indices = np.nonzero(g_abs)[0]
        if len(nonzero_indices) >= 2:
            # Use trapezoidal rule on significant segments
            for i in range(len(nonzero_indices) - 1):
                idx1, idx2 = nonzero_indices[i], nonzero_indices[i+1]
                if idx2 - idx1 == 1:  # Adjacent points
                    y1, y2 = g_abs[idx1], g_abs[idx2]
                    # Trapezoidal area
                    norm_l2_sq += (y1 + y2) * (idx2 - idx1) / 2.0
                else:
                    # Interpolate for non-adjacent points
                    y1, y2 = g_abs[idx1], g_abs[idx2]
                    # Approximate integral using rectangular method for gaps
                    norm_l2_sq += y1 * (idx2 - idx1)  # Simplified approximation
            
            # Add squared endpoint terms
            if len(nonzero_indices) > 0:
                norm_l2_sq += g_abs[nonzero_indices[0]]**2 / 2.0
                norm_l2_sq += g_abs[nonzero_indices[-1]]**2 / 2.0
        elif len(nonzero_indices) == 1:
            norm_l2_sq = g_abs[nonzero_indices[0]]**2
        else:
            norm_l2_sq = 0.0
            
        # L1 norm with proper normalization
        norm_l1 = np.sum(g_abs) / (len(g) + 1) if len(g) > 0 else 1e-15
        
        # Infinity norm
        norm_inf = np.max(g_abs) if len(g_abs) > 0 else 1e-15
        
        return norm_l2_sq, norm_l1, norm_inf

@njit
def compute_c2_score_sparse(f_vals, sparse_computer):
    """
    Compute the C2 score using sparse autoconvolution
    """
    norm_l2_sq, norm_l1, norm_inf = sparse_computer.compute_autoconvolution_norms_sparse(f_vals)
    
    # Avoid division by zero
    if norm_l1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    return norm_l2_sq / (norm_l1 * norm_inf)

def generate_adaptive_initialization(n):
    """Generate initial population with adaptive pattern diversity"""
    patterns = []
    
    # Pattern 1: Multi-peak with varying heights and widths
    x = np.linspace(-1, 1, n)
    pattern1 = np.zeros(n)
    num_peaks = np.random.randint(2, 6)
    for i in range(num_peaks):
        center = np.random.uniform(-0.8, 0.8)
        width = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.5, 1.5)
        pattern1 += height * np.exp(-((x - center)**2) / (2 * width**2))
    patterns.append(pattern1.tolist())
    
    # Pattern 2: Block-like structure with varying block sizes
    pattern2 = np.zeros(n)
    num_blocks = np.random.randint(3, 8)
    for i in range(num_blocks):
        start = int(i * n / num_blocks)
        end = int((i + 1) * n / num_blocks)
        height = np.random.uniform(0.3, 1.2)
        pattern2[start:end] = height
    patterns.append(pattern2.tolist())
    
    # Pattern 3: Linear trend with noise
    pattern3 = np.linspace(0.2, 1.0, n) + np.random.normal(0, 0.1, n)
    pattern3 = np.maximum(pattern3, 0)
    patterns.append(pattern3.tolist())
    
    # Pattern 4: Exponential decay with pulse
    pattern4 = np.zeros(n)
    center = n // 2
    for i in range(n):
        distance = abs(i - center) / (n // 2)
        pattern4[i] = np.exp(-distance)
    # Add a pulse at random position
    pulse_pos = np.random.randint(0, n)
    pattern4[pulse_pos] = np.random.uniform(0.5, 1.5)
    patterns.append(pattern4.tolist())
    
    # Pattern 5: Sine wave with amplitude modulation
    pattern5 = np.sin(2 * np.pi * x * 3) + 0.5 * np.sin(2 * np.pi * x * 8)
    pattern5 = 0.5 + 0.5 * pattern5
    pattern5 = np.maximum(pattern5, 0)
    patterns.append(pattern5.tolist())
    
    # Select best performing pattern
    best_pattern = patterns[0]
    best_score = -1.0
    
    sparse_computer = SparseAutoconvolutionComputer(n)
    
    for p in patterns:
        try:
            score = compute_c2_score_sparse(p, sparse_computer)
            if score > best_score:
                best_score = score
                best_pattern = p
        except:
            continue
    
    return best_pattern

class AdaptiveParticleSwarm:
    """Hybrid particle swarm optimization with adaptive neighborhoods"""
    
    def __init__(self, n_dimensions, n_particles=30, max_iterations=50):
        self.n_dimensions = n_dimensions
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        
    def optimize(self, objective_func):
        """Run adaptive particle swarm optimization"""
        # Initialize particles
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_scores = []
        
        for _ in range(self.n_particles):
            # Initialize with diverse patterns
            pos = [np.random.random() * 2.0 for _ in range(self.n_dimensions)]
            particles.append(pos)
            velocities.append([0.0] * self.n_dimensions)
            personal_best_positions.append(pos.copy())
            personal_best_scores.append(objective_func(pos))
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # Adaptive neighborhood setup
        def get_neighbors(idx, iteration):
            # Dynamic neighborhood size based on iteration
            neighborhood_size = max(3, int(self.n_particles * (1 - iteration / self.max_iterations)))
            neighbors = [idx]
            # Add nearby particles
            for i in range(1, neighborhood_size):
                if idx + i < self.n_particles:
                    neighbors.append(idx + i)
                if idx - i >= 0:
                    neighbors.append(idx - i)
            return neighbors
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Adaptive inertia weight
            self.inertia_weight = 0.9 - (0.5 * iteration / self.max_iterations)
            
            for i in range(self.n_particles):
                # Get neighbors
                neighbors = get_neighbors(i, iteration)
                
                # Find best neighbor
                best_neighbor_idx = neighbors[0]
                best_neighbor_score = personal_best_scores[neighbors[0]]
                for n_idx in neighbors[1:]:
                    if personal_best_scores[n_idx] < best_neighbor_score:
                        best_neighbor_score = personal_best_scores[n_idx]
                        best_neighbor_idx = n_idx
                
                # Update velocity and position for particle i
                for d in range(self.n_dimensions):
                    r1, r2 = np.random.random(), np.random.random()
                    
                    # Cognitive component
                    cognitive = self.cognitive_coeff * r1 * (
                        personal_best_positions[i][d] - particles[i][d]
                    )
                    
                    # Social component (best in neighborhood)
                    social = self.social_coeff * r2 * (
                        personal_best_positions[best_neighbor_idx][d] - particles[i][d]
                    )
                    
                    # Velocity update
                    velocities[i][d] = (
                        self.inertia_weight * velocities[i][d] +
                        cognitive +
                        social
                    )
                    
                    # Position update
                    particles[i][d] = particles[i][d] + velocities[i][d]
                    particles[i][d] = max(0.0, min(2.0, particles[i][d]))  # Bounds
                
                # Evaluate new position
                new_score = objective_func(particles[i])
                
                # Update personal best
                if new_score < personal_best_scores[i]:  # Minimizing
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if new_score < global_best_score:
                        global_best_score = new_score
                        global_best_position = particles[i].copy()
        
        return global_best_position

def adaptive_evolutionary_optimization():
    """Enhanced evolutionary optimization with adaptive strategies"""
    best_c2 = float('inf')
    best_f = None
    
    # Try multiple random starts
    for start_num in range(NUM_STARTS):
        # Initialize population with adaptive initialization
        population = []
        pop_size = POPULATION_SIZE
        
        for i in range(pop_size):
            n = random.randint(200, 1500)  # Adaptive sizing
            individual = generate_adaptive_initialization(n)
            # Ensure non-negative values
            individual = [max(0.0, x) for x in individual]
            population.append(individual)
        
        # Create objective function wrapper for optimization
        def objective_wrapper(x):
            # Convert to list of floats
            f_vals = [float(xi) for xi in x]
            try:
                sparse_computer = SparseAutoconvolutionComputer(len(f_vals))
                c2 = compute_c2_score_sparse(f_vals, sparse_computer)
                return c2  # We maximize, so return positive
            except:
                return 1e10  # Penalty for invalid solutions
        
        # Run particle swarm optimization on the best individuals
        try:
            # Select top performers from initial population
            scores = []
            for ind in population:
                sparse_computer = SparseAutoconvolutionComputer(len(ind))
                try:
                    score = compute_c2_score_sparse(ind, sparse_computer)
                    scores.append(score)
                except:
                    scores.append(1e10)
            
            # Sort by performance
            sorted_indices = np.argsort(scores)
            top_individuals = [population[i] for i in sorted_indices[:pop_size//2]]
            
            # Run PSO on top individuals
            if len(top_individuals) > 0:
                # Use first top individual as reference for dimensions
                ref_individual = top_individuals[0]
                swarm = AdaptiveParticleSwarm(len(ref_individual), 20, 30)
                pso_result = swarm.optimize(objective_wrapper)
                
                # Convert to correct format
                final_solution = [max(0.0, float(x)) for x in pso_result]
                final_c2 = objective_wrapper(final_solution)
                
                if final_c2 < best_c2:
                    best_c2 = final_c2
                    best_f = final_solution
                    
        except Exception as e:
            continue
    
    # If we didn't find a good solution, fallback to standard approach
    if best_f is None:
        # Use simple evolutionary approach
        try:
            population = [generate_adaptive_initialization(500) for _ in range(POPULATION_SIZE)]
            
            # Create objective function
            def objective(x):
                f_vals = [float(xi) for xi in x]
                try:
                    sparse_computer = SparseAutoconvolutionComputer(len(f_vals))
                    c2 = compute_c2_score_sparse(f_vals, sparse_computer)
                    return c2  # We maximize
                except:
                    return 1e10
            
            # Run differential evolution
            bounds = [(0.0, 2.0) for _ in range(len(population[0]))]
            result = differential_evolution(
                objective,
                bounds,
                maxiter=GENERATIONS,
                popsize=POPULATION_SIZE,
                mutation=MUTATION_RATE,
                recombination=CROSSOVER_PROB,
                seed=42,
                disp=False
            )
            
            if result.success:
                final_solution = [max(0.0, float(x)) for x in result.x]
                sparse_computer = SparseAutoconvolutionComputer(len(final_solution))
                final_c2 = compute_c2_score_sparse(final_solution, sparse_computer)
                
                if final_c2 < best_c2:
                    best_c2 = final_c2
                    best_f = final_solution
                    
        except Exception as e:
            pass
    
    # Apply refinement to best solution
    if best_f is not None:
        try:
            # Use local optimization with trust region method
            bounds = [(0.0, 2.0) for _ in range(len(best_f))]
            
            def local_objective(x):
                x = [max(0.0, xi) for xi in x]
                try:
                    sparse_computer = SparseAutoconvolutionComputer(len(x))
                    c2 = compute_c2_score_sparse(x, sparse_computer)
                    return c2
                except:
                    return 1e10
            
            local_result = minimize(
                local_objective,
                best_f,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            refined_solution = [max(0.0, float(x)) for x in local_result.x]
            sparse_computer = SparseAutoconvolutionComputer(len(refined_solution))
            refined_c2 = compute_c2_score_sparse(refined_solution, sparse_computer)
            
            if refined_c2 < best_c2:
                best_c2 = refined_c2
                best_f = refined_solution
                
        except Exception as e:
            pass
    
    return best_f if best_f is not None else generate_adaptive_initialization(500)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run the optimization
    start_time = time.time()
    try:
        f_values = adaptive_evolutionary_optimization()
        elapsed_time = time.time() - start_time
        # Add safety check to ensure valid output
        if f_values is None or len(f_values) == 0:
            # Fallback to simple uniform distribution
            f_values = [0.5] * 200
    except:
        # Final fallback
        f_values = [0.5] * 200
    
    # Ensure we return a reasonable-sized list
    if len(f_values) < 50:
        f_values = f_values + [0.5] * (50 - len(f_values))
    elif len(f_values) > 10000:
        f_values = f_values[:10000]
    
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
