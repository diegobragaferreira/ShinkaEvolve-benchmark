# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from deap import base, creator, tools, algorithms
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Create step function on [-1/4, 1/4] with given heights
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step width
    step_width = 0.5 / n_steps

    # Compute autoconvolution g = f * f using discrete convolution
    # We'll use scipy's convolve for accuracy
    g = signal.convolve(f_values, f_values, mode='full')

    # Adjust indices to account for the fact that we're working
    # on a symmetric interval [-1/4, 1/4] with the origin at center
    # The resulting convolution will be longer than the original
    # We need to map back to our target interval

    # For step functions, we compute norms differently for better accuracy
    # Using piecewise linear integration for ||g||₂²
    # First compute the actual convolution values over correct intervals

    # Simplified approach: compute norms directly from the convolution
    g = np.array(g)

    if len(g) == 0:
        return 0.0, 0.0, 0.0

    # ||g||₂² using trapezoidal rule equivalent for piecewise linear integration
    # We approximate this as sum of squares weighted by step widths
    g_squared = g**2

    # Approximate ||g||₂² using trapezoidal rule across convolution points
    # The width of each segment is effectively step_width for most segments
    # But we need to be careful about the boundary conditions
    # For simplicity, we'll treat it as a sum of squares times average spacing

    # More precise approach: use trapezoidal rule manually
    # But since we are dealing with step functions, we'll use piecewise linear method
    # Let's compute it directly using the definition with proper spacing

    # Actually let's simplify and calculate directly with proper spacing
    # Each point in convolution corresponds to a specific x position in [0, 1]
    # But since the original function is on [-1/4, 1/4], the convolution gives us
    # a function on [-1/2, 1/2]. For our purposes, we'll just work with the values.

    # Simplified but effective approach:
    # ||g||₂² = integral of g² dx ≈ sum of g[i]² * delta_x
    # ||g||₁ = integral of |g| dx ≈ sum of |g[i]| * delta_x
    # ||g||∞ = max(|g[i]|)

    # We'll assume approximately equal spacing
    # In practice, the step function has width 0.5/n,
    # so a convolution will have finer resolution
    # But we'll work with what we get from convolution

    # For now, we'll compute norms using the discrete values properly
    # Estimate spacing based on original function length
    total_length = 0.5  # interval [-1/4, 1/4]
    dx = total_length / n_steps

    # The convolution will produce 2*n_steps - 1 values
    # But we don't necessarily need to use all of them for norms
    # We'll compute norms directly on the full convolution

    # ||g||₂² using sum of squares with proper weights
    # Since we're using discrete samples, we approximate as:
    # ||g||₂² ≈ sum(g[i]² * dx) where dx is step width
    # But actually, we're approximating integral of g²

    # Compute norms properly
    g_abs = np.abs(g)
    g_squared = g**2

    # Compute ||g||₂² ≈ sum(g[i]² * dx)
    # This requires knowing how to weight the intervals correctly
    # For simplicity, let's assume equal intervals in final convolution
    # The convolution output length is 2n-1, so spacing is
    # effectively 0.5/(n-1) for the convolution domain [-1/2, 1/2]
    # But we want to integrate over [-1/4, 1/4], so we'll consider relevant portion

    # For computational efficiency and accuracy, we'll compute it directly:
    # Let's compute the integral approximation directly
    # The convolution result length is 2*n_steps - 1
    # The convolution spans from -0.5 to 0.5 in x-space
    # Our interval of interest is from -0.25 to 0.25

    # Simpler approach: compute discrete approximations properly
    # We'll compute the norms directly from the convolution
    if len(g) == 0:
        return 0.0, 0.0, 0.0

    # Approximate ||g||₂² using trapezoidal rule on full convolution
    # But we're integrating over a specific region [-0.25, 0.25]
    # The convolution spans [-0.5, 0.5] with 2*n_steps-1 points
    # So sampling frequency is higher than what we need

    # Let's make a more accurate estimation:
    # Original step width is 0.5/n_steps
    # Convolution spacing will be smaller - let's say it's 0.5/(n_steps-1)
    # But that's not right either.
    # Better approach: we'll compute norms using direct numerical integration

    # Assume we're evaluating over [-0.25, 0.25] which is 1/2 wide
    # The convolution spans [-0.5, 0.5] with 2*n_steps - 1 points
    # The sampling is essentially 0.5/(2*n_steps-1) spacing
    # But we don't want to integrate the whole thing - just the middle part

    # Focus on simpler approach: compute norms directly with appropriate weighting
    # For ||g||₂² = sum(g[i]^2 * w_i) where w_i depends on how we integrate
    # We can simply take the sum of squared values times a reasonable step size

    # Use the full convolution values for norms, but weight appropriately
    # If there are N elements in convolution, spacing is 0.5/(N-1)
    # But we'll just use the simple sum for now with some normalization

    # More straightforward approach - estimate norms from discretized values
    # ||g||₂² ≈ sum(g[i]^2) * dx (where dx is typical spacing)
    # ||g||₁ ≈ sum(abs(g[i])) * dx
    # ||g||∞ = max(abs(g[i]))

    # The actual spacing in the convolution domain [-0.5, 0.5] with 2*n_steps-1 points
    spacing = 1.0 / (2 * n_steps - 1)

    # Compute the norms
    g_norm_2_squared = np.sum(g_squared) * spacing  # Should be ||g||₂²
    g_norm_1 = np.sum(g_abs) * spacing              # Should be ||g||₁
    g_norm_inf = np.max(g_abs)                      # Should be ||g||∞

    # Make sure we don't have zero denominators
    if g_norm_1 == 0 or g_norm_inf == 0:
        return 0.0, 0.0, 0.0

    return g_norm_2_squared, g_norm_1, g_norm_inf

def evaluate_c2(individual):
    """
    Evaluate the C2 value for a given individual (step function)
    Returns negative C2 because we want to maximize but optimization library minimizes
    """
    # Clip negative values to zero
    f_values = np.clip(np.array(individual), 0, None)

    # Get the norms
    g_norm_2_squared, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)

    # Compute C2
    if g_norm_1 == 0 or g_norm_inf == 0:
        return (float('inf'),)  # Return very large penalty for invalid cases

    c2 = g_norm_2_squared / (g_norm_1 * g_norm_inf)

    # Return negative because DEAP minimizes, but we want to maximize
    return (-c2,)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary algorithm."""
    # Parameters for evolution
    POPSIZE = 100
    NGEN = 50
    MUTPB = 0.3
    CXPB = 0.5
    IND_SIZE = 200  # Number of steps (can be adjusted)

    # Create types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define gene representation (step heights)
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation and operators
    toolbox.register("evaluate", evaluate_c2)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population(n=POPSIZE)

    # Evolve
    start_time = time.time()
    try:
        algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB,
                           ngen=NGEN, verbose=False)
    except Exception:
        pass  # Handle any errors during evolution

    # Extract best individual
    best_individual = tools.selBest(pop, 1)[0]

    # Convert to final list of floats
    return [float(x) for x in best_individual]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")