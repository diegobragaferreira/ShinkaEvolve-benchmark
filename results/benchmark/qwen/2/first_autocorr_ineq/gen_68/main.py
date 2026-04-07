# EVOLVE-BLOCK-START

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad

def compute_convolution_pytorch(seq1, seq2):
    """Compute convolution using PyTorch for efficient gradient computation."""
    # Convert to PyTorch tensors
    t1 = torch.tensor(seq1, dtype=torch.float32)
    t2 = torch.tensor(seq2, dtype=torch.float32)
    
    # Use PyTorch's convolution function (cross-correlation with flipped kernel)
    # For autoconvolution, we convolve seq1 with itself
    # We pad to avoid circular convolution effects
    padding = len(seq1) - 1
    conv_result = torch.nn.functional.conv1d(
        t1.view(1, 1, -1), 
        t2.view(1, 1, -1), 
        padding=padding
    ).squeeze()
    
    # Return the valid part (non-padded)
    return conv_result[:len(seq1)].tolist()

def compute_c1_pytorch(sequence):
    """Compute C1 value for a given sequence using PyTorch."""
    n = len(sequence)
    if n < 1:
        return float('inf')
    
    # Convert to tensor
    seq_tensor = torch.tensor(sequence, dtype=torch.float32, requires_grad=True)
    
    # Compute sum
    sum_a = seq_tensor.sum()
    
    if sum_a < 1e-10:
        return float('inf')
    
    # Compute autoconvolution using PyTorch
    conv_result = torch.nn.functional.conv1d(
        seq_tensor.view(1, 1, -1), 
        seq_tensor.view(1, 1, -1), 
        padding=n-1
    ).squeeze()[:n]
    
    # Get max convolution value
    max_conv = conv_result.max()
    
    # Compute C1
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1.item()

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Optimizes sequence using gradient-based approach with adaptive learning rate."""
    try:
        n = len(sequence)
        if n < 1:
            return None
            
        # Convert to PyTorch tensor with gradient tracking
        seq_tensor = torch.tensor(sequence, dtype=torch.float32, requires_grad=True)
        
        # Set up optimizer with adaptive learning rate
        optimizer = optim.Adam([seq_tensor], lr=0.01)
        
        # Number of optimization steps
        steps = 50
        
        # Optimize over several steps with decreasing learning rate
        for i in range(steps):
            optimizer.zero_grad()
            
            # Compute sum and autoconvolution
            sum_a = seq_tensor.sum()
            if sum_a < 1e-10:
                return None
                
            # Compute autoconvolution
            conv_result = torch.nn.functional.conv1d(
                seq_tensor.view(1, 1, -1), 
                seq_tensor.view(1, 1, -1), 
                padding=n-1
            ).squeeze()[:n]
            
            # Max convolution value
            max_conv = conv_result.max()
            
            # Compute C1
            c1 = 2 * n * max_conv / (sum_a ** 2)
            
            # We want to maximize 1/C1, so minimize -1/C1
            loss = -1.0 / c1
            
            # Compute gradients
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Ensure non-negativity
            with torch.no_grad():
                seq_tensor.clamp_(min=0)
                
            # Adaptive learning rate decay
            if i % 10 == 0 and i > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
        
        # Convert back to list
        optimized_sequence = seq_tensor.detach().numpy().tolist()
        
        return optimized_sequence
    
    except Exception as e:
        # In case of failure, return original sequence
        return sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence using gradient descent."""
    # Initialize with a random sequence
    n = np.random.randint(100, 1000)
    init_sequence = [np.random.rand() for _ in range(n)]
    
    # Normalize to ensure positive values  
    init_sum = sum(init_sequence)
    if init_sum < 0.01:
        init_sequence[0] = 0.1
        
    # Perform gradient-based optimization
    best_sequence = get_good_direction_to_move_into(init_sequence)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
