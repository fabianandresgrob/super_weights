import torch
from typing import Optional, Union

def safe_matmul(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                result_device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Safely compute matrix multiplication between tensors on different devices.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor  
        result_device: Device for result ('cpu', 'cuda', 'auto')
        
    Returns:
        Result tensor on specified device
    """
    # Store original devices for reference
    device1 = tensor1.device
    device2 = tensor2.device
    
    # If both on same device, compute directly
    if device1 == device2:
        result = torch.matmul(tensor1, tensor2)
        return result.to(result_device) if result_device != 'auto' else result
    
    # Choose computation device (prefer GPU if available)
    if device1.type == 'cuda':
        compute_device = device1
    elif device2.type == 'cuda':
        compute_device = device2
    else:
        compute_device = device1
    
    # Move tensors to computation device (creates copies, doesn't modify originals)
    temp_tensor1 = tensor1.to(compute_device)
    temp_tensor2 = tensor2.to(compute_device)
    
    # Compute result
    result = torch.matmul(temp_tensor1, temp_tensor2)
    
    # Move result to desired device
    if result_device == 'auto':
        final_result = result
    else:
        final_result = result.to(result_device)
    
    # GPU memory cleanup
    if compute_device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return final_result

def safe_einsum(equation: str, *tensors, result_device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """Safe einsum for more complex tensor operations"""
    # Similar logic but for einsum operations
    pass