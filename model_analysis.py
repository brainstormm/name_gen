import torch


def inspect_gradients(model):
    total_norm = 0
    print("\nDetailed Gradient Analysis:")
    print("-" * 50)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Get gradient data
            grad_data = param.grad.data
            
            # Calculate norm
            param_norm = grad_data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # Print detailed information
            print(f"\nParameter: {name}")
            print(f"Shape: {grad_data.shape}")
            print(f"Type: {grad_data.dtype}")
            print(f"Device: {grad_data.device}")
            print(f"Gradient Norm: {param_norm.item():.6f}")
            
            # Print some sample values
            if grad_data.numel() > 0:  # if tensor is not empty
                print("Sample values (first 5):")
                flat_grad = grad_data.flatten()
                print(flat_grad[:5].tolist())
            
            # Print statistics
            print(f"Mean: {grad_data.mean().item():.6f}")
            print(f"Std: {grad_data.std().item():.6f}")
            print(f"Min: {grad_data.min().item():.6f}")
            print(f"Max: {grad_data.max().item():.6f}")
            print("-" * 50)
    
    total_norm = total_norm**0.5
    print(f"\nTotal Gradient Norm: {total_norm:.6f}")
    return total_norm
