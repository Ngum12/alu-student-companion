def enable_lightweight_mode():
    """Enable lightweight mode for resource-constrained environments"""
    import torch
    
    # Force CPU usage.
    device = "cpu"
    
    # Use smaller model precision
    torch_dtype = torch.float32
    
    # Disable gradient computation
    torch.set_grad_enabled(False)
    
    return {
        "mode": "lightweight",
        "device": device,
        "model_precision": "float32"
    }