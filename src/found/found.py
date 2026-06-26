import torch
import torch.nn as nn

def found_vgg13(model, img: torch.Tensor, eps: float = 0.07, steps: int = 10, lr: float = 0.01) -> torch.Tensor:
    """
    Args:
        model: Pytorch model
        img: tensor of shape (Batch, Channel, Height, Width), scaled [0, 1].
        eps: The maximum allowed pixel perturbation (L-infinity bounding box).
        steps: Number of optimization iterations
        lr: Step size for the gradient update.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    img = img.to(device)
    
    base_features = model.extract_features(img).detach()

    # Initialize delta (the adversarial watermark) as zeros
    delta = torch.zeros_like(img, requires_grad=True, device=device)

    # Optimizer to update delta perturbation 
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        perturbed_img = torch.clamp(img + delta, 0.0, 1.0)
        
        perturbed_features = model.extract_features(perturbed_img)
        
        # Negative sign because pytorch optimizers minimize loss
        loss = -torch.mean((base_features - perturbed_features) ** 2)
        
        loss.backward()
        optimizer.step()
        
        # limit constraint (L-infinity projection)
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
            
    final_disrupted_img = torch.clamp(img + delta, 0.0, 1.0)
    return final_disrupted_img.cpu()
    
