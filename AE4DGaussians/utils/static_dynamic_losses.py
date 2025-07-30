"""
Loss functions for static/dynamic Gaussian classification
"""

import torch
import torch.nn.functional as F

def static_dynamic_classification_loss(classification_probs, target_threshold=0.1, sparsity_weight=0.01):
    """
    Loss function to encourage clear static/dynamic classification
    
    Args:
        classification_probs: [N, 1] tensor with values between 0 and 1
        target_threshold: threshold for considering a point as having clear classification
        sparsity_weight: weight for sparsity regularization
    
    Returns:
        Total classification loss
    """
    # Binary entropy loss: encourages values to be close to 0 or 1
    eps = 1e-8
    binary_entropy = -(classification_probs * torch.log(classification_probs + eps) + 
                      (1 - classification_probs) * torch.log(1 - classification_probs + eps))
    
    # Sparsity loss: encourages most points to be static (closer to 0)
    sparsity_loss = classification_probs.mean()
    
    return binary_entropy.mean() + sparsity_weight * sparsity_loss

def temporal_consistency_loss(classification_probs_t1, classification_probs_t2, consistency_weight=0.1):
    """
    Temporal consistency loss to encourage stable classification across time
    
    Args:
        classification_probs_t1: Classification probabilities at time t1
        classification_probs_t2: Classification probabilities at time t2  
        consistency_weight: Weight for consistency regularization
    
    Returns:
        Temporal consistency loss
    """
    if classification_probs_t1 is None or classification_probs_t2 is None:
        return torch.tensor(0.0, device="cuda")
    
    # L2 loss between classifications at different times
    consistency_loss = F.mse_loss(classification_probs_t1, classification_probs_t2)
    
    return consistency_weight * consistency_loss

def motion_based_supervision_loss(classification_probs, motion_magnitude, motion_threshold=0.01, supervision_weight=0.05):
    """
    Supervision loss based on actual motion magnitude
    Points with high motion should be classified as dynamic
    
    Args:
        classification_probs: [N, 1] predicted probabilities (0=static, 1=dynamic)
        motion_magnitude: [N, 1] actual motion magnitude for each point
        motion_threshold: threshold above which points should be dynamic
        supervision_weight: weight for supervision loss
    
    Returns:
        Motion-based supervision loss
    """
    if motion_magnitude is None:
        return torch.tensor(0.0, device="cuda")
    
    # Create pseudo ground truth labels based on motion
    pseudo_labels = (motion_magnitude > motion_threshold).float()
    
    # Binary cross entropy loss
    bce_loss = F.binary_cross_entropy(classification_probs, pseudo_labels)
    
    return supervision_weight * bce_loss
