"""Read-only gradient diagnostics; batch only same-device CUDA scalar transfers."""
import math

import torch


def _same_dense_cuda(gradients):
    if not gradients:
        return False
    device = gradients[0].device
    return device.type == 'cuda' and all(
        g.device == device and g.layout == torch.strided and g.is_floating_point()
        for g in gradients)


def gradients_finite(gradients):
    gradients = tuple(gradients)
    if _same_dense_cuda(gradients):
        flags = [torch.isfinite(g).all() for g in gradients]
        return all(torch.stack(flags).cpu().tolist())
    # Keep CPU/mixed-device/unsupported-layout short-circuit behavior.
    return all(torch.isfinite(g).all() for g in gradients)


def gradient_l2_norm(gradients):
    gradients = tuple(gradients)
    if _same_dense_cuda(gradients):
        norms = [torch.linalg.vector_norm(g.detach(), dtype=torch.float64)
                 for g in gradients]
        return math.hypot(*torch.stack(norms).cpu().tolist())
    # Per-tensor float64 reductions and original Python merge order are unchanged.
    return math.hypot(*(float(torch.linalg.vector_norm(g.detach(), dtype=torch.float64))
                        for g in gradients))
