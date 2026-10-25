"""Explicit normalization construction for versioned deterministic FWT models."""
from torch import nn


def group_count(channels, maximum):
    if type(channels) is not int or channels < 1 or type(maximum) is not int or maximum < 1:
        raise ValueError('Channels and maximum groups must be positive integers')
    return next(value for value in range(min(channels, maximum), 0, -1) if channels % value == 0)


def convert_batch_to_group(module, maximum=8):
    """Keep BN affine Parameter objects and eps, remove only running statistics.

    Called after the ordinary architecture has been initialized, so the same
    seed retains its initial named parameters and does not draw additional RNG.
    GroupNorm constructor initialization is constant, not random.
    """
    group_count(1, maximum)
    if isinstance(module, nn.BatchNorm2d):
        converted = nn.GroupNorm(group_count(module.num_features, maximum),
            module.num_features, eps=module.eps, affine=module.affine)
        if module.affine:
            converted.weight, converted.bias = module.weight, module.bias
        converted.train(module.training)
        return converted
    for name, child in list(module.named_children()):
        setattr(module, name, convert_batch_to_group(child, maximum))
    return module
