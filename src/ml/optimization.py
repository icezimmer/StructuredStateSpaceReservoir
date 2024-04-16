from torch import optim


def setup_optimizer(model, lr, weight_decay):
    """
    S4 requires a specific optimizer setup.

    The parameters of the kernel in the S4 layer (A, B, C, dt eventually) typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_name = f"group {i}"
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
                             f"Optimizer for {group_name} parameters",
                             f"{len(g['params'])} tensors",
                         ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer
