import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from functools import partial


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))

def combine(tree, operator, a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat

def _scan(tree, operator, elems, axis):
    """Perform scan on `elems`."""
    num_elems = elems[0].shape[axis]

    if num_elems < 2:
        return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(tree, operator,
                            [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
                            [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems])

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(tree, operator, reduced_elems, axis)

    if num_elems % 2 == 0:
        even_elems = combine(tree, operator,
                             [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])
    else:
        even_elems = combine(tree, operator,
                             odd_elems,
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
        torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
        if result.shape.numel() > 0 and elem.shape[axis] > 0 else
        result if result.shape.numel() > 0 else
        torch.ops.aten.slice(elem, axis, 0, 1)  # Jax allows/ignores concat with 0-dim, Pytorch does not
        for (elem, result) in zip(elems, even_elems)]

    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

# Pytorch impl. of jax.lax.associative_scan
def associative_scan(operator, elems, axis=0, reverse= False):
    # if not callable(operator):
    #     raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    assert axis >= 0 or axis < elems_flat[0].ndim, "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    scans = _scan(tree, operator, elems_flat, axis)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(scans, tree)

# @torch.jit.script
def _interleave(a, b, axis: int):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    b_trunc = (a.shape[axis] == b.shape[axis] + 1)
    if b_trunc:
        pad = [0, 0] * b.ndim
        pad[(b.ndim - axis - 1) * 2 + 1] = 1  # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        # TODO: find torch alternative for slice_along axis for torch.jit.script to work
        interleaved = torch.ops.aten.slice(interleaved, axis, 0, b.shape[axis] + a.shape[axis] - 1)
    return interleaved
