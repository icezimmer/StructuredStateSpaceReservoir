import torch
from structured_kernels import cauchy_mult_sym_fwd, cauchy_mult_sym_bwd


def _cauchy_mult(v, z, w):
    return CauchyMultiplySymmetric.apply(v, z, w)


def cauchy_mult(v, z, w):
    """ Wrap the cuda method to deal with shapes """
    v, w = torch.broadcast_tensors(v, w)
    shape = v.shape
    # z_shape = z.shape
    # z = z.squeeze()
    assert len(z.shape) == 1

    v = v.contiguous()
    w = w.contiguous()
    z = z.contiguous()

    N = v.size(-1)
    assert w.size(-1) == N
    y = _cauchy_mult(v.view(-1, N), z, w.view(-1, N))
    y = y.view(*shape[:-1], z.size(-1))
    return y


class CauchyMultiplySymmetric(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, z, w):
        batch, N = v.shape
        supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        L = z.shape[-1]
        if not N in supported_N_values:
            raise NotImplementedError(f'Only support N values in {supported_N_values}')
        max_L_value = 32 * 1024 * 64 * 1024
        if L > max_L_value:
            raise NotImplementedError(f'Only support L values <= {max_L_value}')
        if not (v.is_cuda and z.is_cuda and w.is_cuda):
            raise NotImplementedError(f'Only support CUDA tensors')
        ctx.save_for_backward(v, z, w)
        return cauchy_mult_sym_fwd(v, z, w)

    @staticmethod
    def backward(ctx, dout):
        v, z, w = ctx.saved_tensors
        dv, dw = cauchy_mult_sym_bwd(v, z, w, dout)
        return dv, None, dw