import torch
from torch.autograd import Function, Variable


class GradientReversalLayer(Function):

    # 重写父类方法的时候，最好添加默认参数，不然会有warning（为了好看。。）
    @ staticmethod
    # def forward(ctx, x, lambd, **kwargs: None):
    #     # 　其实就是传入dict{'lambd' = lambd}
    #     ctx.lambd = lambd
    #     return x.view_as(x)
    def forward(ctx, *x, **kwargs):
        ctx.lambd = x[1]
        return x[0].view_as(x[0])

    @staticmethod
    def backward(ctx, *grad_output):
        # 传入的是tuple，我们只需要第一个
        return grad_output[0] * -ctx.lambd, None

