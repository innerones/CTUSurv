from torch.autograd import gradcheck
import torch
from models import ptopk
input = (torch.randn(2, 20, dtype=torch.double, requires_grad=True), 5)
func_p = ptopk.PerturbedTopK.apply
test = gradcheck(func_p, input, eps=1e-6, atol=1e-4)
print(test)
