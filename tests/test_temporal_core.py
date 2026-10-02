from __future__ import annotations
import math
import numpy as np
import mpmath as mp
import torch
from scipy.linalg import expm
from reference.temporal_core import (psi3, mode_response, temporal_defect,
    temporal_jets, anchored_amplitudes, TemporalParameterMLP)


def oracle(x: float) -> float:
    with mp.workdps(90):
        z = mp.mpf(float(x))
        if z == 0:
            return 1.0/3.0
        return float((z*z - 2*z + 2 - 2*mp.exp(-z)) / (z*z*z))


def test_values():
    grid = np.r_[0.0, np.logspace(-12, 6, 221), 0.5-1e-8, 0.5, 0.5+1e-8]
    for dtype, tolerance in [(torch.float64, 3e-13), (torch.float32, 5e-6)]:
        x = torch.tensor(grid, dtype=dtype)
        ref = np.array([oracle(v) for v in x.double().numpy()])
        out = psi3(x).double().numpy()
        assert np.max(np.abs((out-ref)/ref)) < tolerance
        assert np.all(out > 0)
    assert psi3(torch.tensor([0.,1.], dtype=torch.bfloat16)).dtype == torch.float32


def test_jets():
    a = torch.tensor([[[0.3,-0.4,0.6]]], dtype=torch.float64)
    rates = torch.tensor([[[0.0,1.2,8.0]]], dtype=torch.float64)
    for hv in [0.0,1e-3,0.04,0.5,2.0]:
        h = torch.tensor(hv, dtype=torch.float64, requires_grad=True)
        jets = temporal_jets(a,rates,h)
        f = jets[0].sum()
        for j in range(1,4):
            f, = torch.autograd.grad(f,h,create_graph=True)
            assert torch.allclose(f,jets[j].sum(),atol=2e-11,rtol=2e-10)


def test_gradcheck():
    a = torch.tensor([[0.4,-0.3,0.1]],dtype=torch.float64,requires_grad=True)
    lam = torch.tensor([[0.03,0.4,8.0]],dtype=torch.float64,requires_grad=True)
    h = torch.tensor(0.2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(temporal_defect,(a,lam,h),atol=2e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(temporal_defect,(a,lam,h),atol=3e-6,rtol=1e-4)


def test_anchor_and_zero():
    e3 = torch.tensor([[0.2,-0.7]],dtype=torch.float64)
    free = torch.tensor([[[0.3,-0.1],[0.9,0.2]]],dtype=torch.float64)
    a = anchored_amplitudes(free,e3)
    assert torch.allclose(a.sum(-1),3*e3)
    rates = torch.tensor([[[0.1,1.,10.]]],dtype=torch.float64)
    jets = temporal_jets(a,rates,torch.tensor(0.,dtype=torch.float64))
    assert all(torch.count_nonzero(t)==0 for t in jets[:3])
    assert torch.allclose(jets[3],6*e3)
    net = TemporalParameterMLP(8,2,4,64).double()
    x = torch.randn(16,8,dtype=torch.float64)
    assert torch.count_nonzero(net(x,torch.tensor(.4,dtype=torch.float64)))==0
    # A single encoder evaluation reused at several h is the intended contract.
    encoded = net.encode(x)
    assert encoded.amplitudes.shape == (16,2,4)
    assert encoded.rates.shape == (16,1,4)


def leading_matrix(A, B):
    result = np.zeros_like(A)
    for p in range(4):
        for q in range(4-p):
            r=3-p-q
            result += (np.linalg.matrix_power(A/2,p)/math.factorial(p)
                       @ (np.linalg.matrix_power(B,q)/math.factorial(q))
                       @ (np.linalg.matrix_power(A/2,r)/math.factorial(r)))
    return np.linalg.matrix_power(A+B,3)/6-result


def test_global_order_on_exact_linear_system():
    A = np.array([[-.3,.8],[-.1,-.2]])
    B = np.array([[-1.0,.1],[.4,-.8]])
    u0 = np.array([.7,-.4]); T=1.
    E3=leading_matrix(A,B)
    errs=[]
    for n in [8,16,32,64]:
        h=T/n
        S=expm(.5*h*A)@expm(h*B)@expm(.5*h*A)
        # lambda=0, amplitude=3*e3 gives h**3*e3.
        M=S+h**3*E3
        got=np.linalg.matrix_power(M,n)@u0
        errs.append(np.linalg.norm(got-expm(T*(A+B))@u0))
    orders=np.log2(np.array(errs[:-1])/np.array(errs[1:]))
    assert np.min(orders[-2:]) > 2.8
    assert np.max(orders[-2:]) < 3.2
