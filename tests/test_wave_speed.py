
import numpy as np
from splitstep_toe.core.engine import step_2d

def test_pulse_speed():
    ny=nx=31; h=1.0
    R0=np.zeros((ny,nx)); R0[ny//2,nx//2]=1.0
    R1=R0.copy()
    kappa=0.25; lam=0.; gamma=1.0
    for _ in range(3):
        R2=step_2d(R0,R1,kappa,lam,gamma,h)
        R0,R1 = R1,R2
    assert np.isfinite(R1).all()
