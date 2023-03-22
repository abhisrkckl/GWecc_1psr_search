from enterprise.signals.gp_signals import MarginalizingTimingModel, FourierBasisGP
from enterprise_extensions.blocks import white_noise_block
from enterprise_gwecc import gwecc_1psr_block
from enterprise.signals.parameter import Uniform
from enterprise.signals.utils import powerlaw

def powerlaw_red_noise_block(vary=True, components=30):
    log10_A = Uniform(-20, -11)
    gamma = Uniform(0, 10)
    pl = powerlaw(log10_A=log10_A, gamma=gamma)   
    
    return FourierBasisGP(pl, components=components, Tspan=None)

def model_gwecc_1psr(
    noise_only=False, 
    tm_use_svd=True, 
    wn_vary=False, 
    wn_inc_ecorr=True,
    rn_vary=True, 
    rn_components=30,
    ecw_param_dict={}
):
    tm = MarginalizingTimingModel(use_svd=tm_use_svd)
    wn = white_noise_block(vary=wn_vary, inc_ecorr=wn_inc_ecorr, select="backend", tnequad=False)
    rn = powerlaw_red_noise_block(vary=rn_vary, components=rn_components)
    
    if noise_only:
        return tm + wn + rn
    
    wf = gwecc_1psr_block(**ecw_param_dict)
    return tm + wn + rn + wf