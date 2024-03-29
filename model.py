from enterprise.signals.gp_signals import FourierBasisGP, MarginalizingTimingModel
from enterprise.signals.signal_base import PTA
from enterprise.signals.parameter import Uniform, Constant
from enterprise.signals.utils import powerlaw
from enterprise_extensions.blocks import white_noise_block
from enterprise_gwecc import gwecc_1psr_block
import numpy as np

def powerlaw_red_noise_block(components=30, vary=True, log10_A=Uniform(-20, -11), gamma=Uniform(0, 15)):
    pl = powerlaw(log10_A=log10_A, gamma=gamma)  if vary else powerlaw(log10_A=Constant(), gamma=Constant())
    return FourierBasisGP(pl, components=components, Tspan=None)


def model_gwecc_1psr(
    noise_only=False,
    tm_use_svd=True,
    wn_vary=False,
    wn_include_ecorr=True,
    wn_normal_efac=True,
    rn_vary=True,
    rn_components=30,
    ecw_param_dict={},
):
    tm = MarginalizingTimingModel(use_svd=tm_use_svd)
    wn = white_noise_block(
        vary=wn_vary,
        inc_ecorr=wn_include_ecorr,
        select="backend",
        tnequad=False,
        efac1=wn_normal_efac,
    )
    rn = powerlaw_red_noise_block(components=rn_components, vary=rn_vary)

    if noise_only:
        return tm + wn + rn

    wf = gwecc_1psr_block(**ecw_param_dict)
    return tm + wn + rn + wf

def prior_transform_fn(pta):
    mins = np.array([param.prior.func_kwargs['pmin'] for param in pta.params])
    maxs = np.array([param.prior.func_kwargs['pmax'] for param in pta.params])
    spans = maxs-mins
    
    def prior_transform(cube):
        return spans*cube + mins
    
    return prior_transform
