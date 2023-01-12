from analysis import read_data
from enterprise.signals.parameter import Uniform
from enterprise.signals.gp_signals import TimingModel
from enterprise.signals.signal_base import PTA
from enterprise.signals.white_signals import MeasurementNoise
from enterprise_gwecc import gwecc_1psr_block
from enterprise_warp.bilby_warp import get_bilby_prior_dict, PTABilbyLikelihood

import numpy as np
import bilby
import pickle

def main():
    prefix = "JPSR00_simulate"
    parfile = f"{prefix}.par"
    timfile = f"{prefix}.tim"
    psr = read_data(parfile, timfile, prefix, noise_dict=False)

    name = "gwecc"

    tref = max(psr.toas)
    priors = {
        "alpha": Uniform(0, 1)(f"{name}_alpha"),
        "psi": Uniform(0, np.pi)(f"{name}_psi"),
        "cos_inc": Uniform(-1, 1)(f"{name}_cos_inc"),
        "log10_M": Uniform(6, 9)(f"{name}_log10_M"),
        "eta": Uniform(0, 0.25)(f"{name}_eta"),
        "log10_F": Uniform(-9, -7)(f"{name}_log10_F"),
        "e0": Uniform(0.01, 0.8)(f"{name}_e0"),
        "gamma0": Uniform(0, np.pi)(f"{name}_gamma0"),
        "gammap": Uniform(0, np.pi),
        "l0": Uniform(0, 2 * np.pi)(f"{name}_l0"),
        "lp": Uniform(0, 2 * np.pi),
        "tref": tref,
        "log10_zc": Uniform(-4, -3)(f"{name}_log10_zc"),
    }

    tm = TimingModel()
    wn = MeasurementNoise(efac=1)
    wf = gwecc_1psr_block(**priors)
    model = tm + wn + wf

    pta = PTA([model(psr)])

    print(pta.param_names)
    
    x0 = [p.sample() for p in pta.params]
    print(x0, pta.get_lnlikelihood(x0))
    
    bilby_prior = get_bilby_prior_dict(pta)
    bilby_likelihood = PTABilbyLikelihood(pta, parameters=dict.fromkeys(bilby_prior.keys()))
    bilby_label = prefix

    result = bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_prior,
        outdir='./chains/',
        label=bilby_label,
        sampler='nestle',
        resume=False,
    )

if __name__ == "__main__":
    main()
