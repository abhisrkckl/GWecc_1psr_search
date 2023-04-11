import numpy as np
import pickle
from enterprise_extensions.empirical_distr import EmpiricalDistribution2D
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions.sampler import JumpProposal
from dynesty import DynamicNestedSampler
from dynesty.results import print_fn as dynesty_print_fn
from analysis import prior_transform_fn


def create_red_noise_empirical_distr(psr, chain_file):
    samples = np.genfromtxt(chain_file)
    param_names = [f"{psr.name}_red_noise_gamma", f"{psr.name}_red_noise_log10_A"]

    gamma_bins = np.linspace(0, 15, 25)
    log10_A_bins = np.linspace(-20, -11, 25)
    return EmpiricalDistribution2D(
        param_names, samples, bins=[gamma_bins, log10_A_bins]
    )

def run_ptmcmc(pta, Niter, outdir, groups=None, empdist=None):
    x0 = np.array([p.sample() for p in pta.params])
    ndim = len(x0)
    cov = np.diag(np.ones(ndim) * 0.01**2)
    x0 = np.hstack(x0)

    print("Starting sampler...\n")

    try:
        jp = JumpProposal(pta, empirical_distr=empdist)
    except Exception:
        # For hypermodel
        jp = JumpProposal(pta.models[1], empirical_distr=empdist)

    sampler = ptmcmc(
        ndim,
        pta.get_lnlikelihood,
        pta.get_lnprior,
        cov,
        groups=groups,
        outDir=outdir,
        resume=False,
        verbose=True,
    )
    # gwecc_params = list(filter(lambda par: "gwecc" in par, pta.param_names))
    sampler.addProposalToCycle(jp.draw_from_prior, 20)
    sampler.addProposalToCycle(jp.draw_from_empirical_distr, 20)

    # This sometimes fails if the acor package is installed, but works otherwise.
    # I don't know why.
    sampler.sample(x0, Niter)

def read_ptmcmc_chain(outdir, burnin_fraction):
    chain_file = f"{outdir}/chain_1.txt"
    chain = np.loadtxt(chain_file)
    print("Chain shape :", chain.shape)

    # burnin_fraction = settings["ptmcmc_burnin_fraction"]
    burn = int(chain.shape[0] * burnin_fraction)
    return chain[burn:, :-4]

def print_dynesty_progress(
    results,
    niter,
    ncall,
    **kwargs,
):
    if niter % 1000 == 0:
        dynesty_print_fn(results, niter, ncall, **kwargs)


def run_dynesty(pta, outdir):
    prior_transform = prior_transform_fn(pta)
    ndim = len(pta.param_names)
    sampler = DynamicNestedSampler(
        pta.get_lnlikelihood, prior_transform, ndim, nlive=1000
    )
    sampler.run_nested(print_progress=True, print_func=print_dynesty_progress)
    res = sampler.results

    result_pkl = f"{outdir}/dynesty_result.pkl"
    with open(result_pkl, "wb") as respkl:
        pickle.dump(res, respkl)

    return res.samples_equal()