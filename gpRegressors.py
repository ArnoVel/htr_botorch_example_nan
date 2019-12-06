# import gpytorch_local.gpytorch as gpt

import botorch
from botorch.models import SingleTaskGP
from botorch.models import HeteroskedasticSingleTaskGP
from botorch.models.utils import check_min_max_scaling, check_standardization
from botorch.sampling import IIDNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.rbf_kernel import RBFKernel

from torch import Tensor
from torch.nn import Module
import warnings
from typing import Optional
import numpy as np
import torch
import gpytorch

# class GPSKIBaseModel(gpt.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
#         # SKI requires a grid size hyperparameter.
#         #This util can help with that. Here we are using a grid
#         #that has the same number of points as the training data
#         #(a ratio of 1.0).
#         #Performance can be sensitive to this parameter,
#         #so you may want to adjust it for your own problem
#         #on a validation set.
#         grid_size = gpt.utils.grid.choose_grid_size(train_x,2.0)
        
#         self.mean_module = gpt.means.ConstantMean()
#         self.covar_module = gpt.kernels.GridInterpolationKernel(
#             gpt.kernels.ScaleKernel(gpt.kernels.RBFKernel()),
#             grid_size=grid_size, num_dims=1,
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpt.distributions.MultivariateNormal(mean_x, covar_x)



class MostLikelyHeteroGP(object):
    def __init__(self,
                    x_tr: Tensor,
                    y_tr: Tensor,
                    covar_module: Optional[Module] = None,
                    num_var_samples: int = 100,
                    max_iter: int = 10,
                    atol_mean: float = 1e-04,
                    atol_var: float = 1e-04,
                    ):

        if covar_module is None:
            self.covar_module = ScaleKernel(RBFKernel())

        check_min_max_scaling(x_tr)
        check_standardization(y_tr)

        self.x_tr = x_tr
        self.y_tr = y_tr


        self.step_n = 0

        # start with homoskedastic gp
        self.current_model = SingleTaskGP(train_X=x_tr, train_Y=y_tr,
                          covar_module=covar_module)
        self.current_model.likelihood.noise_covar.register_constraint("raw_noise",
                                                      GreaterThan(1e-5))
        self.current_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.current_model.likelihood,
                                                    self.current_model)

        self.num_var_samples = num_var_samples

        self.max_iter = max_iter
        self.atol_mean = atol_mean
        self.atol_var = atol_var

        print('initializing the homoskedastic GP.')
        self.fit_model(self.current_mll)
        self.observe_variance()


    def fit_model(self, mll):
        botorch.fit.fit_gpytorch_model(mll)

    def observe_variance(self, check_before_save=False):
        self.current_mll.eval()

        with torch.no_grad():
            # the posterior is the conditional distribution
            # of the parameters
            self.current_posterior = self.current_mll.model.posterior(self.x_tr.clone())
            # predictive posterior is the cond. probability of the targets
            # after integrating out the parameters
            self.current_predictive = self.current_mll.model.posterior(self.x_tr.clone(),
                                                            observation_noise=True)
            sampler = IIDNormalSampler(num_samples=self.num_var_samples, resample=True)
            predictive_samples = sampler(self.current_predictive)
            self.observed_var = 0.5 * ((predictive_samples - self.y_tr.reshape(-1,1))**2).mean(dim=0)


            if check_before_save:
                new_mean = self.current_posterior.mean
                new_var = self.current_posterior.variance

                means_equal = torch.all(torch.lt(torch.abs(torch.add(self.mean_save, -new_mean)), self.atol_mean))
                self.max_change_in_means = torch.max(torch.abs(torch.add(self.mean_save, -new_mean)))

                var_equal = torch.all(torch.lt(torch.abs(torch.add(self.var_save, -new_var)), self.atol_var))
                self.max_change_in_var = torch.max(torch.abs(torch.add(self.var_save, -new_var)))

                self.converged = (means_equal and var_equal)

            # save mean and variance of the parameters for future comparison
            self.mean_save = self.current_posterior.mean
            self.var_save = self.current_posterior.variance

    def _iterate_em_like_procedure(self):

        # assumes variance have been observed before,
        # at the start, good enough to init, and call
        # fit_current ---> observe_variance
        # will give you the homoskedastic variance at the training point

        hetero_model = HeteroskedasticSingleTaskGP(train_X=self.x_tr, train_Y=self.y_tr,
                                                   train_Yvar=self.observed_var)
        hetero_mll = gpytorch.mlls.ExactMarginalLogLikelihood(hetero_model.likelihood,
                                                              hetero_model)

        try:
            print('fitting hetero model at step ',self.step_n)
            hetero_mll.train()
            self.fit_model(hetero_mll)
        except Exception as e:
            msg = f'Fitting failed on iteration {self.step_n}. the current MLL will not be changed'
            warnings.warn(msg, e)
            raise
        else:
            print('success, replacing old MLL.')
            self.current_model = hetero_model
            self.current_mll = hetero_mll

            self.observe_variance(check_before_save=True)
            self.step_n += 1

    def train(self,num_iters):
        ''' trains for num_iters more iterations from current point,
            stops if max_iter is reached in the process.
            example: current corresponds to iteration 22,
            call with num_iters = 10, and max_iter = 40,
            advances to iteration 32/40'''

        for i in range(num_iters):
            try:
                self._iterate_em_like_procedure()
            except Exception as e:
                print('aborted training.',e)
                raise

            if self.converged:
                print('convergence of both mean and variance on training points.',
                    'Training stopped.')
                break

    def visualize_predicitions(self,num_samples=1000):
        self.current_mll.eval()
        with torch.no_grad():
            test_x = torch.linspace(self.x_tr.min(), self.x_tr.max(), num_samples).cuda()
            posterior_f_vals = self.current_mll.model(test_x)
            posterior_targets = self.current_mll.likelihood(self.current_mll.model(test_x),test_x)

        return test_x, posterior_f_vals, posterior_targets