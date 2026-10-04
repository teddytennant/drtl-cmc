drtl-cmc implements doubly robust target inference for a generalized linear regression coefficient when a covariate is completely missing in the target sample (Zhao and Deng, arXiv:2609.24086).

Install from this directory with `pip install -e .`. Run `python -m drtl_cmc.demo` for one correct-specification logistic replicate and one misspecified replicate.

This is not a reproduction of the paper's numerical study. Bootstrap standard errors are not implemented. Inverse-probability weighting (Lipsitz) and predict-then-debias are not implemented: those reference methods do not apply when the covariate is completely missing in the target. The UK Biobank analysis is not included. Logistic conditional functionals are Monte Carlo averages under the fitted Gaussian covariate model rather than a closed form. Demo and test sample sizes are smaller than Section 5.
