# lar_testing

This repository contains an illustration of the numerical experiments performed in the paper entitled
> *Multiple Testing and Variable Selection Along the Path of the Least Angle Regression*, [arXiv:1906.12072v5](https://arxiv.org/abs/1906.12072v5).

by Jean-Marc Azaïs and [Yohann De Castro](https:ydecastro.github.io). The Python code can be downloaded at

[Github repository lar_testing](https://github.com/ydecastro/lar_testing)

and the code `lar_testing-v2.0` used in the paper *arXiv:1906.12072v5* has been posted on Zenodo:

[![DOI](https://zenodo.org/badge/182083824.svg)](https://zenodo.org/badge/latestdoi/182083824)

## Comparison with Knockoff, FCD and SLOPE on HIV dataset

The [**first notebook** called *Multiple Spacing Tests*](https://github.com/ydecastro/lar_testing/blob/master/multiple_spacing_tests.ipynb) presents the numerical experiments of the paper entitled

> *Multiple Testing and Variable Selection Along a Path of the Least Angle Regression*, [arXiv:1906.12072v5](https://arxiv.org/abs/1906.12072v5).

We present the following points:
- A comparison of the **power and FDR control** on simulated data for **GtST, FCD and Knockoff** in **Section I**;
- Presentation of **HIV dataset** in **Section II**;
- A comparison of the **power and FDR control on HIV dataset** for **GtST, FCD, Knockoff and Slope** in **Section III**;
- A **new formulation of LARS algorithm** in **Section IV**;

The methods considered are:
- **[Knockoff]** Knockoff filters for FDR control and we use the implementation presented on the webpage <https://web.stanford.edu/group/candes/knockoffs/> based on the paper:
> *Controlling the False Discovery Rate via Knockoffs*, [arXiv:1404.5609](https://arxiv.org/abs/1404.5609);
- **[FCD]** False Discovery Control via Debiasing and we use the implementation of debiased lasso presented on the webpage <https://web.stanford.edu/~montanar/sslasso/> and as presented in the paper:
> *False Discovery Rate Control via Debiased Lasso*, [arXiv:1803.04464](https://arxiv.org/abs/1803.04464);
- **[Slope]** Slope for FDR control, as presented in the paper:
> *SLOPE - Adaptive variable selection via convex optimization* [arXiv:1407.3824](https://arxiv.org/abs/1407.3824);
- **[GtSt-BH]** Generalized t-Spacing tests on successive entries of the LARS path comibined with a Benjamini–Hochberg procedure presented in the paper:
> *Multiple Testing and Variable Selection along Least Angle Regression's path*, [arXiv:1906.12072v5](https://arxiv.org/abs/1906.12072v5).


## Numerical joint law
The [**second notebook**](https://github.com/ydecastro/lar_testing/blob/master/Law_LAR.ipynb) gives an empirical evidence of the joint law shown in the paper
> *Multiple Testing and Variable Selection along Least Angle Regression's path*, [arXiv:1906.12072v5](https://arxiv.org/abs/1906.12072v5).

*Thank you for your time!*
