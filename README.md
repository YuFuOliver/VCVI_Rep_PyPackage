# VCVI Rep PyPackage

This repository contains a Python package to replicate the results in the paper **"Vector Copula Variational Inference and Dependent Block Posterior Approximations"** by Yu Fu, Michael Stanley Smith, and Anastasios Panagiotelis ([arxiv link](https://arxiv.org/abs/2503.01072)). The dependencies will be installed automatically by installing the package. The package is developed and maintained by Yu Fu.

The `VCVI` package contains variational inference (VI) algorithms used in the paper. `VCVI` is developed based on **PyTorch**.

The results in Table2 & Table3 are trained on **GPU** (via Google Colab), and the rest results are trained on **CPU**. To facilitate users who cannot access to a GPU from Google Colab, we provide instructions about how to replicate Sections 4.1 & 4.2 by a local CPU/GPU.

## Replication
### Install VCVI Package

```bash
pip install git+https://github.com/YuFuOliver/VCVI_Rep_PyPackage.git#subdirectory=VCVI
```

### Download Replication Files:

#### Empirical results in Section 4.1 (`logit_reg`)
- Table 2 (and Table A3 in the online appendix): `logit_reg/TABLE_ELBO`:
  - Run `TABLE_ELBOs_real_GPU_Colab.ipynb` & `TABLE_ELBO_simu_GPU_Colab.ipynb`, and then `TABLE_ELBOs.ipynb`
  - `TABLE_ELBOs_real_GPU_Colab.ipynb` & `TABLE_ELBO_simu_GPU_Colab.ipynb` are designed to run on Google Colab
- Figure 1: `logit_reg/PLOT_dep.ipynb`
- Figure 2: `logit_reg/PLOT_qsar.ipynb`
- Table A1 & Table A5 in the online appendix: `logit_reg/TABLE_time`:
  - Run `TABLE_time_real.ipynb` & `TABLE_time_simu.ipynb`, and then `TABLE_times.ipynb`


#### `correlation`: Empirical results in Section 4.2
- `TABLE_ELBOs_GPU_to30.ipynb` & `TABLE_ELBOs_GPU_49.ipynb` contain code for Table 3 (and Table A4 in the online appendix)
  - The codes are designed to run on Google Colab
- `PLOT_ELBO&beta.ipynb` produces Figure 3

#### `SVUC`: Empirical results in Section 4.3
- `TABLE_para.ipynb` & `TABLE_ELBOtime.ipynb` contain codes for Table 4
- `PLOT_muzeta.ipynb` produces Figure 4

#### `spline_add`: Empirical results in Section 4.4
- `TABLE_para.ipynb` & `TABLE_ELBOtime.ipynb` contain codes for Table 5
- `PLOT_func.ipynb` produces Figure 5












<!-- ### GPU Support

GPU algorithms are available as a separate package `VCVI_GPU`. Install separately:

```bash
pip install git+https://github.com/YuFuOliver/VCVI_Rep_PyPackage.git#subdirectory=VCVI/GPU
```

> **Note:** GPU package allows flexible PyTorch versions for different CUDA setups. -->

<!-- ## Usage
The packages are highly user-friendly. However, the purpose of this readme is to introduce how to replicate the paper but not to explain how to use the packages.

The packages support user-defined posterior distributions or any model written in **Stan**.

Training is as simple as:
```python
# mean field variational inference
from VCVI import MFVI

mf = MFVI(optimizer='Adam', sampling=False,
          stan_model=None, log_post=log_post)
          
ELBO_mf = mf.train(num_iter=40000)
``` -->



## Citation
If you use this package in your research, please cite our paper:  
> [Vector Copula Variational Inference and Dependent Block Posterior Approximations](https://arxiv.org/abs/2503.01072)
