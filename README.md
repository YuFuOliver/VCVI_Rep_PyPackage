# VCVI Rep PyPackage

This repository contains Python packages `VCVI` & `VCVI_GPU` to replicate the results in the paper **"Vector Copula Variational Inference and Dependent Block Posterior Approximations"** by Yu Fu, Michael Stanley Smith, and Anastasios Panagiotelis ([arxiv link](https://arxiv.org/abs/2503.01072)). The dependencies will be installed automatically by installing the packages. The packages are developed by Yu Fu.

The package `VCVI` contains variational inference (VI) algorithms used in the paper. The package `VCVI_GPU` contains several VI algorithms from `VCVI` that can be run on GPU. `VCVI` and `VCVI_GPU` are developed based on **PyTorch**.

The results in **Table2** & **Table3** are trained on **GPU** (via Google Colab), and the rest results are trained on **CPU**. To facilitate users who cannot access to a GPU from Google Colab, we provide additional instructions about how to replicate **Table2** & **Table3** by a local CPU/GPU.

## Replication
### Install `VCVI` Package

```bash
pip install git+https://github.com/YuFuOliver/VCVI_Rep_PyPackage.git#subdirectory=VCVI
```

### Download Replication Files: 
https://drive.google.com/drive/folders/1QYH6k9dr4CV8yKARyH2hQAhvypV9jAK9?usp=sharing

#### Empirical results in Section 4.1 (`logit_reg/`):
- **Table 2** (and **Table A3** in the online appendix): `TABLE_ELBO/`:
  - Run `TABLE_ELBOs_real_GPU_Colab.ipynb` & `TABLE_ELBO_simu_GPU_Colab.ipynb` (designed to be run on Google Colab), and then `TABLE_ELBOs.ipynb`
- **Figure 1**: `PLOT_dep.ipynb`
- **Figure 2**: `PLOT_qsar.ipynb`
- **Table A1** & **Table A5** in the online appendix: `TABLE_time/`:
  - Run `TABLE_time_real.ipynb` & `TABLE_time_simu.ipynb`, and then `TABLE_times.ipynb`

#### Empirical results in Section 4.2 (`correlation/`):
- **Table 3** (and **Table A4** in the online appendix): `TABLE_ELBOs_GPU_to30.ipynb` & `TABLE_ELBOs_GPU_49.ipynb` (designed to be run on Google Colab)
- **Figure 3**: `PLOT_ELBO&beta.ipynb`

#### Empirical results in Section 4.3 (`SVUC/`):
- **Table 4**: `TABLE_para.ipynb` & `TABLE_ELBOtime.ipynb`
- **Figure 4**: `PLOT_muzeta.ipynb`

#### Empirical results in Section 4.4 (`spline_add/`):
- **Table 5**: `TABLE_para.ipynb` & `TABLE_ELBOtime.ipynb`
- **Figure 5**: `PLOT_func.ipynb`

## Replicate Table 2 & Table 3 on local a CPU/GPU

To run VCVI algorithms on a GPU:
- Install a CUDA version of PyTorch from: https://pytorch.org/get-started/locally/
- Install the package `VCVI_GPU` containing VCVI algorithms on a GPU:
```bash
pip install git+https://github.com/YuFuOliver/VCVI_Rep_PyPackage.git#subdirectory=VCVI/GPU
```

**Table2**:
- For real datasets: `logit_reg/TABLE_ELBO/TABLE_ELBOs_real_local.ipynb`
  - `device=cpu` to run on CPU; `device=gpu` to run on GPU
  - `vi_methods` controls different VA methods (rows in the table)
  - `datasets` controls different datasets (columns in the table)
- For simulation datasets: `logit_reg/TABLE_ELBO/TABLE_ELBOs_simu_local.ipynb`
  - `device=cpu` to run on CPU; `device=gpu` to run on GPU
  - `vi_methods` controls different VA methods (rows in the table)
  - `n_samples` controls different simulation datasets (columns in the table)

**Table 3**: `correlation/TABLE_ELBOs_local.ipynb`
- `device=cpu` to run on CPU; `device=gpu` to run on GPU
- `vi_methods` controls different VA methods (rows in the table)
- `num_states` controls the number of states to be included (columns in the table); `num_iters` should be consistent with `num_states` (each number of states corresponds to a number of iterations)

## Design of the Packages
The packages are highly user-friendly. The purpose of this readme is to introduce how to replicate the paper. The design of the packages is introduced briefly here.

Training a model is as simple as:
```python
# mean field variational inference
from VCVI import MFVI

mf = MFVI(optimizer='Adam', sampling=False,
          stan_model=None, log_post=log_post)
          
ELBO_mf = mf.train(num_iter=40000)
```

The packages support **user-defined posterior distributions** or any model written in **Stan**.
- An example of user-defined posterior can be found in replication files: `logit_reg/logh_logitreg_autodiff.py`
- An example of a model written in Stan can be found in replication files: `SVUC/mcmc/SVUC_model.stan`


## Citation
If you use this package in your research, please cite our paper:  
> [Vector Copula Variational Inference and Dependent Block Posterior Approximations](https://arxiv.org/abs/2503.01072)
