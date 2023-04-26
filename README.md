# Neural Laplace Control for Continuous-time Delayed Systems (Code)

[![arXiv](https://img.shields.io/badge/arXiv-2206.04843-b31b1b.svg)](https://arxiv.org/abs/2302.12604)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository is the official implementation of [Neural Laplace Control for Continuous-time Delayed Systems](https://arxiv.org/abs/2302.12604). 

1. Run/Follow steps in [install.sh](setup/install.sh)
2. Replicate experimental results by running and configuring [run_exp_multi.py](run_exp_multi.py).
    ```sh
    python run_exp_multi.py
    ```
3. Process the output log file using [process_logs.py](process_results/process_logs.py) by updating the `LOG_PATH` variable to point to the recently generated log file.
    ```sh
    python process_results/process_logs.py
    ```

#### Retraining
To retrain all models from scratch (much slower), set the following variables to `True` in [run_exp_multi.py](run_exp_multi.py) before running it:
```python
RETRAIN = True
FORCE_RETRAIN = True
```

#### Large files:
To obtain large files like saved models for this work, please download these from Google Drive [here](https://drive.google.com/drive/folders/1j8IijW5iVrxD7hSstBfFpmojAkArP5CU?usp=sharing) and place them into corresponding directories.


## Resources & Other Great Tools 📝
* 💻 [Neural Laplace](https://github.com/samholt/NeuralLaplace): Neural Laplace: Differentiable Laplace Reconstructions for modelling any time observation with O(1) complexity.

### Acknowledgements & Citing `Neural Laplace Control` ✏️

If you use `Neural Laplace Control` in your research, please cite it as follows:

```
@inproceedings{holt2023neural,
  title={Neural Laplace Control for Continuous-time Delayed Systems},
  author={Holt, Samuel and H{\"u}y{\"u}k, Alihan and Qian, Zhaozhi and Sun, Hao and van der Schaar, Mihaela},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1747--1778},
  year={2023},
  organization={PMLR}
}
```
