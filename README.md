# Neural Laplace Control (Code)

This repo contains the code for the paper "Neural Laplace Control for Continuous-time Delayed Systems"

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
