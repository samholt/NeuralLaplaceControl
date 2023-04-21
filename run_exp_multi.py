import logging
import os
import time
import traceback
from functools import partial
from pathlib import Path

import torch
import wandb
from torch import multiprocessing
from tqdm import tqdm

from config import dotdict, get_config, seed_all
from mppi_with_model import mppi_with_model_evaluate_single_step
from train_utils import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODELS = ["nl", "oracle", "random", "delta_t_rnn", "node", "latent_ode"]
ENVIRONMENTS = ["oderl-cartpole", "oderl-acrobot", "oderl-pendulum"]
DELAYS = list(range(4))
RETRAIN = False
FORCE_RETRAIN = False
START_FROM_CHECKPOINT = True
MODEL_TRAIN_SEED = 0
PRINT_SETTINGS = False

trainable_models = [model_name for model_name in MODELS if not ("random" in model_name or "oracle" in model_name)]


def train_model_wrapper(args, **kwargs):
    try:
        (env_name, delay, model_name) = args

        config = kwargs["config"]  # pylint: disable=redefined-outer-name
        config = dotdict(config)
        kwargs["config"] = config
        # pylint: disable-next=logging-fstring-interpolation
        logger = create_logger_in_process(config.log_path)  # pylint: disable=redefined-outer-name
        # pylint: disable-next=logging-fstring-interpolation
        logger.info(f"[Now training model] {model_name} \t {env_name} \t {delay}")
        seed_all(config.seed_start)
        kwargs["delay"] = delay
        model, results = train_model(model_name, env_name, **kwargs)  # pylint: disable=unused-variable
        results["errored"] = False
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pylint: disable-next=logging-fstring-interpolation
        logger.exception(f"[Error] {e}")  # pyright: ignore
        # pylint: disable-next=logging-fstring-interpolation
        logger.info(  # pyright: ignore
            f"[Failed training model] {env_name} {model_name} delay={delay} \t "  # pyright: ignore
            f"model_seed={MODEL_TRAIN_SEED} \t | error={e}"
        )
        traceback.print_exc()
        results = {"errored": True}
        print("")
    results.update({"delay": delay, "model_name": model_name, "env_name": env_name})  # pyright: ignore
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(f"[Training Result] {model_name} result={results}")  # pyright: ignore
    return results


def mppi_with_model_evaluate_single_step_wrapper(args, **kwargs):
    try:
        (env_name, delay, model_name, seed) = args

        seed_all(seed)
        config = kwargs["config"]  # pylint: disable=redefined-outer-name
        config = dotdict(config)
        kwargs["config"] = config
        logger = create_logger_in_process(config.log_path)  # pylint: disable=redefined-outer-name
        # pylint: disable-next=logging-fstring-interpolation
        logger.info(f"[Now evaluating mppi model] {model_name} \t {env_name} \t {delay}")
        results = mppi_with_model_evaluate_single_step(
            model_name=model_name,
            action_delay=delay,
            env_name=env_name,
            seed=seed,
            **kwargs,
        )
        results["errored"] = False
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pylint: disable-next=logging-fstring-interpolation
        logger.exception(f"[Error] {e}")  # pyright: ignore
        # pylint: disable-next=logging-fstring-interpolation
        logger.info(  # pyright: ignore
            f"[Failed evaluating mppi model] {env_name} {model_name} delay={delay} \t "  # pyright: ignore
            f"model_seed={MODEL_TRAIN_SEED} \t | error={e}"
        )
        traceback.print_exc()
        results = {"errored": True}
        print("")
    results.update({"delay": delay, "model_name": model_name, "env_name": env_name, "seed": seed})  # pyright: ignore
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(f"[Evaluate Result] result={results}")  # pyright: ignore
    return results


def main(config, wandb=None):  # pylint: disable=redefined-outer-name
    model_training_results_l = []
    model_eval_results_l = []

    pool_outer = multiprocessing.Pool(config.collect_expert_cores_per_env_sampler)
    if config.retrain:
        train_all_model_inputs = [
            (env_name, delay, model_name)
            for env_name in ENVIRONMENTS
            for delay in DELAYS
            for model_name in trainable_models
        ]
        # pylint: disable-next=logging-fstring-interpolation
        logger.info(f"Going to train for {len(train_all_model_inputs)} tasks")
        with multiprocessing.Pool(1) as pool_outer:  # 12
            multi_wrapper_train_model = partial(
                train_model_wrapper,
                config=dict(config),
                wandb=None,
                model_seed=config.model_seed,
                retrain=config.retrain,
                start_from_checkpoint=config.start_from_checkpoint,
                force_retrain=config.force_retrain,
                print_settings=config.print_settings,
                evaluate_model_when_trained=False,
            )
            for i, result in tqdm(  # pylint: disable=unused-variable
                enumerate(pool_outer.imap_unordered(multi_wrapper_train_model, train_all_model_inputs)),
                total=len(train_all_model_inputs),
                smoothing=0,
            ):
                # pylint: disable-next=logging-fstring-interpolation
                logger.info(f"[Model Completed training] {result}")
                model_training_results_l.append(result)

    # Compute the results - in multiprocessing now
    mppi_evaluate_all_model_inputs = [
        (env_name, delay, model_name, seed)
        for env_name in ENVIRONMENTS
        for delay in DELAYS
        for model_name in MODELS
        for seed in range(config.seed_start, config.seed_runs + config.seed_start)
    ]
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(f"Evaluating mppi for seed input {len(mppi_evaluate_all_model_inputs)} tasks")
    if config.multi_process_results:
        pool_outer = multiprocessing.Pool(12)  # 12, 8 , 18
    multi_wrapper_mppi_evaluate = partial(
        mppi_with_model_evaluate_single_step_wrapper,
        config=dict(config),
        roll_outs=config.mppi_roll_outs,
        time_steps=config.mppi_time_steps,
        lambda_=config.mppi_lambda,
        sigma=config.mppi_sigma,
        dt=config.dt,
        encode_obs_time=config.encode_obs_time,
        save_video=config.save_video,
    )
    if config.multi_process_results:
        for i, result in tqdm(
            enumerate(pool_outer.imap_unordered(multi_wrapper_mppi_evaluate, mppi_evaluate_all_model_inputs)),
            total=len(mppi_evaluate_all_model_inputs),
            smoothing=0,
        ):
            # pylint: disable-next=logging-fstring-interpolation
            logger.info(f"[Model Completed evaluation mppi] {result}")
            model_eval_results_l.append(result)
    else:
        for i, task_input in tqdm(
            enumerate(mppi_evaluate_all_model_inputs),
            total=len(mppi_evaluate_all_model_inputs),
            smoothing=0,
        ):
            result = multi_wrapper_mppi_evaluate(task_input)
            # pylint: disable-next=logging-fstring-interpolation
            logger.info(f"[Model Completed evaluation mppi] {result}")
            model_eval_results_l.append(result)
    if config.multi_process_results:
        pool_outer.close()


def generate_log_file_path(file, log_folder="logs"):
    file_name = os.path.basename(os.path.realpath(file)).split(".py")[0]
    Path(f"./{log_folder}").mkdir(parents=True, exist_ok=True)
    path_run_name = "{}-{}".format(file_name, time.strftime("%Y%m%d-%H%M%S"))
    return f"{log_folder}/{path_run_name}_log.txt"


def create_logger_in_process(log_file_path):
    logger = multiprocessing.get_logger()  # pylint: disable=redefined-outer-name
    if not logger.hasHandlers():
        formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file_path)
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    log_path = generate_log_file_path(__file__)
    logger = create_logger_in_process(log_path)
    defaults = get_config()
    defaults["log_path"] = log_path
    if defaults["multi_process_results"]:
        torch.multiprocessing.set_start_method("spawn")
    defaults["retrain"] = RETRAIN
    defaults["force_retrain"] = FORCE_RETRAIN
    defaults["start_from_checkpoint"] = START_FROM_CHECKPOINT
    defaults["print_settings"] = PRINT_SETTINGS
    defaults["model_train_seed"] = MODEL_TRAIN_SEED
    defaults["sweep_mode"] = True  # Real run settings
    defaults["end_training_after_seconds"] = int(1350 * 6.0)

    wandb.init(config=defaults, project=defaults["wandb_project"])  # pyright: ignore
    config = wandb.config
    seed_all(0)
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(f"Starting run \t | See log at : {log_path}")
    main(config, wandb)
    wandb.finish()
    logger.info("Run over. Fin.")
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(f"[Log found at] {log_path}")
