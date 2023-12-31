import argparse
import random
import pandas as pd
import os
import datetime
import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.envs.vecenv import SubprocVectorEnv, VectorEnv
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import (VecMdpPathCollector, VecMdpStepCollector)
from rlkit.torch.dsac.dsac import DSACTrainer
from rlkit.torch.dsac.networks import QuantileMlp, softmax
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.torch_rl_algorithm import TorchVecOnlineRLAlgorithm
from rlkit.core import logger
from utils import write_to_csv as wtc
from environments.citation_envs.phlabenv import CitationEnv
import wandb

wandb.login()
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

def experiment(env, variant):

    dummy_env   = env
    obs_dim     = dummy_env.observation_space.low.size
    action_dim  = dummy_env.action_space.low.size

    # Parallelization
    expl_env = VectorEnv([lambda: env for _ in range(variant['expl_env_num'])])
    expl_env.seed(variant["seed"])
    expl_env.action_space.seed(variant["seed"])
    eval_env = SubprocVectorEnv([lambda: env for _ in range(variant['eval_env_num'])])
    eval_env.seed(variant["seed"])

    M               = variant['layer_size']
    num_quantiles   = variant['num_quantiles']
    tau_embedding   = variant['tau_embedding']
    hidden_sizes    = variant['structureDNN']

    zf1 = QuantileMlp(
        input_size      = obs_dim + action_dim,
        output_size     = 1,
        num_quantiles   = num_quantiles,
        hidden_sizes    = hidden_sizes,
        embedding_size  = tau_embedding
    )
    print('Number of parameters', sum(p.numel() for p in zf1.parameters() if p.requires_grad))

    zf2 = QuantileMlp(
        input_size      = obs_dim + action_dim,
        output_size     = 1,
        num_quantiles   = num_quantiles,
        hidden_sizes    = hidden_sizes,
        embedding_size  = tau_embedding
    )
    target_zf1 = QuantileMlp(
        input_size      = obs_dim + action_dim,
        output_size     = 1,
        num_quantiles   = num_quantiles,
        hidden_sizes    = hidden_sizes,
        embedding_size  = tau_embedding
    )
    target_zf2 = QuantileMlp(
        input_size      = obs_dim + action_dim,
        output_size     = 1,
        num_quantiles   = num_quantiles,
        hidden_sizes    = hidden_sizes,
        embedding_size  = tau_embedding
    )

    policy = TanhGaussianPolicy(
        obs_dim         = obs_dim,
        action_dim      = action_dim,
        hidden_sizes    = hidden_sizes,
    )
    eval_policy     = MakeDeterministic(policy)

    target_policy   = TanhGaussianPolicy(
        obs_dim         = obs_dim,
        action_dim      = action_dim,
        hidden_sizes    = hidden_sizes
    )

    # Fraction Proposal Network
    fp = target_fp = None
    if variant['trainer_kwargs'].get('tau_type') == 'fqf':
        fp = FlattenMlp(
            input_size          = obs_dim + action_dim,
            output_size         = num_quantiles,
            hidden_sizes        = [M // 2, M // 2],
            output_activation   = softmax,
        )
        target_fp = FlattenMlp(
            input_size          = obs_dim + action_dim,
            output_size         = num_quantiles,
            hidden_sizes        = [M // 2, M // 2],
            output_activation   = softmax,
        )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )

    trainer = DSACTrainer(
        env             = dummy_env,
        policy          = policy,
        target_policy   = target_policy,
        zf1             = zf1,
        zf2             = zf2,
        target_zf1      = target_zf1,
        target_zf2      = target_zf2,
        fp              = fp,
        target_fp       = target_fp,
        num_quantiles   = num_quantiles,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchVecOnlineRLAlgorithm(
        trainer         = trainer,
        exploration_env = expl_env,
        evaluation_env  = eval_env,
        exploration_data_collector = expl_path_collector,
        evaluation_data_collector  = eval_path_collector,
        replay_buffer   = replay_buffer,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()

    expl_env.close()
    eval_env.close()
    dummy_env.close()
    del algorithm, trainer, replay_buffer


# The main entry point of the program
if __name__ == "__main__":

    script_directory = os.path.dirname(os.path.abspath(__file__))

    best_final  = -999999999
    best_seed   = 0

    # Free up the GPU memory and make it available for other computations
    torch.cuda.empty_cache()

    # Create an argument parser with a description
    parser  = argparse.ArgumentParser(description='Distributional Soft Actor Critic', conflict_handler='resolve')

    # Add command-line arguments to the parser
    parser.add_argument('--config', type=str, default = "configs/Attitude/config.yaml")
    parser.add_argument('--gpu', type=int, default=0, help = "using cpu with -1")
    parser.add_argument('--seed', type=int, default=0)

    # Parse the command-line arguments
    args        = parser.parse_args()

    # Open the configuration file specified in the '--config' argument for reading
    with open(args.config, 'r', encoding="utf-8") as f:
        # Load the YAML contents of the file into a Python dictionary
        variant = yaml.load(f, Loader = yaml.FullLoader)

    env         = CitationEnv(configuration = variant, mode = 'nominal')
    # model_name  = "Cessna500_nonlin_risky_correct"
    model_name  = 'RUN-DSAC-C'

    runs        = []
    runs_dir    = os.path.join(script_directory, "data/run-order")
    now         = datetime.datetime.now()
    timestamp   = now.strftime("%Y-%m-%d_%H-%M-%S")
    runs_file   = f"runs_{timestamp}.csv"

    # Number of seeds = number of runs
    num_seeds   = 100

    for i in range(num_seeds):

        random.seed(None)
        sel_seed        = random.randint(1, 10000)
        variant["seed"] = sel_seed

        # Generate a log prefix based on the configuration
        log_prefix = "_".join(["RUN-DSAC-C-100STRONG", model_name, str(variant["version"])])

        # Set up the logger with the log prefix, configuration, and seed
        logging_dir = setup_logger(log_prefix, variant=variant, seed=sel_seed)


        wandb.init(
            project = 'RUN-DSAC-C 100 STRONG',
            name    = f'Sperimentare_MAE_{sel_seed}',
            config = {
                "batch_size":                   variant["algorithm_kwargs"]["batch_size"],
                "num_epochs":                   variant["algorithm_kwargs"]["num_epochs"],
                "structure_DNN":                variant["structureDNN"],
                "tau_embedding":                variant["tau_embedding"],
                "num_quantiles":                variant["num_quantiles"],
                "replay_buffer_size":           variant["replay_buffer_size"],
                "discount":                     variant["trainer_kwargs"]["discount"],
                "alpha":                        variant["trainer_kwargs"]["alpha"],
                "policy_lr":                    variant["trainer_kwargs"]["policy_lr"],
                "zf_lr":                        variant["trainer_kwargs"]["zf_lr"],
                "use_automatic_entropy_tuning": variant["trainer_kwargs"]["use_automatic_entropy_tuning"],
                "polyak_step":                  variant["trainer_kwargs"]["polyak_step"],
                "target_update_period":         variant["trainer_kwargs"]["target_update_period"],
                "lambda_t":                     variant["trainer_kwargs"]["lambda_t"],
                "lambda_s":                     variant["trainer_kwargs"]["lambda_s"],
                "aspiration":                   variant["aspiration"],
                "location":                     logging_dir,
            }
        )


        # If the '--gpu' argument is greater than or equal to 0, set GPU mode to True with the specified GPU index
        if args.gpu >= 0:
            ptu.set_gpu_mode(True, args.gpu)

        # Set the random seed for reproducibility
        set_seed(sel_seed)

        # Run the experiment with the specified variant
        experiment(env, variant)

        wandb.finish()

        # Evaluate the best runs
        data        = pd.read_csv(os.path.join(logging_dir, "progress.csv"), header=0)
        returns     = np.mean(data["evaluation/Average Returns"][-7:])

        if returns > best_final:
            best_final  = returns
            best_seed   = sel_seed

        runs.append((logging_dir, returns))
        logger.reset()
        sorted_runs     = sorted(runs, key = lambda x: x[1])
        wtc(runs_dir, runs_file, sorted_runs)

    del data
    print('Best reward:', best_final, 'Seed:', best_seed)

