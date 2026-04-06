import os
from pathlib import Path
import torch
import wandb
from fed_trainers.trainers.factory import get_trainer, get_dataloaders
from fed_trainers.trainers.params import add_arguments
from fed_trainers.trainers.utils import get_logger, set_seed, log_data_statistics, compute_sample_probability, \
    compute_steps, get_sigma, create_wandb_report
from fed_trainers.datasets.emg_utils import get_num_users


def train(args):
    set_seed(args.seed)
    dataloaders = get_dataloaders(args)
    log_data_statistics(dataloaders, args)

    q = compute_sample_probability(args)
    steps = compute_steps(args)
    logger = get_logger(args)
    logger.info(f"steps: {steps}")
    logger.info(f"sample probability (q): {q}")

    args.noise_multiplier, actual_epsilon = (args.noise_multiplier, None) if args.eps < 0 else get_sigma(q, steps,
                                                                                                         args.eps,
                                                                                                         args.delta,
                                                                                                         rgp=False)

    logger.info(f"noise_multiplier: {args.noise_multiplier}")
    logger.info(f"actual_epsilon: {actual_epsilon}")

    trainer = get_trainer(args)

    trainer.train(args, dataloaders)

def main():

    data_name = 'keypressemg'
    use_gp = os.environ.get('USE_GP', 'False')
    dp_method = 'gep_public'
    num_users = get_num_users()
    num_classes = 26
    num_public_clients = 3
    working_dir = Path(__file__).resolve().parents[2]

    args = add_arguments(data_name, dp_method, num_classes, num_public_clients, num_users, use_gp, working_dir)

    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    logger = get_logger(args)
    logger.info(f"Args: {args}")
    logger.debug('Debug Logger Set')

    set_seed(args.seed)

    q = compute_sample_probability(args)
    steps = compute_steps(args)

    logger.info(f"steps: {steps}")
    logger.info(f"sample probability (q): {q}")

    args.noise_multiplier, actual_epsilon = (args.noise_multiplier, None) if args.eps < 0 else get_sigma(q, steps, args.eps, args.delta, rgp=False)

    logger.info(f"noise_multiplier: {args.noise_multiplier}")
    logger.info(f"actual_epsilon: {actual_epsilon}")

    exp_name = f'{dp_method.upper()}_{data_name.upper()}_lr_{args.lr}_clip_{args.clip}_noise_{args.noise_multiplier}_seed_{args.seed}'
    if use_gp:
        exp_name = f'GP_{exp_name}'

    # Weights & Biases
    if args.wandb:
        run = wandb.init(project="dec25_sweeps", name=exp_name, tags=[args.run_tag])
        wandb.config.update(args)
        report = create_wandb_report(args)

    train(args)

    if args.wandb:
        run.finish()

if __name__ == '__main__':
    main()
