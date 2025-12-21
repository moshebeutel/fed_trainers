#!/bin/bash

EPOCHS=3

# Loop over seeds 101 202 303
for seed in 101 202 303
do
  for inner_steps in 5
  do
    for eps in 8 1
    do
        # Run train_cifar10_gep_private
        echo "Running train_cifar10_gep_private with seed $seed epsilon $eps inner steps $inner_steps"
        poetry run train_cifar10_gep_private --seed $seed --wandb true --eps $eps --n_epochs $EPOCHS --inner-steps $inner_steps

        # Run train_cifar10_gep_public
        echo "Running train_cifar10_gep_public with seed $seed epsilon $eps inner_steps $inner_steps"
        poetry run train_cifar10_gep_public --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS --inner-steps $inner_steps

        # Run train_cifar10_gep_public
        echo "Running train_cifar10_gep_public with seed $seed epsilon $eps inner_steps $inner_steps"
        poetry run train_cifar10_gep_public_private --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS --inner-steps $inner_steps

        # Run train_cifar10_sgd_dp
        echo "Running train_cifar10_sgd_dp with seed $seed epsilon $eps inner_steps $inner_steps"
        poetry run train_cifar10_sgd_dp --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS --inner-steps $inner_steps

        echo "Completed all runs for epsilon $eps"
        echo "----------------------------------------"
    done
    echo "Completed all runs for inner_steps $inner_steps"
    echo "----------------------------------------"
  done
  echo "Completed all runs for seed $seed"
  echo "----------------------------------------"
done
echo "All training runs completed!"