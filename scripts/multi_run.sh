#!/bin/bash

EPOCHS=5

# Loop over seeds 101 202 303
for seed in 101 202 303
do
  for eps in 8 3 1
  do
      echo "Running with seed: $seed"

      # Run train_cifar10_gep_private
      echo "Running train_cifar10_gep_private with seed $seed epsilon $eps"
      poetry run train_cifar10_gep_private --seed $seed --wandb true --eps $eps --n_epochs $EPOCHS

      # Run train_cifar10_gep_public
      echo "Running train_cifar10_gep_public with seed $seed epsilon $eps"
      poetry run train_cifar10_gep_public --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS

      # Run train_cifar10_gep_public
      echo "Running train_cifar10_gep_public with seed $seed epsilon $eps"
      poetry run train_cifar10_gep_public_private --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS

      # Run train_cifar10_sgd_dp
      echo "Running train_cifar10_sgd_dp with seed $seed epsilon $eps"
      poetry run train_cifar10_sgd_dp --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS

      echo "Completed all runs for epsilon $eps"
      echo "----------------------------------------"
  done
  echo "Completed all runs for seed $seed"
  echo "----------------------------------------"
done
echo "All training runs completed!"