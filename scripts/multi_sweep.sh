#!/bin/bash

EPOCHS=3

# Loop over seeds 101 202 303
for seed in 101 202 303
do
  for eps in 8 1
  do
      echo "Running with seed: $seed"

      # Run sweep_cifar10_gep_private
      echo "Running sweep_cifar10_gep_private with seed $seed epsilon $eps"
      poetry run sweep_cifar10_gep_private --seed $seed --wandb true --eps $eps --n_epochs $EPOCHS

      # Run sweep_cifar10_gep_public
      echo "Running sweep_cifar10_gep_public with seed $seed epsilon $eps"
      poetry run sweep_cifar10_gep_public --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS

#      # Run sweep_cifar10_gep_public_private
#      echo "Running sweep_cifar10_gep_public_private with seed $seed epsilon $eps"
#      poetry run sweep_cifar10_gep_public_private --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS

      # Run sweep_cifar10_sgd_dp
      echo "Running sweep_cifar10_sgd_dp with seed $seed epsilon $eps"
      poetry run sweep_cifar10_sgd_dp --seed $seed --wandb true --eps $eps  --n_epochs $EPOCHS

      echo "Completed all runs for epsilon $eps"
      echo "----------------------------------------"
  done
  echo "Completed all runs for seed $seed"
  echo "----------------------------------------"
done
echo "All training runs completed!"