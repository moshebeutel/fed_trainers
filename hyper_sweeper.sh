#!/bin/bash

# Define the Python script filenames
script1="sweeper_cifar10_dp_no_gp.py"
script2="sweeper_cifar10_gep_no_gp.py"
script3="sweeper_cifar10_gep_public_no_gp.py"

# Run the first Python script
echo "Running $script1..."
python3 "$script1"

# Check if the first script was successful
if [ $? -ne 0 ]; then
    echo "$script1 failed. Exiting."
    exit 1
fi

# Run the second Python script
echo "Running $script2..."
python3 "$script2"

# Check if the second script was successful
if [ $? -ne 0 ]; then
    echo "$script2 failed. Exiting."
    exit 1
fi

# Run the third Python script
echo "Running $script3..."
python3 "$script3"

# Check if the third script was successful
if [ $? -ne 0 ]; then
    echo "$script3 failed. Exiting."
    exit 1
fi

echo "All scripts ran successfully."
