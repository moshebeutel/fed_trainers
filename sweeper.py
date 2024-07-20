
import math
import subprocess

# ["num_blocks", "block_size", "optimizer", "lr", "num_clients_agg", "clip"]]

for num_epochs in [20]:
    for num_clients in [50]:
        for num_client_agg in [50, 30, 10]:
            for num_blocks in [3]:
                for block_size in [3, 2, 1]:
                    for optimizer in ['sgd', 'adam']:
                        for lr in [0.0001, 0.001, 0.01, 0.1]:
                            for grad_client in [50.0, 5.0, 1.0, 0.1]:
                                sample_prob = float(num_clients) / float(num_client_agg)
                                num_steps = math.ceil(num_epochs * sample_prob)
                                subprocess.run(['poetry', 'run', 'python', 'trainer_cifar10_no_gp.py',
                                                '--num-steps', str(num_steps),
                                                '--num-clients', str(num_clients),
                                                '--block-size', str(block_size),
                                                '--optimizer', optimizer,
                                                '--lr', str(lr),
                                                '--num-client-agg', str(num_client_agg),
                                                '--num-blocks', str(num_blocks)])

