
import math
import subprocess

# ["num_blocks", "block_size", "optimizer", "lr", "num_clients_agg", "clip"]]

for num_epochs in [10]:
    for num_clients in [30]:
        for num_client_agg in [10]:
            for num_blocks in [3]:
                for block_size in [1]:
                    for optimizer in ['adam']:
                        for lr in [0.001]:
                            for sigma in [1.0, 2.016, 4.72, 12.79, 25.0]:
                                for num_public_clients in [5, 10]:
                                    for history_size in [50, 100]:
                                        for grad_clip in [0.1]:

                                            sample_prob = float(num_clients) / float(num_client_agg)
                                            num_steps = math.ceil(num_epochs * sample_prob)
                                            subprocess.run(['poetry', 'run', 'python', 'trainer_cifar10_gep_public_no_gp.py',
                                                            '--num-steps', str(num_steps),
                                                            '--num-clients', str(num_clients),
                                                            '--block-size', str(block_size),
                                                            '--optimizer', optimizer,
                                                            '--lr', str(lr),
                                                            '--num-client-agg', str(num_client_agg),
                                                            '--num-blocks', str(num_blocks),
                                                            '--num-private-clients', str(num_clients - num_public_clients),
                                                            '--num-public-clients', str(num_public_clients),
                                                            '--noise-multiplier', str(sigma),
                                                            '--clip', str(grad_clip),
                                                            '--basis-gradients-history-size', str(history_size)],
                                                           )

