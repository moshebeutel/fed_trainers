
import math
import subprocess

# ["num_blocks", "block_size", "optimizer", "lr", "num_clients_agg", "clip"]]

data_name = 'putEMG'

for num_epochs in [10]:
    for num_clients in [23]:
        for num_client_agg in [5]:
            for num_blocks in [3]:
                for block_size in [1]:
                    for optimizer in ['adam']:
                        for lr in [0.001, 0.0001]:
                            for sigma in [0.0, 1.0, 2.016, 4.72, 12.79, 25.0]:
                                for grad_clip in [0.1, 0.01]:

                                    sample_prob = float(num_clients) / float(num_client_agg)
                                    num_steps = math.ceil(num_epochs * sample_prob)
                                    subprocess.run(['poetry', 'run', 'python', f'trainer_{data_name}_dp_no_gp.py',
                                                    '--data-name', data_name,
                                                    '--num-steps', str(num_steps),
                                                    '--num-clients', str(num_clients),
                                                    '--block-size', str(block_size),
                                                    '--optimizer', optimizer,
                                                    '--lr', str(lr),
                                                    '--num-client-agg', str(num_client_agg),
                                                    '--num-blocks', str(num_blocks),
                                                    '--num-private-clients', str(num_clients),
                                                    '--num-public-clients', '0',
                                                    '--noise-multiplier', str(sigma),
                                                    '--clip', str(grad_clip)])

