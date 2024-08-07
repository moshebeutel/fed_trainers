import math
import subprocess

# ["num_blocks", "block_size", "optimizer", "lr", "num_clients_agg", "clip"]]
num_users = 50
data_name = 'putEMG'
classes_per_client = 2 if data_name == 'cifar10' else 20
script_name = 'cifar10'
if data_name == 'putEMG':
    script_name = 'putEMG'
    num_users = 16
    classes_per_client = 8

for num_epochs in [30]:
    for num_clients in [num_users]:
        for num_client_agg in [5]:
            for num_blocks in [3]:
                for block_size in [1]:
                    # for sigma in [0.0]:
                    for sigma in [0.0, 2.016, 4.72, 12.79, 25.0]:
                        for optimizer in ['adam', 'sgd']:
                        # for optimizer in ['adam']:
                        #     for lr in [0.001]:
                            for lr in [0.01, 0.001, 0.0001]:
                                # for history_size in [50]:
                                for history_size in [50, 100]:
                                    # for grad_clip in [1.0]:
                                    for grad_clip in [10.0, 5.0, 1.0, 0.1, 0.01]:
                                        print(f'@@@ Run gep_no_gp SIGMA {sigma} lr {lr} '
                                              f'grad_clip {grad_clip} optimizer {optimizer} '
                                              f'history_size {history_size}  %%%')
                                        sample_prob = float(num_clients) / float(num_client_agg)
                                        num_steps = math.ceil(num_epochs * sample_prob)
                                        subprocess.run(['poetry', 'run', 'python', f'trainer_{script_name}_gep_no_gp.py',
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
                                                        '--clip', str(grad_clip),
                                                        '--basis-gradients-history-size', str(history_size)],
                                                       )
