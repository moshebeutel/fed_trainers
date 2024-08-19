import math
import subprocess

# for data_name in ['putEMG', 'cifar10', 'mnist']:
for data_name in ['cifar10']:

    print(f'@@@ *** %%% GEP_PUBLIC  {data_name} %%% *** @@@')

    num_users = 50
    # data_name = 'mnist'
    classes_per_client = 2 if data_name in ['cifar10', 'mnist'] else 20
    script_name = 'cifar10'
    if data_name == 'putEMG':
        script_name = 'putEMG'
        num_users = 44
        classes_per_client = 8
    for num_epochs in [20]:
    # for num_epochs in [3]:
        for num_clients in [num_users]:
            for num_client_agg in [5]:
                for num_blocks in [3]:
                    for block_size in [1]:
                        # for sigma in [0.0, 12.79]:
                        for sigma in [0.0, 2.016, 4.72, 12.79, 25.0]:
                        #     for optimizer in ['adam', 'sgd']:
                            for optimizer in ['adam']:
                                for lr in [0.01]:
                                # for lr in [0.01, 0.001]:
                                    for num_public_clients in [5]:
                                        # for history_size in [50, 100]:
                                        for history_size in [160]:
                                            # for basis_size in [25, 50]:
                                            for basis_size in [0.75*history_size, history_size]:
                                                clip_list = [1.0] if sigma == 0.0 else [0.001]
                                                for grad_clip in clip_list:
                                                # for grad_clip in [1.0, 0.1, 0.01]:
                                                    for seed in [43, 44, 45, 46, 47]:

                                                        print(f'@@@ Run gep_public_no_gp SIGMA {sigma} lr {lr} '
                                                              f'grad_clip {grad_clip} optimizer {optimizer} '
                                                              f'history_size {history_size} basis_size {basis_size}  %%%')

                                                        sample_prob = float(num_clients) / float(num_client_agg)
                                                        num_steps = math.ceil(num_epochs * sample_prob)
                                                        subprocess.run(['poetry', 'run', 'python',
                                                                        f'trainer_{script_name}_gep_public_no_gp.py',
                                                                        '--data-name', data_name,
                                                                        '--classes-per-client', str(classes_per_client),
                                                                        '--num-steps', str(num_steps),
                                                                        '--num-clients', str(num_clients),
                                                                        '--block-size', str(block_size),
                                                                        '--optimizer', optimizer,
                                                                        '--lr', str(lr),
                                                                        '--seed', str(seed),
                                                                        '--num-client-agg', str(num_client_agg),
                                                                        '--num-blocks', str(num_blocks),
                                                                        '--num-private-clients',
                                                                        str(num_clients - num_public_clients),
                                                                        '--num-public-clients', str(num_public_clients),
                                                                        '--noise-multiplier', str(sigma),
                                                                        '--clip', str(grad_clip),
                                                                        '--basis-size', str(int(basis_size)),
                                                                        '--gradients-history-size', str(history_size)],
                                                                       )

