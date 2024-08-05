# #!/bin/env python
# import argparse
# import logging
# import math
# from collections import OrderedDict, defaultdict
# from multiprocessing import Process
# from pathlib import Path
# import numpy as np
# import torch
# import torch.utils.data
# from tqdm import trange
# import copy
# # from models.feature_emg_convnet import FeatureEmgConvnet
# from models.model3d import RawEmg3DConvnet
# from pFedGP.Learner import pFedGPFullLearner, pFedGPIPDataLearner, pFedGPIPComputeLearner
# from backbone import CNNTarget
# from p_fed_gp_emg.utils import get_device, set_logger, set_seed, detach_to_numpy, calc_metrics, str2bool
#
# import os
# import glob
# import pickle
# import re
# from typing import List, Dict
#
# import pandas as pd
# from sklearn.metrics import confusion_matrix
#
# import putemg_features
# from putemg_features import biolab_utilities
# import wandb
#
# import trainer_sgd_dp_no_gp
# import utils
#
# parser = argparse.ArgumentParser(description="Personalized Federated Learning")
#
# parser.add_argument("--raw", type=str2bool, default=True, help="Work on raw data or preprocessed features")
# parser.add_argument("--putemg_feature_folder", type=str, default='../features-dataframes')
# parser.add_argument("--putemg_reduced_dataframes", type=str, default='../reduced_dataframes')
# parser.add_argument("--putemg_raw_folder", type=str, default='../../putemg-downloader/Data-HDF5/')
# parser.add_argument("--result_folder", type=str, default='../wandb/results')
# parser.add_argument("--saved_models_folder", type=str, default='../saved_models/', help="Path to saved models folder")
# # parser.add_argument("--result_folder", type=str, default='../shallow_learn_results/')
# parser.add_argument("--nf", type=str2bool, default=True)
# parser.add_argument("--nc", type=str2bool, default=True)
# parser.add_argument("--read-every-to-files", type=str2bool, default=False)
# ##################################
# #       Optimization args        #
# ##################################
# parser.add_argument("--num-steps", type=int, default=400)
# parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'])
# parser.add_argument("--batch-size", type=int, default=8)
# parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
# parser.add_argument("--num-client-agg", type=int, default=5, help="number of kernels")
# parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
# parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
# parser.add_argument("--grad-clip-norm", type=float, default=0.01)
# parser.add_argument("--dp-noise", type=float, default=0.027)
# parser.add_argument("--dp-noise-multiplier", type=float, default=0)
# parser.add_argument("--desired-epsilon", type=float, default=0.1)
# # parser.add_argument("--lot", type=int, default=)
# ################################
# #       GP args        #
# ################################
# parser.add_argument('--kernel-baseline-pth', type=str, default='saved_kernel_full_acc_0.76923.pth',
#                     choices=['', 'saved_base_kernel.pth', 'saved_kernel_full_acc_0.76923.pth'],
#                     help='saved trained kernel weights filename')
# parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
# parser.add_argument('--kernel-function', type=str, default='RBFKernel',
#                     choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
#                     help='kernel function')
# parser.add_argument('--model-variant', type=str, default='ip_data',
#                     choices=['full', 'ip_data', 'ip_compute'],
#                     help='model variants with or without  inducing points learned or compute ')
# parser.add_argument('--embed-dim', type=int, default=84)
# parser.add_argument('--objective', type=str, default='predictive_likelihood',
#                     choices=['predictive_likelihood', 'marginal_likelihood'])
# parser.add_argument('--predict-ratio', type=float, default=0.5,
#                     help='ratio of samples to make predictions for when using predictive_likelihood objective')
# parser.add_argument('--num-inducing-points', type=int, default=100, help='number of inducing points per class')
# parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
# parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
# parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
# parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
# parser.add_argument('--outputscale', type=float, default=8., help='output scale')
# parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
# parser.add_argument('--outputscale-increase', type=str, default='constant',
#                     choices=['constant', 'increase', 'decrease'],
#                     help='output scale increase/decrease/constant along tree')
# parser.add_argument('--balance-classes', type=str2bool, default=False,
#                     help='Balance classes dist. per client in PredIP')
# #############################
# #       General args        #
# #############################
# parser.add_argument("--num-workers", type=int, default=1, help="number of workers")
# parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
# parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
# parser.add_argument("--eval-every", type=int, default=50, help="Eval model every X selected steps")
# parser.add_argument("--save-every", type=int, default=2, help="Save model every X selected evaluations")
# parser.add_argument("--save-path", type=str, default=(Path.home() / 'saved_models').as_posix(),
#                     help="dir path for saved models")parser.add_argument("--seed", type=int, default=42, help="seed value")
# parser.add_argument('--wandb', type=str2bool, default=False)
# ##########################################################
# #       Test Convnet Backbone args                       #
# ##########################################################
# parser.add_argument("--backbone", type=str, choices=['simple', 'conv3d'], default='conv3d',
#                     help="Type of backbone")
# parser.add_argument("--backbone-output", type=int, default=64, help="backbone output size")
# parser.add_argument("--depthwise-multiplier", type=int, default=32, help="Depthwise multiplier")
# # parser.add_argument("--window_size", type=int, default=1, help="window size for 3d backbone")
# parser.add_argument("--window_size", type=int, default=260, help="window size for 3d backbone")
# parser.add_argument('--raw-every', type=int, default=50, help="take every n-th raw reading ")
#
# args = parser.parse_args()
#
# log_level = logging.INFO
#
# set_logger(level=log_level)
# logging.getLogger().setLevel(log_level)
#
# logging.info(f'Logger set. Log level  = {logging.getLevelName(logging.getLogger().getEffectiveLevel())}')
# logging.debug(f'window size = {args.window_size}')
# set_seed(args.seed)
#
# exp_name = f'IP_DP_using_{args.kernel_baseline_pth}_{args.model_variant}_clip_{args.grad_clip_norm}_Noise_{args.dp_noise}'
# logging.info(f'Experiment Name : {exp_name}')
#
# logging.info(f'Experiment Name : {exp_name}')
# # Weights & Biases
# if args.wandb:
#     wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
#     wandb.config.update(args)
#
# #
# # if args.exp_name != '':
# #     exp_name += '_' + args.exp_name
# #
# # logging.info(str(args))
# # args.out_dir = (Path(args.save_path) / exp_name).as_posix()
# # out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
# # logging.info(out_dir)
#
#
# putemg_folder = os.path.abspath(args.putemg_feature_folder) if not args.raw \
#     else os.path.abspath(args.putemg_raw_folder)
# if not os.path.isdir(putemg_folder):
#     logging.error('{:s} is not a valid folder'.format(putemg_folder))
#     exit(1)
#
# result_folder = os.path.abspath(args.result_folder)
# if not os.path.isdir(result_folder):
#     logging.error('{:s} is not a valid folder'.format(result_folder))
#     exit(1)
# if not args.raw:
#     filtered_data_folder = os.path.join(result_folder, 'filtered_data')
#     calculated_features_folder = os.path.join(result_folder, 'calculated_features')
#
# # list all hdf5 files in given input folder
# all_files = [f for f in sorted(glob.glob(os.path.join(putemg_folder, "*.hdf5")))]
# # Moshe take only gesture sequential for part of users
# user_trains = [f'emg_gestures-{user}-{traj}' for user in ['04', '05', '06', '07'] for traj in
#                ['sequential', 'repeats_short', 'repeats_long']]
# train_user_files = [f for f in all_files if any([a for a in user_trains if a in f])]
# all_files = train_user_files
# # if not skipped filter the input data and save to consequent output files
# if not args.nf:
#     # create folder for filtered data
#     if not os.path.exists(filtered_data_folder):
#         os.makedirs(filtered_data_folder)
#
#     # by each filename in download folder
#     for file in all_files:
#         basename = os.path.basename(file)
#         filename = os.path.splitext(basename)[0]
#         print('Denoising file: {:s}'.format(basename))
#
#         # read raw putEMG data file and run filter
#         df: pd.DataFrame = pd.read_hdf(file)
#         biolab_utilities.apply_filter(df)
#
#         # save filtered data to designated folder with prefix filtered_
#         output_file = filename + '_filtered.hdf5'
#         print('Saving to file: {:s}'.format(output_file))
#         df.to_hdf(os.path.join(filtered_data_folder, output_file),
#                   'data', format='table', mode='w', complevel=5)
# else:
#     logging.info('Denoising skipped!')
#
# # if not skipped calculate features from filtered files
# if not args.nc:
#     # create folder for calculated features
#     if not os.path.exists(calculated_features_folder):
#         os.makedirs(calculated_features_folder)
#
#     # by each filename in download folder
#     for file in all_files:
#         basename = os.path.basename(file)
#         filename = os.path.splitext(basename)[0]
#
#         filtered_file_name = filename + '_filtered.hdf5'
#         print('Calculating features for {:s} file'.format(filtered_file_name))
#
#         # for filtered data file run feature extraction, use xml with limited feature set
#         ft: pd.DataFrame = putemg_features.features_from_xml('./features_shallow_learn.xml',
#                                                              os.path.join(filtered_data_folder, filtered_file_name))
#
#         # save extracted features file to designated folder with features_filtered_ prefix
#         output_file = filename + '_filtered_features.hdf5'
#         print('Saving result to {:s} file'.format(output_file))
#         ft.to_hdf(os.path.join(calculated_features_folder, output_file),
#                   'data', format='table', mode='w', complevel=5)
# else:
#     logging.info('Feature extraction skipped!')
#
# # create list of records
# all_feature_records = [biolab_utilities.Record(os.path.basename(f)) for f in all_files]
# logging.info(all_feature_records)
# # data can be additionally filtered based on subject id
# records_filtered_by_subject = biolab_utilities.record_filter(all_feature_records)
# logging.debug(records_filtered_by_subject)
# # records_filtered_by_subject = record_filter(all_feature_records,
# #                                             whitelists={"id": ["01", "02", "03", "04", "07"]})
# # records_filtered_by_subject = pu.record_filter(all_feature_records, whitelists={"id": ["01"]})
# dropped_cols = ['type', 'subject', 'trajectory', 'date_time', 'TRAJ_GT_NO_FILTER', 'VIDEO_STAMP']
# if args.read_every_to_files:
#     def write_reduced_df_to_file(fullname, filename, raw_every):
#         df = pd.DataFrame(pd.read_hdf(fullname))
#         df = df[::raw_every]
#         df.drop(dropped_cols, axis=1, inplace=True)
#         df.to_hdf(f"{filename}_every_{raw_every}.hdf5", key='df', mode='w')
#         del df
#
#
#     for r in records_filtered_by_subject:
#         logging.info(f"Reading dataframe input file: {r}")
#         filename = os.path.splitext(r.path)[0]
#         fullname = os.path.join(calculated_features_folder,
#                                 filename + '_filtered_features.hdf5') if not args.raw else os.path.join(putemg_folder,
#                                                                                                         filename + '.hdf5')
#         logging.info(f"Spawn subprocess for Reading dataframe input file: {r}")
#         p = Process(target=write_reduced_df_to_file, args=(fullname, filename, args.raw_every))
#         p.start()
#         p.join()
#         logging.info(f"subprocess returned for Reading dataframe input file: {r}")
#
# # load feature data to memory
# dfs: Dict[biolab_utilities.Record, pd.DataFrame] = {}
# for r in records_filtered_by_subject:
#     logging.info(f"Reading dataframe input file: {r}")
#     filename = os.path.splitext(r.path)[0]
#     fullname = os.path.join(calculated_features_folder, filename + '_filtered_features.hdf5') if not args.raw \
#         else os.path.join(args.putemg_reduced_dataframes, f"{filename}_every_{args.raw_every}.hdf5")
#     dfs[r] = pd.DataFrame(pd.read_hdf(fullname))
#     for col in dropped_cols:  # add dummy cols. prepare_data in putemg_utilities expects them
#         dfs[r][col] = [1] * dfs[r].shape[0]
# logging.info(f'Finished reading \n{records_filtered_by_subject}\n to memory.\nStart processing.')
# # create k-fold validation set, with 3 splits - for each experiment day 3 combination are generated
# # this results in 6 data combination for each subject
# splits_all = biolab_utilities.data_per_id_and_date(records_filtered_by_subject, n_splits=2)
#
# device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
# device = 'cpu'
# logging.info(f'device = {device}')
# # defines feature sets to be used in shallow learn
# feature_sets = {
#     "RMS": ["RMS"],
#     # "Hudgins": ["MAV", "WL", "ZC", "SSC"],
#     # "Du": ["IAV", "VAR", "WL", "ZC", "SSC", "WAMP"]
# } if not args.raw else {'EMG': ["EMG"]}
#
# # defines gestures to be used in shallow learn
# gestures = {
#     0: "Idle",
#     1: "Fist",
#     2: "Flexion",
#     3: "Extension",
#     4: "Pinch index",
#     5: "Pinch middle",
#     6: "Pinch ring",
#     7: "Pinch small"
# }
# global_label_map = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5, 8: 6, 9: 7}
# num_classes = args.backbone_output
# classes_per_client = 8
# num_clients = len(splits_all.values())
#
# # defines channels configurations for which classification will be run
# channel_range = {
#     "24chn": {"begin": 1, "end": 24},
#     # "8chn_1band": {"begin": 1, "end": 8},
#     # "8chn_2band": {"begin": 9, "end": 16},
#     # "8chn_3band": {"begin": 17, "end": 24}
# }
#
#
# @torch.no_grad()
# def eval_model(global_model, GPs, feature_set_name, features, X_bar=None):
#     results = defaultdict(lambda: defaultdict(list))
#     targets = []
#     preds = []
#     step_results = []
#
#     global_model.eval()
#     logging.info('eval_model')
#     for client_id in range(num_clients):
#         running_loss, running_correct, running_samples = 0., 0., 0.
#         logging.info(f'eval_model client_id {client_id}')
#         # iterate over each internal data
#         for i_s, subject_data in enumerate(list(splits_all.values())[client_id]):
#             logging.info(f'eval_model client_id {client_id} subject_data {subject_data}')
#             is_first_iter = True
#             # get data of client
#             # prepare training and testing set based on combination of k-fold split, feature set and gesture set
#             # this is also where gesture transitions are deleted from training and test set
#             # only active part of gesture performance remains
#             data = biolab_utilities.prepare_data(dfs, subject_data, features, list(gestures.keys()))
#
#             # list columns containing only feature data
#             regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
#             cols = list(filter(regex.search, list(data["train"].columns.values)))
#
#             # strip columns to include only selected channels, eg. only one band
#             cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_') + 1:]) <= ch_range["end"])]
#
#             # extract limited training x and y, only with chosen channel configuration
#             train_x = torch.tensor(data["train"][cols].to_numpy(), dtype=torch.float32)
#             # logging.info(train_x.shape)
#             num_windows = train_x.shape[0] // args.window_size
#             train_x = train_x[:(num_windows * args.window_size), :].reshape(-1, args.window_size, n_features)
#             # logging.info(train_x.shape)
#
#             train_y = torch.LongTensor(data["train"]["output_0"].to_numpy())
#             # logging.info(train_y.shape)
#             train_y = train_y[: (num_windows * args.window_size)]
#             # logging.info(train_y.shape)
#             train_y = train_y[::args.window_size]
#             # logging.info(train_y.shape)
#
#             # # extract limited testing x and y, only with chosen channel configuration
#             test_x = torch.tensor(data["test"][cols].to_numpy(), dtype=torch.float32)
#             # logging.info(train_x.shape)
#             num_windows = test_x.shape[0] // args.window_size
#             test_x = test_x[:(num_windows * args.window_size), :].reshape(-1, args.window_size, n_features)
#             # logging.info(train_x.shape)
#             test_y_true = torch.LongTensor(data["test"]["output_0"].to_numpy())
#             # logging.info(train_y.shape)
#             test_y_true = test_y_true[: (num_windows * args.window_size)]
#             # logging.info(train_y.shape)
#             test_y_true = test_y_true[::args.window_size]
#             # logging.info(train_y.shape)
#             if test_x.shape[0] != test_y_true.shape[0]:
#                 logging.warning(f"test_x.shape {test_x.shape} test_y_true.shape {test_y_true.shape}")
#                 print()
#             train_loader = torch.utils.data.DataLoader(
#                 torch.utils.data.TensorDataset(train_x, train_y),
#                 shuffle=False,
#                 batch_size=args.batch_size,
#                 num_workers=args.num_workers
#             )
#
#             test_loader = torch.utils.data.DataLoader(
#                 torch.utils.data.TensorDataset(test_x, test_y_true),
#                 shuffle=False,
#                 batch_size=args.batch_size,
#                 num_workers=args.num_workers
#             )
#
#             # build tree at each step
#             GPs[client_id], label_map, Y_train, X_train = build_tree(global_model, client_id, train_loader, X_bar)
#             GPs[client_id].eval()
#
#             if X_bar is not None:
#                 client_X_bar = X_bar[list(label_map.values()), ...]
#                 # client_X_bar = X_bar[list(label_map.keys()), ...]
#             client_data_labels = []
#             client_data_preds = []
#
#             invalid_batch_count = 0
#             for batch_count, batch in enumerate(test_loader):
#                 img, label = tuple(t.to(device) for t in batch)
#                 if set(label.tolist()) & set(label_map.keys()) != set(label.tolist()):
#                     logging.warning(
#                         f'Not all labels in label_map label {set(label.tolist())} label_map {set(label_map.keys())}')
#                     invalid_batch_count += 1
#                     continue
#
#                 Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype,
#                                       device=label.device)
#
#                 X_test = global_model(img)
#
#                 if X_bar is not None:
#                     loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, client_X_bar,
#                                                              is_first_iter)
#                 else:
#                     loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)
#
#                 running_loss += loss.item()
#                 running_correct += pred.argmax(1).eq(Y_test).sum().item()
#                 running_samples += len(Y_test)
#
#                 is_first_iter = False
#                 targets.append(Y_test)
#                 preds.append(pred)
#
#                 client_data_labels.append(Y_test)
#                 client_data_preds.append(pred)
#
#             # calculate confusion matrix
#             cm = confusion_matrix(detach_to_numpy(torch.cat(client_data_labels, dim=0)),
#                                   detach_to_numpy(torch.max(torch.cat(client_data_preds, dim=0), dim=1)[1]))
#
#             # save classification results to output structure
#             step_results.append({"id": client_id, "split": i_s, "clf": 'pFedGP',
#                                  "feature_set": feature_set_name,
#                                  "cm": cm, "y_true": detach_to_numpy(torch.cat(client_data_labels, dim=0)),
#                                  "y_pred": detach_to_numpy(torch.max(torch.cat(client_data_preds, dim=0), dim=1)[1])})
#
#         # erase tree (no need to save it)
#         GPs[client_id].tree = None
#         batch_count -= invalid_batch_count
#         results[client_id]['loss'] = running_loss / float(batch_count + 1)
#         results[client_id]['correct'] = running_correct
#         results[client_id]['total'] = running_samples
#
#     target = detach_to_numpy(torch.cat(targets, dim=0))
#     full_pred = detach_to_numpy(torch.cat(preds, dim=0))
#     labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)
#
#     return results, labels_vs_preds, step_results
#
#
# def get_optimizer(network, curr_x_bar=None):
#     if curr_x_bar is not None:
#         params = [
#             {'params': curr_X_bar},
#             {'params': network.parameters()}
#         ]
#     return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
#         if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)
#
#
# @torch.no_grad()
# def build_tree(net, client_id, loader, curr_x_bar=None):
#     """
#     Build GP tree per client
#     :return: List of GPs
#     """
#
#     for k, batch in enumerate(loader):
#         batch = (t.to(device) for t in batch)
#         train_data, clf_labels = batch
#
#         z = net(train_data)
#         X = torch.cat((X, z), dim=0) if k > 0 else z
#         Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels
#
#     # build label map
#     client_labels, client_indices = torch.sort(torch.unique(Y))
#     label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
#     offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
#                                  device=Y.device)
#     if curr_x_bar is not None:
#         GPs[client_id].build_base_tree(X, offset_labels, curr_x_bar)  # build tree
#     else:
#         GPs[client_id].build_base_tree(X, offset_labels)  # build tree
#     return GPs[client_id], label_map, offset_labels, X
#
#
# criteria = torch.nn.CrossEntropyLoss()
#
# ###############################
# # init net and GP #
# ###############################
# for ch_range_name, ch_range in channel_range.items():
#     logging.info("======================== " + ch_range_name + " =======================")
#
#     output: Dict[str, any] = dict()
#
#     output["gestures"] = gestures
#     output["classifiers"] = {"pFedGP": {"predictor": "pFedGP", "args": {}}}
#     output["feature_sets"] = feature_sets
#     output["results"]: List[Dict[str, any]] = list()
#
#     # for each feature set
#     for feature_set_name, features in feature_sets.items():
#         logging.info("======================== " + feature_set_name + " =======================")
#
#         if ch_range_name == '24chn':
#             n_features = 24 if feature_set_name in ["RMS",
#                                                     "EMG"] else 144 if feature_set_name == "Du" else 96  # "Hudgins"
#         else:  # 8chn_2band
#             n_features = 8 if feature_set_name in ["RMS",
#                                                    "EMG"] else 48 if feature_set_name == "Du" else 32  # "Hudgins"
#
#         clients = splits_all
#         gp_counter = 0
#
#         # NN
#         net = CNNTarget(n_features=n_features) if args.backbone == 'simple' else \
#             RawEmg3DConvnet(number_of_classes=num_classes,
#                             window_size=args.window_size,
#                             depthwise_multiplier=args.depthwise_multiplier,
#                             logger=logging.getLogger())
#         # FeatureEmgConvnet(number_of_class=num_classes, channels=1).to(device)
#         if args.kernel_baseline_pth:
#             net.load_state_dict(torch.load(args.saved_models_folder + args.kernel_baseline_pth, map_location=device))
#         net = net.to(device)
#
#         GPs = torch.nn.ModuleList([])
#         for client_id in range(num_clients):
#             learner = pFedGPFullLearner(args, classes_per_client) if args.model_variant == 'full' \
#                 else pFedGPIPDataLearner(args, classes_per_client) if args.model_variant == 'ip_data' \
#                 else pFedGPIPComputeLearner(args, classes_per_client)  # if args.model_variant == 'ip_compute'
#             GPs.append(learner)  # GP instances
#
#         # Inducing locations
#         if args.model_variant != 'full':
#             # X_bar = torch.nn.Parameter(
#             #     torch.randn((num_classes, args.num_inducing_points, args.embed_dim), device=device) * 0.01,
#             #     requires_grad=True)
#             X_bar = torch.nn.Parameter(torch.randn((len(gestures), args.num_inducing_points, num_classes),
#                                                    device=device) * 0.01, requires_grad=True)
#
#         results = defaultdict(list)
#
#         ################
#         # init metrics #
#         ################
#         last_eval = -1
#         best_step = -1
#         best_acc = -1
#         best_val_loss = -1
#         test_best_based_on_step, test_best_min_based_on_step = -1, -1
#         test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
#         step_iter = trange(args.num_steps)
#         test_avg_loss = 10
#         test_avg_acc = 0
#         backprops_count = 0
#         for step in step_iter:
#
#             # print tree stats every 100 epochs
#             to_print = True if step % 100 == 0 else False
#
#             # select several clients
#             client_ids = np.random.choice(num_clients, size=args.num_client_agg, replace=False)
#
#             # initialize global model params
#             params = OrderedDict()
#             for n, p in net.named_parameters():
#                 params[n] = torch.zeros_like(p.data)
#
#             if args.model_variant != 'full':
#                 # initialize inducing points
#                 X_bar_step = torch.zeros_like(X_bar.data, device=device)
#
#             # iterate over each client
#             train_avg_loss = 0
#             num_samples = 0
#
#             for j, client_id in enumerate(client_ids):
#                 # privacy_engine_initialized = False
#                 curr_global_net = copy.deepcopy(net)
#                 trainer_sgd_dp_no_gp.train()
#
#                 if args.model_variant != 'full':
#                     curr_X_bar = copy.deepcopy(X_bar)
#                     optimizer = get_optimizer(curr_global_net, curr_X_bar)
#                 else:
#                     optimizer = get_optimizer(curr_global_net)
#
#                 # get the first value to
#                 # values_view = splits_all.values()
#                 # value_iterator = iter(values_view)
#                 # first_value = next(value_iterator)
#
#                 # iterate over each internal data
#                 for subject_data in list(splits_all.values())[client_id]:
#
#                     # get data of client
#                     # prepare training and testing set based on combination of k-fold split, feature set and gesture set
#                     # this is also where gesture transitions are deleted from training and test set
#                     # only active part of gesture performance remains
#                     data = biolab_utilities.prepare_data(dfs, subject_data, features, list(gestures.keys()))
#                     # for df in dfs.values():
#                     #     df.drop(utils.COLS_TO_DROP, axis=1, inplace=True)
#
#                     # list columns containing only feature data
#                     regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
#                     cols = list(filter(regex.search, list(data["train"].columns.values)))
#
#                     # strip columns to include only selected channels, eg. only one band
#                     cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_') + 1:]) <= ch_range["end"])]
#
#                     # extract limited training x and y, only with chosen channel configuration
#                     train_x = torch.tensor(data["train"][cols].to_numpy(), dtype=torch.float32)
#                     # logging.info(train_x.shape)
#                     num_windows = train_x.shape[0] // args.window_size
#                     train_x = train_x[:(num_windows * args.window_size), :].reshape(-1, args.window_size, n_features)
#                     # logging.info(train_x.shape)
#
#                     train_y = torch.LongTensor(data["train"]["output_0"].to_numpy())
#                     # logging.info(train_y.shape)
#                     train_y = train_y[: (num_windows * args.window_size)]
#                     # logging.info(train_y.shape)
#                     train_y = train_y[::args.window_size]
#                     # logging.info(train_y.shape)
#
#                     train_loader = torch.utils.data.DataLoader(
#                         torch.utils.data.TensorDataset(train_x, train_y),
#                         shuffle=True,
#                         batch_size=args.batch_size,
#                         num_workers=args.num_workers
#                     )
#                     logging.debug(f'len(train_loader){len(train_loader)}')
#
#                     # if privacy_engine_initialized:
#                     #     logging.debug('Define PrivacyEngine')
#                     #     privacy_engine = PrivacyEngine()
#                     #
#                     #     curr_global_net, optimizer, train_loader = privacy_engine.make_private(
#                     #         module=curr_global_net,
#                     #         optimizer=optimizer,
#                     #         data_loader=train_loader,
#                     #         noise_multiplier=args.dp_noise_multiplier,
#                     #         max_grad_norm=args.grad_clip_norm,
#                     #     )
#                     #     privacy_engine_initialized = True
#                     # else:
#                     #     logging.debug('Define Differentially Private dataloader')
#                     #     train_loader = DPDataLoader.from_data_loader(train_loader, distributed=False)
#
#                     # build tree at each step
#                     if args.model_variant != 'full':
#                         GPs[client_id], label_map, _, __ = build_tree(curr_global_net, client_id, train_loader,
#                                                                       curr_X_bar)
#
#                     else:
#                         GPs[client_id], label_map, _, __ = build_tree(curr_global_net, client_id, train_loader)
#                     GPs[client_id].train()
#
#                     for i in range(args.inner_steps):
#
#                         # init optimizers
#                         optimizer.zero_grad()
#
#                         for k, batch in enumerate(train_loader):
#                             batch = (t.to(device) for t in batch)
#                             img, label = batch
#
#                             z = curr_global_net(img)
#                             X = torch.cat((X, z), dim=0) if k > 0 else z
#                             Y = torch.cat((Y, label), dim=0) if k > 0 else label
#
#                         offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=torch.int64,
#                                                      device=Y.device)
#                         logging.debug(f'X.shape = {X.shape}')
#                         logging.debug(f'offset_labels.shape = {offset_labels.shape}')
#                         logging.debug(f'X = {X}')
#                         logging.debug(f'offset_labels= {offset_labels}')
#
#                         if args.model_variant != 'full':
#                             client_X_bar = curr_X_bar[list(label_map.values()), ...]
#                             # client_X_bar = curr_X_bar[list(label_map.keys()), ...]
#                             loss = GPs[client_id](X, offset_labels, client_X_bar, to_print=to_print)
#                         else:
#                             loss = GPs[client_id](X, offset_labels, to_print=to_print)
#                         loss *= args.loss_scaler
#
#                         # propagate loss
#                         loss.backward()
#
#                         # max_grad_before_clip = max([torch.abs(torch.max(p.grad)).item()
#                         #                             for p in curr_global_net.parameters()])
#
#                         # ret_norm = torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), args.grad_clip_norm)
#
#                         # max_grad_after_clip = max([torch.abs(torch.max(p.grad)).item()
#                         #                             for p in curr_global_net.parameters()])
#
#                         # Our clipping & noising
#                         for p in curr_global_net.parameters():
#                             # Clip Gradients
#                             grad_norm = float(torch.linalg.vector_norm(p.grad))
#                             p.grad = torch.clamp(p.grad, min=-args.grad_clip_norm, max=args.grad_clip_norm)
#                             # Add DP noise to gradients
#                             noise = torch.randn_like(p.grad) * args.dp_noise
#                             p.grad += noise
#
#                         # Paper clipping & noising
#                         grad_norm = 0
#                         for p in curr_global_net.parameters():
#                             # Sum grid norms
#                             grad_norm += float(torch.linalg.vector_norm(p.grad))
#                         for p in curr_global_net.parameters():
#                             # Clip gradients
#                             p.grad /= max(1, grad_norm / args.grad_clip_norm)
#                             # Add DP noise to gradients
#                             noise = torch.randn_like(p.grad) * args.dp_noise
#                             p.grad += noise
#                             p.grad /= args.batch_size # use batch as the paper's 'lot'
#
#                         optimizer.step()
#                         backprops_count += 1
#                         if args.wandb:
#                             wandb.log(
#                                 {
#                                     # 'clip_ret_norm': ret_norm,
#                                     'backprops_count': backprops_count,
#                                     # 'max_grad_before_clip':max_grad_before_clip,
#                                     # 'max_grad_after_clip':max_grad_after_clip
#                                     # 'noise_norm_sum': noise_norm_sum,
#                                     # 'grad_norm_sum': grad_norm_sum,
#                                     # 'single_backprop_delta_args': single_backprop_delta_args,
#                                     # 'single_backprop_delta_measured': single_backprop_delta_measured
#                                 }
#                             )
#                         train_avg_loss += loss.item() * offset_labels.shape[0]
#                         num_samples += offset_labels.shape[0]
#
#                         step_iter.set_description(
#                             f"Step: {step + 1}, client: {client_id}, Inner Step: {i}, Loss: {loss.item()}"
#                         )
#                     # end of: for i in range(args.inner_steps):
#                 for n, p in curr_global_net.named_parameters():
#                     params[n] += p.data
#                 if args.model_variant != 'full':
#                     X_bar_step += curr_X_bar.data
#                 # erase tree (no need to save it)
#                 GPs[client_id].tree = None
#
#             train_avg_loss /= num_samples
#
#             # average parameters
#             for n, p in params.items():
#                 params[n] = p / args.num_client_agg
#
#             # update new parameters
#             net.load_state_dict(params)
#             if args.model_variant != 'full':
#                 X_bar.data = X_bar_step.data / args.num_client_agg
#             # if (step + 1) == args.num_steps:
#             if (step % args.eval_every) == (args.eval_every - 1) or (step + 1) == args.num_steps:
#                 if args.model_variant != 'full':
#                     test_results, labels_vs_preds_val, step_results = eval_model(net, GPs, feature_set_name, features,
#                                                                                  X_bar)
#                 else:
#                     test_results, labels_vs_preds_val, step_results = eval_model(net, GPs, feature_set_name, features, )
#
#                 test_avg_loss, test_avg_acc = calc_metrics(test_results)
#                 logging.info(f"Step: {step + 1}, Test Loss: {test_avg_loss:.4f},  Test Acc: {test_avg_acc:.4f}")
#                 for i in step_results:
#                     output["results"].append(i)
#
#                 if best_acc < test_avg_loss:
#                     best_val_loss = test_avg_loss
#                     best_acc = test_avg_acc
#                     best_step = step
#
#                     best_model = copy.deepcopy(net)
#                     if args.model_variant != 'full':
#                         best_X_bar = copy.deepcopy(X_bar)
#                     best_labels_vs_preds_val = labels_vs_preds_val
#                     torch.save(net.state_dict(), args.saved_models_folder
#                                + f'saved_kernel_{args.model_variant}_acc_{best_acc}.pth')
#                 else:
#                     save_every_steps = args.eval_every * args.save_every
#
#                     if (step % save_every_steps) == (save_every_steps - 1):
#                         torch.save(net.state_dict(), args.saved_models_folder
#                                    + f'saved_kernel_{args.model_variant}_acc_{best_acc}.pth')
#
#             if args.wandb:
#                 wandb.log(
#                     {
#                         'custom_step': step,
#                         'train_loss': train_avg_loss,
#                         'test_avg_loss': test_avg_loss,
#                         'test_avg_acc': test_avg_acc,
#                         'best_acc': best_acc,
#                         'best_val_loss': best_val_loss
#                     }
#                 )
#
#     # for each channel configuration dump classification results to file
#     file = os.path.join(result_folder, "classification_result_" + exp_name + "_" + ch_range_name + ".bin")
#     pickle.dump(output, open(file, "wb"))
#     if args.wandb:
#         wandb.save(file)
