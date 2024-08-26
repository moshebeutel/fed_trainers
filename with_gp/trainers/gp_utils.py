import copy
from collections import defaultdict
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from common.utils import detach_to_numpy, get_optimizer, get_device


@torch.no_grad()
def eval_model(args, global_model, client_ids, train_loaders, eval_loaders, GPs):
    # results: defaultdict[int, defaultdict[str, float]] = defaultdict()
    results = defaultdict(lambda: defaultdict(list))

    targets = []
    preds = []
    step_results = []
    device = get_device()
    global_model.eval()
    num_clients = len(client_ids)

    for client_id in range(num_clients):
        is_first_iter = True
        running_loss, running_correct, running_samples = 0., 0., 0.

        test_loader = eval_loaders[client_id]
        train_loader = train_loaders[client_id]

        # build tree at each step
        GPs[client_id], label_map, Y_train, X_train = build_tree(global_model, client_id, train_loader, GPs)
        GPs[client_id].eval()
        client_data_labels = []
        client_data_preds = []

        for batch_count, batch in enumerate(test_loader):
            img, label = tuple(t.to(device) for t in batch)
            Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype,
                                  device=label.device)

            X_test = global_model(img)
            loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)
            batch_size = Y_test.shape[0]
            running_loss += (loss.item() * batch_size)
            running_correct += pred.argmax(1).eq(Y_test).sum().item()
            running_samples += batch_size

            is_first_iter = False
            targets.append(Y_test)
            preds.append(pred)

            client_data_labels.append(Y_test)
            client_data_preds.append(pred)

        # calculate confusion matrix
        cm = confusion_matrix(detach_to_numpy(torch.cat(client_data_labels, dim=0)),
                              detach_to_numpy(torch.max(torch.cat(client_data_preds, dim=0), dim=1)[1]))

        # save classification results to output structure
        step_results.append({"id": client_id,
                             "cm": cm,
                             "y_true": detach_to_numpy(torch.cat(client_data_labels, dim=0)),
                             "y_pred": detach_to_numpy(torch.max(torch.cat(client_data_preds, dim=0), dim=1)[1])})

        # erase tree (no need to save it)
        GPs[client_id].tree = None

        results[client_id]['loss'] = running_loss / running_samples
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)

    return results, labels_vs_preds, step_results


@torch.no_grad()
def build_tree(net, client_id, loader, GPs: torch.nn.ModuleList):
    """
    Build GP tree per client
    :return: List of GPs
    """
    device = get_device()
    for k, batch in enumerate(loader):
        batch = (t.to(device) for t in batch)
        train_data, clf_labels = batch

        z = net(train_data)
        X = torch.cat((X, z), dim=0) if k > 0 else z
        Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels

    # build label map
    client_labels, client_indices = torch.sort(torch.unique(Y))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                 device=Y.device)

    GPs[client_id].build_base_tree(X, offset_labels)  # build tree
    return GPs[client_id], label_map, offset_labels, X


def local_train(args, net, train_loader,
                client_id: int,
                GPs: torch.nn.ModuleList,
                pbar: tqdm, pbar_dict: Dict):
    local_net = copy.deepcopy(net)
    local_net.train()
    optimizer = get_optimizer(args, local_net)
    device = get_device()
    train_avg_loss = 0.0

    # build tree at each step
    GPs[client_id], label_map, _, __ = build_tree(local_net, client_id, train_loader, GPs)
    GPs[client_id].train()

    for i in range(args.inner_steps):
        for k, batch in enumerate(train_loader):
            x, label = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # forward prop
            pred = local_net(x)

            X = torch.cat((X, pred), dim=0) if k > 0 else pred
            Y = torch.cat((Y, label), dim=0) if k > 0 else label

        offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                     device=Y.device)

        loss = GPs[client_id](X, offset_labels, to_print=args.eval_every)
        # loss *= args.loss_scaler

        # propagate loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
        optimizer.step()

        train_avg_loss += loss.item() * offset_labels.shape[0]

        pbar_dict.update({"Inner Step": f'{(i + 1)}'.zfill(3),
                          "Train Current Loss": f'{loss.item():5.2f}'})
        pbar.set_postfix(pbar_dict)

        # end of for k, batch in enumerate(train_loader):
    # end of for i in range(args.inner_steps):

    return local_net, train_avg_loss
