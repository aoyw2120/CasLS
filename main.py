import gc
import time

import torch
import torch.nn as nn
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

from construct_graph import construct_graph
from metric import MSLELoss
from dataset import MyDataset
from model import GraphormerNeuralSDEModel
from parsers import create_parser
from utils import *

parser = create_parser()
args = parser.parse_args()


def train_epoch(model, train_loader, optimizer, criterion, epoch_i):
    model.train()
    train_loss = 0
    train_mape = 0
    loop = tqdm(train_loader, desc=f'Training Epoch [{epoch_i}]')
    for batch_index, data_label in enumerate(loop):
        data = []
        label = []
        for item in data_label:
            data.append(item[0])
            label.append(item[1])
        gpu_data = [item.to('cuda') for item in data]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(1)
        label = label.to('cuda')
        # forward propagation
        optimizer.zero_grad()
        # start_1 = time.time()
        pred = model(gpu_data)
        # end_1 = time.time()
        # print('forward time:'+str(end_1-start_1))
        loss = criterion(pred, label)
        # back propagation
        # start_2 = time.time()
        loss.backward()
        # end_2 = time.time()
        # print('backward time:'+str(end_2-start_2))
        # start_3 = time.time()
        optimizer.step()
        # end_3 = time.time()
        # print('update time:'+str(end_3-start_3))

        train_loss += loss.item()  # loss的尺寸？

        epsilon = 1e-8
        # mape = torch.mean(torch.abs((label - pred) / (label + epsilon)))
        mape = torch.mean(
            torch.abs((torch.log2(label + epsilon) - torch.log2(pred + epsilon)) / torch.log2(label + epsilon)))
        train_mape += mape.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_mape = train_mape / len(train_loader)

    return avg_train_loss, avg_train_mape


def valid_epoch(model, val_loader, criterion, epoch_i):
    model.eval()
    val_loss = 0
    val_mape = 0
    loop = tqdm(val_loader, desc=f'Validating Epoch [{epoch_i}]')
    with torch.no_grad():
        for batch_idx, data_label in enumerate(loop):
            data = []
            label = []
            for item in data_label:
                data.append(item[0])
                label.append(item[1])
            gpu_data = [item.to('cuda') for item in data]
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(1)
            label = label.to('cuda')
            pred = model(gpu_data)
            loss = criterion(pred, label)
            val_loss += loss.item()

            epsilon = 1e-8
            # mape = torch.mean(torch.abs((label - pred) / (label + epsilon)))
            mape = torch.mean(
                torch.abs((torch.log2(label + epsilon) - torch.log2(pred + epsilon)) / torch.log2(label + epsilon)))
            val_mape += mape.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = val_mape / len(val_loader)
        return avg_val_loss, avg_val_mape


def test_epoch(model, test_loader):
    model.eval()
    total_msle = 0.0
    total_mape = 0.0
    # total_samples = 0
    loop = tqdm(test_loader, desc=f'Testing')
    with torch.no_grad():
        for batch_idx, data_label in enumerate(loop):
            data = []
            label = []
            for item in data_label:
                data.append(item[0])
                label.append(item[1])
            gpu_data = [item.to('cuda') for item in data]
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(1)
            label = label.to('cuda')
            pred = model(gpu_data)
            # calculate MSLE
            msle = torch.mean((torch.log2(pred + 1) - torch.log2(label + 1)) ** 2)
            total_msle += msle.item()
            # calculate MAPE
            epsilon = 1e-8
            mape = torch.mean(
                torch.abs((torch.log2(label + epsilon) - torch.log2(pred + epsilon)) / torch.log2(label + epsilon)))
            total_mape += mape.item()
        avg_msle = total_msle / len(test_loader)
        avg_mape = total_mape / len(test_loader)
        return avg_msle, avg_mape


def main():
    data_process_start_time = time.time()
    # user_size, all_cascades, all_timestamps, train, valid, test = split_data(args.dataset, args.train_rate,  original
                                                                             # args.valid_rate, args.random_seed)
    user_size = 490474  # new
    '''if 'train_graph_list.pickle' not in os.listdir('cache'):  original
        train_graph_list, train_y = construct_graph(args.dataset, train, args.max_seq, args.observation_window)
        val_graph_list, valid_y = construct_graph(args.dataset, valid, args.max_seq, args.observation_window)
        test_graph_list, test_y = construct_graph(args.dataset, test, args.max_seq, args.observation_window)
        with open('cache/train_graph_list.pickle', 'wb') as handle:
            pickle.dump(train_graph_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/valid_graph_list.pickle', 'wb') as handle:
            pickle.dump(val_graph_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/test_graph_list.pickle', 'wb') as handle:
            pickle.dump(test_graph_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/train_y.pickle', 'wb') as handle:
            pickle.dump(train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/valid_y.pickle', 'wb') as handle:
            pickle.dump(valid_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/test_y.pickle', 'wb') as handle:
            pickle.dump(test_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:'''
    with open('cache/train_graph_list.pickle', 'rb') as handle:
        train_graph_list = pickle.load(handle)
    with open('cache/valid_graph_list.pickle', 'rb') as handle:
        val_graph_list = pickle.load(handle)
    with open('cache/test_graph_list.pickle', 'rb') as handle:
        test_graph_list = pickle.load(handle)
    with open('cache/train_y.pickle', 'rb') as handle:
        train_y = pickle.load(handle)
    with open('cache/valid_y.pickle', 'rb') as handle:
        valid_y = pickle.load(handle)
    with open('cache/test_y.pickle', 'rb') as handle:
        test_y = pickle.load(handle)
    # del all_cascades, all_timestamps, train, valid, test  original
    gc.collect()
    train_set = MyDataset(train_graph_list, train_y)
    val_set = MyDataset(val_graph_list, valid_y)
    test_set = MyDataset(test_graph_list, test_y)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, collate_fn=lambda x: x)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0, collate_fn=lambda x: x)

    first_batch = next(iter(train_loader))
    print(first_batch)

    data_process_end_time = time.time()
    print("Data processing finished! Time used: {:.3f} minutes.".
          format((data_process_end_time - data_process_start_time) / 60))
    # model, optimizer, loss
    train_out = find_max_degree(train_graph_list, 0) # 176~183 is original
    val_out = find_max_degree(val_graph_list, 0)
    test_out = find_max_degree(test_graph_list, 0)
    out_degree = max(train_out, val_out, test_out)
    train_in = find_max_degree(train_graph_list, 1)
    val_in = find_max_degree(val_graph_list, 1)
    test_in = find_max_degree(test_graph_list, 1)
    in_degree = max(train_in, val_in, test_in)
    model = GraphormerNeuralSDEModel(user_size, in_degree, out_degree)  # model = GraphormerNeuralSDEModel(user_size, 64, 64)
    model = model.to('cuda')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)  #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True

    criterion = MSLELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    train_loss_list = []
    train_mape_list = []
    val_loss_list = []
    val_mape_list = []
    for epoch_i in range(args.epoch):
        print(f'======================== Epoch {epoch_i + 1} ========================')
        start_train = time.time()
        train_loss, train_mape = train_epoch(model, train_loader, optimizer, criterion, epoch_i + 1)
        end_train = time.time()
        train_loss_list.append(train_loss)
        train_mape_list.append(train_mape)
        print('===== Train')
        print(f'Mean Prediction loss at epoch{epoch_i + 1}: {train_loss}, MAPE: {train_mape}')
        print(f'Train time at epoch{epoch_i + 1}: {end_train - start_train} second')
        start_val = time.time()
        val_loss, val_mape = valid_epoch(model, val_loader, criterion, epoch_i + 1)
        end_val = time.time()
        val_loss_list.append(val_loss)
        val_mape_list.append(val_mape)
        print('===== Validation')
        print(f'Mean Prediction loss at epoch{epoch_i + 1}: {val_loss}, MAPE: {val_mape}')
        print(f'Train time at epoch{epoch_i + 1}: {end_val - start_val} second')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    model.load_state_dict(torch.load(args.model_path))
    result_msle, result_mape = test_epoch(model, test_loader)
    print(f'Test result: MSLE: {result_msle}, MAPE: {result_mape}')
    print(train_loss_list)
    print(train_mape_list)
    print(val_loss_list)
    print(val_mape_list)


if __name__ == '__main__':
    main()
