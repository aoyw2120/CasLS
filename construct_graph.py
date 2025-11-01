import math
import pickle

import torch
from torch_geometric.data import Data

from utils import Options
from parsers import create_parser

parser = create_parser()
args = parser.parse_args()


def normalize(x, c):
    if x == 0:
        return 0
    return math.log(x + 1) / math.log(c + 1)


def construct_graph(dataset, data, max_seq, observation):
    """
    This function transforms the data into PyG format.
    Args:
        dataset(str): The name of dataset
        data(list): train data or validation data or test data
        max_seq(int):
        observation()
    Returns:
        graph_list(list): The list of PyG in a batch
        y_true: The total number of cascades i.e. ground truth
    """
    graph_list = []
    y_true = []
    options = Options(dataset)
    with open(options.u2idx_dict, 'rb') as f:
        u2idx = pickle.load(f)
    with open(options.idx2u_dict, 'rb') as f:
        idx2u = pickle.load(f)
    # convert the raw data into a list that stores social relationship
    social_edges = []
    with open(options.net_data, 'r') as handle:
        edge_list = handle.read().strip().split("\n")
        edge_list = [edge.split(',') for edge in edge_list]
        edge_list = [(u2idx[edge[1]], u2idx[edge[0]]) for edge in edge_list if edge[0] in u2idx and edge[1] in u2idx]
        social_edges += edge_list  # 列表，元素是元组，使用的是id，int类型
    # build fans dictionary for conveniently finding fans
    fans_dict = dict()
    for i in range(len(u2idx)):
        fans_dict.setdefault(i, [i])
    for social_edge in social_edges:
        fans_dict[social_edge[0]].append(social_edge[1])  # int int id
    # count historic transmit between two users in observation window
    historic_cnt_dict = dict()
    if 'data1' in options.data:
        for line in open(options.data):  # 这里需要修改？改成data？好像不用改，但是要限定在observation_window里面
            chunks = line.strip().split(',')  # 二维
            start_timestamp = chunks[0].split(' ')[-1]
            end_timestamp = int(start_timestamp) + observation
            for i, chunk in enumerate(chunks):
                if i == len(chunks) - 1 or int(chunks[i + 1].split(' ')[-1]) > end_timestamp:
                    break
                else:
                    key = str(u2idx[chunks[i].split(' ')[-2]]) + ' ' + str(u2idx[chunks[i + 1].split(' ')[-2]])  # strid
                    if key not in historic_cnt_dict:
                        historic_cnt_dict[key] = 1
                    else:
                        historic_cnt_dict[key] += 1
    elif 'data2' in options.data:
        line_cnt = 0
        for line in open(options.data):
            print(line_cnt)
            parts = line.strip().split('\t')
            paths = parts[4].strip().split(' ')
            for path in paths:
                nodes = path.split(':')[0].split('/')  # uid，而且是str  key要改吧？ 改成u2idx
                time = int(path.split(':')[1])
                if time < observation and len(nodes) >= 2:  # list out of range 了，添加判断至少要两个节点
                    key = str(u2idx[nodes[-2]]) + ' ' + str(u2idx[nodes[-1]])
                    if key not in historic_cnt_dict:
                        historic_cnt_dict[key] = 1
                    else:
                        historic_cnt_dict[key] += 1
            line_cnt += 1
    upper_bound = max(historic_cnt_dict.values())
    for fans_dict_key, fans_dict_val in fans_dict.items():
        cnt_list = []
        for item in fans_dict_val[1:]:
            key = str(fans_dict_key) + ' ' + str(item)
            if key not in historic_cnt_dict:
                cnt_list.append(0)
            else:
                cnt_list.append(historic_cnt_dict[key])
        sorted_indices = [index for index, value in sorted(enumerate(cnt_list), key=lambda x: x[1], reverse=True)]
        # sorted_list = sorted(cnt_list, reverse=True)
        if len(sorted_indices) > args.sample_size:
            fans_dict[fans_dict_key] = [fans_dict_val[i + 1] for i in sorted_indices[:args.sample_size]]
        elif 0 < len(sorted_indices) < args.sample_size:
            fans_dict[fans_dict_key] = [fans_dict_val[i + 1] for i in sorted_indices[:len(sorted_indices)]]
        else:
            fans_dict[fans_dict_key] = []
    # build graph
    all_cascades = data[0]  # str id
    all_timestamps = data[1]
    all_len = data[2]
    all_idx = data[3]
    cnt1 = 0  # this counter records current cascade
    for cascade in all_cascades:
        cnt2 = 0  # this counter records current user in a cascade
        start_timestamp = all_timestamps[cnt1][0]
        end_timestamp = start_timestamp + observation
        node_index1 = list()
        node_index2 = list()
        node_timestamp1 = list()
        node_timestamp2 = list()
        edge_strength1 = list()
        edge_strength2 = list()
        raw_edge_list = list()
        for user in cascade:
            if (cnt2 == max_seq or cnt2 == len(cascade) or
                    ('data1' in options.data and all_timestamps[cnt1][cnt2] >= end_timestamp) or
                    ('data2' in options.data and all_timestamps[cnt1][cnt2] >= observation)):
                break
            else:
                key2 = ' '
                if 'data1' in options.data:
                    node_index1.append((int(user), str(all_timestamps[cnt1][cnt2])))
                    node_index2.append(int(user))
                    node_timestamp1.append(all_timestamps[cnt1][cnt2])
                    if cnt2 >= 1:
                        raw_edge_list.append((int(cascade[cnt2 - 1]), str(all_timestamps[cnt1][cnt2 - 1]),
                                              int(cascade[cnt2]), str(all_timestamps[cnt1][cnt2])))
                        key2 = str(cascade[cnt2 - 1]) + ' ' + str(cascade[cnt2])  # str id
                elif 'data2' in options.data and len(user.split('/')) == 1:
                    node_index1.append((int(user), str(all_timestamps[cnt1][cnt2])))
                    node_index2.append(int(user))
                    node_timestamp1.append(all_timestamps[cnt1][cnt2])
                    cnt2 += 1  # 注意
                    continue
                elif 'data2' in options.data and len(user.split('/')) == 2:
                    node_index1.append((int(user.split('/')[1]), str(all_timestamps[cnt1][cnt2])))
                    node_index2.append(int(user.split('/')[1]))
                    node_timestamp1.append(all_timestamps[cnt1][cnt2])
                    flag = 0
                    for i in range(cnt2 - 1, 0, -1):
                        if int(cascade[i].split('/')[-1]) == int(user.split('/')[0]):  # 0621更改
                            flag = 1
                            raw_edge_list.append((int(cascade[i].split('/')[-1]), str(all_timestamps[cnt1][i]),
                                                  int(user.split('/')[1]), str(all_timestamps[cnt1][cnt2])))
                            key2 = str(cascade[i].split('/')[-1]) + ' ' + str(cascade[cnt2].split('/')[1])
                            break
                    if flag == 0:
                        raw_edge_list.append((int(cascade[0].split('/')[-1]), str(all_timestamps[cnt1][0]),
                                              int(user.split('/')[1]), str(all_timestamps[cnt1][cnt2])))
                        key2 = str(cascade[0].split('/')[-1]) + ' ' + str(cascade[cnt2].split('/')[1])
            if key2 in historic_cnt_dict:
                edge_strength1.append(normalize(historic_cnt_dict[key2], upper_bound))
            elif key2 not in historic_cnt_dict and key2 != ' ':
                edge_strength1.append(0)
                print('出现了key不存在的情况')
            cnt2 += 1
        unique_keys = [f"{_}_{ts}" for _, ts in node_index1]
        keys_to_idx = {key: idx for idx, key in enumerate(unique_keys)}  # str int
        _edge_index = []
        for src_id, src_ts, dst_id, dst_ts in raw_edge_list:
            src_key = f"{src_id}_{src_ts}"
            dst_key = f"{dst_id}_{dst_ts}"
            if src_key in keys_to_idx and dst_key in keys_to_idx:
                _edge_index.append((keys_to_idx[src_key], keys_to_idx[dst_key]))
        cnt2 = 0
        cascade_user_set = set()  # int id  和前面合并一下
        for user in cascade:
            if (cnt2 == max_seq or cnt2 == len(cascade) or
                    ('data1' in options.data and all_timestamps[cnt1][cnt2] >= end_timestamp) or
                    ('data2' in options.data and all_timestamps[cnt1][cnt2] >= observation)):
                break
            else:
                if 'data1' in options.data:
                    cascade_user_set.add(int(user))
                elif 'data2' in options.data and len(user.split('/')) == 1:
                    cascade_user_set.add(int(user))
                elif 'data2' in options.data and len(user.split('/')) == 2:
                    cascade_user_set.add(int(user.split('/')[1]))
            cnt2 += 1
        fans_set = set()
        cnt2 = 0
        for user in cascade:  # 处理好友节点
            if (cnt2 == max_seq or cnt2 == len(cascade) or
                    ('data1' in options.data and all_timestamps[cnt1][cnt2] >= end_timestamp) or
                    ('data2' in options.data and all_timestamps[cnt1][cnt2] >= observation)):
                break
            for fan in fans_dict[int(user.split('/')[-1])]:
                if fan not in cascade_user_set and fan != int(user.split('/')[-1]):
                    fans_set.add(fan)
            cnt2 += 1
        node_index3 = list(fans_set)
        cnt3 = len(keys_to_idx)
        for fan in node_index3:
            keys_to_idx[str(fan)] = cnt3  # int int
            node_timestamp2.append(0)
            cnt3 += 1
        cnt2 = 0
        for user in cascade:  # 边索引和边强度
            if (cnt2 == max_seq or cnt2 == len(cascade) or
                    ('data1' in options.data and all_timestamps[cnt1][cnt2] >= end_timestamp) or
                    ('data2' in options.data and all_timestamps[cnt1][cnt2] >= observation)):
                break
            else:
                if 'data1' in options.data:
                    for fan in fans_dict[int(user)]:  # str id
                        if fan not in cascade_user_set and fan != int(user.split('/')[-1]):
                            src_key = f"{user}_{all_timestamps[cnt1][cnt2]}"
                            dst_key = f"{fan}"
                            _edge_index.append((keys_to_idx[src_key], keys_to_idx[dst_key]))
                            edge_strength2.append(0)  #
                elif 'data2' in options.data and len(user.split('/')) == 1:
                    for fan in fans_dict[int(user)]:
                        if fan not in cascade_user_set and fan != int(user.split('/')[-1]):
                            src_key = f"{user}_{all_timestamps[cnt1][cnt2]}"
                            dst_key = f"{fan}"
                            _edge_index.append((keys_to_idx[src_key], keys_to_idx[dst_key]))
                            edge_strength2.append(0)  #
                elif 'data2' in options.data and len(user.split('/')) == 2:
                    for fan in fans_dict[int(user.split('/')[1])]:
                        if fan not in cascade_user_set and fan != int(user.split('/')[-1]):
                            src_key = f"{user.split('/')[-1]}_{all_timestamps[cnt1][cnt2]}"
                            dst_key = f"{fan}"
                            _edge_index.append((keys_to_idx[src_key], keys_to_idx[dst_key]))
                            edge_strength2.append(0)  #
            cnt2 += 1
        graph = Data(x=torch.tensor(node_index2 + node_index3),  # id int tensor
                     node_timestamp=torch.tensor(node_timestamp1 + node_timestamp2),  # int/float tensor
                     edge_index=torch.tensor(_edge_index).t(),
                     edge_strength=torch.tensor(edge_strength1 + edge_strength2)
                     )
        graph_list.append(graph)
        y_true.append(all_len[cnt1])
        cnt1 += 1
    return graph_list, y_true
