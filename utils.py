import os
import pickle
import random
import time

from collections import Counter
from parsers import create_parser

parser = create_parser()
args = parser.parse_args()


class Options(object):  # 可能要添加项
    def __init__(self, data_name='douban'):
        self.data = 'data/data2/' + data_name + '/cascades.txt'
        self.u2idx_dict = 'data/data2/' + data_name + '/u2idx.pickle'
        self.idx2u_dict = 'data/data2/' + data_name + '/idx2u.pickle'
        self.net_data = 'data/data2/' + data_name + '/edges.txt'
        self.embed_dim = 64


def build_global_edge(options):
    """
    This function builds global relation edges from
    Args:
        options(Object): The object of dataset
    Return:
        None
    """
    global_edges = []
    with open(options.data, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            paths = parts[4].strip().split(' ')
            for path in paths:
                nodes = path.split(':')[0].split('/')
                if len(nodes) >= 2:
                    global_edges.append((nodes[-1], nodes[-2]))  # 原uid，而且是str
    global_save_path = options.net_data
    with open(global_save_path, 'w') as f:
        for item in global_edges:
            f.write(f"{item[0]},{item[1]}\n")


def build_index(raw_data):
    """
    This function builds a mapping between users and indexes.
    :param raw_data: The path of dataset
    :return: user_size(int): The number of users
             u2idx(dictionary): The mapping from users to indexes
             idx2u(list): The mapping from indexed to users
    """
    user_size = int()
    u2idx = {}
    idx2u = []
    if 'data1' in raw_data:
        user_set = set()
        line_num = 0
        for line in open(raw_data):
            line_num += 1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    if len(chunk.split()) == 2:
                        user, timestamp = chunk.split()
                        user_set.add(user)
                    elif len(chunk.split()) == 3:
                        root, user, timestamp = chunk.split()
                        user_set.add(user)
                        user_set.add(root)
                except AttributeError:
                    print(line)
                    print(chunk)
                    print(line_num)
        idx = 0
        for user in user_set:
            u2idx[user] = idx
            idx2u.append(user)
            idx += 1
        user_size = len(user_set)
        print("user_size : %d" % user_size)
    elif 'data2' in raw_data:  # aps, twitter, weibo
        user_set = set()
        line_num = 0
        for line in open(raw_data):
            line_num += 1
            if len(line.strip()) == 0:
                continue
            parts = line.strip().split('\t')  # 0:cascade id 1:publisher 2:publish time 3:total number 4:diffusion path
            paths = parts[4].strip().split(' ')
            for path in paths:
                try:
                    nodes = path.split(':')[0].split('/')
                    # timestamp = int(path.split(':')[1])
                    user_set.update(nodes)
                except AttributeError:
                    print(line)
                    print(path)
                    print(line_num)
        idx = 0
        for user in user_set:
            u2idx[user] = idx
            idx2u.append(user)
            idx += 1
        user_size = len(user_set)
        print("user_size : %d" % user_size)
    return user_size, u2idx, idx2u


def split_data(dataset, train_rate, valid_rate, random_seed):  # 是否筛选有效级联？
    """
    This function filters valid cascades from raw data and splits the dataset into training set, validation set and test
    set.
    Args:
        dataset(str): The name of dataset
        train_rate(float): The rate of training set
        valid_rate(float): The rate of validation set
        random_seed(int): The random seed for shuffling
    Returns:
        user_size(int): The number of user in a dataset
        all_cascades(list): 2-d list data2 only save data in the observation window data use id
        all_timestamps(list):
        train(list): 3-d list: cascades 2-d list timestamps 2-d list len list id list
        valid(list):
        test(list):
    """
    options = Options(dataset)
    if 'edges.txt' not in os.listdir('data/' + 'data2/' + dataset):
        build_global_edge(options)
    if 'idx2u.pickle' in os.listdir('data/' + 'data2/' + dataset):  # 改路径
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)
        user_size = len(u2idx)
    else:
        user_size, u2idx, idx2u = build_index(options.data)
        with open(options.u2idx_dict, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.idx2u_dict, 'wb') as handle:
            pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
    all_cascades = []
    all_timestamps = []
    cascade_len = []
    if 'data1' in options.data:
        for line in open(options.data):
            if len(line.strip()) == 0:
                continue
            users = []
            timestamps = []
            if dataset == 'memetracker':
                chunks = line.strip().split()
            else:
                chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    if dataset == 'memetracker' or dataset == 'weibo':
                        user, timestamp = chunk.split(',')
                        users.append(u2idx[user])  # users添加的是id
                        timestamps.append(float(timestamp))
                    else:
                        if len(chunk.split()) == 2:  # Twitter, Douban
                            user, timestamp = chunk.split()
                            if user in u2idx:
                                users.append(u2idx[user])
                                timestamps.append(float(timestamp))
                        elif len(chunk.split()) == 3:  # Android, Christianity
                            root, user, timestamp = chunk.split()
                            if root in u2idx:
                                users.append(u2idx[root])
                                timestamps.append(float(timestamp))
                            if user in u2idx:
                                users.append(u2idx[user])  # users里的是整数id
                                timestamps.append(float(timestamp))
                except AttributeError:
                    print(chunk)
            if 1 < len(users) <= 500:  # 这个数据集不保留长度超过500的级联
                all_cascades.append(users)  # users是列表，all_cascades是二维列表
                all_timestamps.append(timestamps)
                cascade_len.append(len(users))  # 一维列表
    elif 'data2' in options.data:
        cascades_total = 0
        cascades_valid = 0
        # filtered_cascades = []
        for line in open(options.data):
            users = []
            timestamps = []
            cascades_total += 1
            parts = line.split('\t')
            if dataset == 'weibo':  # 发布时间不对，直接筛掉
                hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8
                if hour < 8 or hour >= 18:
                    continue
            elif dataset == 'twitter':
                month = int(time.strftime('%m', time.localtime(float(parts[2]))))
                day = int(time.strftime('%d', time.localtime(float(parts[2]))))
                if month == 4 and day > 10:
                    continue
            elif dataset == 'aps':
                date = parts[2]
                if date > '1997-12':
                    continue
            paths = parts[4].strip().split(' ')
            observation_path = []
            cnt = 0
            for path in paths:
                nodes = path.split(':')[0].split('/')  # str uid
                timestamp = int(path.split(':')[1])
                if timestamp < args.observation_window:  # 只保留观测窗口内的
                    cnt += 1
                    observation_path.append((nodes, timestamp))  # 是否应该转换成id？这里还使用的uid，列表元素是元组，str列表+整数
            if cnt < 10:  # 过短，筛选掉
                continue
            observation_path.sort(key=lambda tup: tup[1])  # sort according to timestamp in each cascade
            # filtered_data_path = []
            for i in range(len(observation_path)):
                if len(observation_path[i][0]) > 1:  # 可能需要只保留至少有2个用户的×根很重要吧，应该单独保存，使用的是id，而且是字符串
                    users.append(str(u2idx[observation_path[i][0][-2]]) + '/' + str(u2idx[observation_path[i][0][-1]]))
                else:
                    users.append(str(u2idx[observation_path[i][0][-1]]))  # 根节点 str id
                timestamps.append(observation_path[i][1])  # 整数
            all_cascades.append(users)  # users是str id列表，all_cascades是二维列表，users里可能是单个的，也可能是u1/u2
            all_timestamps.append(timestamps)  # all_timestamps是二维列表
            cascade_len.append(int(parts[3]))
            cascades_valid += 1
    '''
    The following block is unnecessary. It exists just to maintain consistency with previous work
    '''
    order = [i[0] for i in sorted(enumerate(all_timestamps), key=lambda x: x[1])]
    all_cascades[:] = [all_cascades[i] for i in order]
    all_timestamps = sorted(all_timestamps)
    cascade_len = [cascade_len[i] for i in order]
    cascade_idx = [i for i in range(len(all_cascades))]
    # Training set
    train_idx_start = int(train_rate * len(all_cascades))
    train_cascades = all_cascades[0:train_idx_start]
    train_timestamps = all_timestamps[0:train_idx_start]
    train_len = cascade_len[0:train_idx_start]
    train_idx = cascade_idx[0:train_idx_start]
    train = [train_cascades, train_timestamps, train_len, train_idx]  # [[[], []], [[], []], [], []]
    # Validation set
    valid_idx_start = int((train_rate + valid_rate) * len(all_cascades))
    valid_cascades = all_cascades[train_idx_start:valid_idx_start]
    valid_timestamps = all_timestamps[train_idx_start:valid_idx_start]
    valid_len = cascade_len[train_idx_start:valid_idx_start]
    valid_idx = cascade_idx[train_idx_start:valid_idx_start]
    valid = [valid_cascades, valid_timestamps, valid_len, valid_idx]
    # Test set
    test_cascades = all_cascades[valid_idx_start:]
    test_timestamps = all_timestamps[valid_idx_start:]
    test_len = cascade_len[valid_idx_start:]
    test_idx = cascade_idx[valid_idx_start:]
    test = [test_cascades, test_timestamps, test_len, test_idx]
    # Random shuffle
    random.seed(random_seed)
    random.shuffle(train_cascades)
    random.seed(random_seed)
    random.shuffle(train_timestamps)
    random.seed(random_seed)
    random.shuffle(train_len)
    random.seed(random_seed)
    random.shuffle(train_idx)

    total_len = sum(len(i) for i in all_cascades)
    train_size = len(train_cascades)
    valid_size = len(valid_cascades)
    test_size = len(test_cascades)
    print("training size:%d valid size:%d testing size:%d" % (train_size, valid_size, test_size))
    print("total size:%d" % (len(all_cascades)))
    print("average length:%f" % (total_len / len(all_cascades)))
    print("maximum length:%f" % (max(len(cas) for cas in all_cascades)))
    print("minimum length:%f" % (min(len(cas) for cas in all_cascades)))
    print("user size:%d" % user_size)
    return user_size, all_cascades, all_timestamps, train, valid, test


def find_max_degree(x, flag):  # x 是一个列表，里面的每一个都是对象，0 代表求出度
    """
    This function is used to calculate maximum value of in-degree or out-degree in a certain batch
    Args:
        x(list):
        flag(int): 0 for out-degree, 1 for in-degree
    Returns:
        max_degree(int):
    """
    max_degree = 0
    if flag == 0:
        for i in x:
            counter = Counter(i.edge_index[0].tolist())
            tmp_max = max(counter.values())
            if tmp_max > max_degree:
                max_degree = tmp_max
    elif flag == 1:
        for i in x:
            counter = Counter(i.edge_index[1].tolist())
            tmp_max = max(counter.values())
            if tmp_max > max_degree:
                max_degree = tmp_max
    return max_degree
