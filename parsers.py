import argparse

"""
Dataset | Observation Time                                      |
----------------------------------------------------------------|
weibo   | 1800(0.5 hour) / 3600 (1 hour)                        |
twitter | 3600*24*1 (86400, 1 day) / 3600*24*2 (172800, 2 days) |
aps     | 365*3 (1095, 3 years) / 365*5 (1825, 5 years)         |
"""


def create_parser():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset", default='twitter', type=str, help="Dataset name")
    parser.add_argument('--train_rate', default=0.8, type=float, help="Training set ratio")
    parser.add_argument('--valid_rate', default=0.1, type=float, help="Validation set ratio")
    parser.add_argument('--random_seed', default=300, type=int, help="Random seed for ")
    # cascade prediction
    parser.add_argument("--observation_window", default=172800, type=int, help="Observation time")
    # parser.add_argument("--prediction_time", default=3600*24*32, type=int, help="Prediction time")
    # cascade
    parser.add_argument("--max_seq", default=100, type=int, help="Max length of cascade sequence")
    parser.add_argument("--sample_size", default=5, type=int, help="Sample size of friend")
    # model
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--emb_dim", default=64, type=int, help="Embedding dimension")
    parser.add_argument("--units", default=128, type=int, help="Dimension of ?")
    parser.add_argument("--patience", default=10, type=int, help="Early stopping patience")
    parser.add_argument("--epoch", default=50, type=int, help="Train epochs")
    parser.add_argument("--model_path", default='./best_model/my_model.pth', type=str, help="Path to save model")
    return parser
