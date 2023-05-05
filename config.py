import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float,default=1e-4)

    parser.add_argument('--num-epochs', type=int, default=200)

    parser.add_argument('--seq-len', type=int, default=3)
    parser.add_argument('--prediction-step', type=int, default=1)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--device', type=int, default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
    args_parsed = parser.parse_args()
    return args_parsed