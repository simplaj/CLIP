import argparse
import os

import torch


# 创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='这是一个示例程序')

# 添加参数
parser.add_argument('--mode', type=str, help='train or test')
parser.add_argument('--path', type=str, default='./')
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=float, default=20)
parser.add_argument('--loade', type=int, default=0)
parser.add_argument('--pname', type=str)
parser.add_argument('--wd', type=float, default=4e-5)

# 解析命令行参数
args = parser.parse_args()
args.path = 'train_results/' + args.path if not args.path == './' else args.path
os.makedirs(args.path, exist_ok=True)