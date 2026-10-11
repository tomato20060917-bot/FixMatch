"""エントリーポイント。 例: python main.py --num_train_iter 20000"""
import argparse

from config import build_config
from data import build_datasets, build_loaders
from model import build_algorithm
from engine import train, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="FixMatch on CIFAR-10 (semilearn)")
    # 指定した項目だけ DEFAULT_CONFIG を上書きする
    parser.add_argument("--num_labels", type=int, default=None)
    parser.add_argument("--num_train_iter", type=int, default=None)
    parser.add_argument("--num_eval_iter", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print("🌟 コンフィグを読み込んでいます...")
    config = build_config(vars(args))

    print("🚀 データセットを準備中...")
    dataset_dict = build_datasets(config)
    train_lb_loader, train_ulb_loader, eval_loader = build_loaders(config, dataset_dict)

    print("💥 モデルとFixMatchアルゴリズムを構築中...")
    algorithm = build_algorithm(config)

    print("🔥 トレーニング開始！")
    trainer = train(config, algorithm, train_lb_loader, train_ulb_loader, eval_loader)

    print("👀 最終評価！")
    evaluate(trainer, eval_loader)


if __name__ == "__main__":
    main()
