"""ハイパーパラメータ設定の定義と、semilearn用コンフィグへの変換。"""
from semilearn import get_config

DEFAULT_CONFIG = {
    "algorithm": "fixmatch",
    "net": "wrn_28_2",
    # 分散学習は無効化（シングルGPU）
    "distributed": False,
    "multiprocessing_distributed": False,
    "gpu": 0,
    "gpu_id": 0,
    # データセット
    "dataset": "cifar10",
    "num_labels": 40,
    "num_classes": 10,
    "data_dir": "./data",
    "include_lb_to_ulb": True,
    # バッチサイズ
    "batch_size": 64,
    "eval_batch_size": 256,
    "uratio": 7,
    # 保存まわり
    "save_dir": "./saved_models",
    "save_name": "fixmatch_cifar10",
    "resume": False,
    "overwrite": True,
    "use_tensorboard": False,
    # 学習イテレーション（動作確認用の値）
    "epoch": 1,
    "num_train_iter": 500,
    "num_eval_iter": 500,
}


def build_config(overrides=None):
    """デフォルト設定を overrides で上書きして semilearn のコンフィグを返す。"""
    config_dict = dict(DEFAULT_CONFIG)
    if overrides:
        config_dict.update({k: v for k, v in overrides.items() if v is not None})
    return get_config(config_dict)
