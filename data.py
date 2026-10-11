"""データセットとデータローダーの構築。"""
from semilearn import get_dataset, get_data_loader


def build_datasets(config):
    return get_dataset(
        config,
        config.algorithm,
        config.dataset,
        config.num_labels,
        config.num_classes,
        data_dir=config.data_dir,
        include_lb_to_ulb=config.include_lb_to_ulb,
    )


def build_loaders(config, dataset_dict):
    """(ラベルあり, ラベルなし, 評価) の3つのローダーを返す。"""
    train_lb_loader = get_data_loader(
        config, dataset_dict["train_lb"], config.batch_size
    )
    train_ulb_loader = get_data_loader(
        config, dataset_dict["train_ulb"], int(config.batch_size * config.uratio)
    )
    eval_loader = get_data_loader(
        config, dataset_dict["eval"], config.eval_batch_size
    )
    return train_lb_loader, train_ulb_loader, eval_loader
