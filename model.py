"""ネットワークとアルゴリズムの構築。"""
from semilearn import get_net_builder, get_algorithm


def build_algorithm(config):
    # from_name=False: USB内蔵のネットワークビルダーを使う
    net_builder = get_net_builder(config.net, from_name=False)
    return get_algorithm(config, net_builder, tb_log=None, logger=None)
