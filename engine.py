"""学習と評価の実行。"""
from semilearn import Trainer


def train(config, algorithm, train_lb_loader, train_ulb_loader, eval_loader):
    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
    return trainer


def evaluate(trainer, eval_loader):
    return trainer.evaluate(eval_loader)
