import argparse
import json
import os
from typing import Optional
from datetime import datetime
from math import isfinite

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, SingleAlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 valid_data: StockData,
                 valid_target: Expression,
                 keep_best_only: bool = False,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.valid_data = valid_data
        self.valid_target = valid_target
        self.keep_best_only = keep_best_only
        self.best_valid_ic = float('-inf')

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        if not self.pool.size:
            return
        metrics = dict(zip(('ic', 'rank_ic'), self.pool.test_ensemble(self.valid_data, self.valid_target)))
        for name, value in metrics.items():
            self.logger.record(f'valid/{name}', value)
        if not all(map(isfinite, metrics.values())):
            return
        if self.keep_best_only and metrics['ic'] <= self.best_valid_ic:
            return
        self.best_valid_ic = metrics['ic']
        self.save_checkpoint(metrics)

    def save_checkpoint(self, metrics: dict):
        filename = 'best' if self.keep_best_only else f'{self.num_timesteps}_steps'
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', filename)
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)
        with open(f'{path}_valid.json', 'w') as f:
            json.dump(dict(metrics, steps=self.num_timesteps), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    device_name: str = 'auto',
    output_dir: str = 'out',
    keep_best_only: bool = False,
):
    reseed_everything(seed)

    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    close = Feature(FeatureType.CLOSE)
    horizon = 20
    target = Ref(close, -horizon) / close - 1

    data = StockData(instrument=instruments,
                     start_time='2009-01-01',
                     end_time='2018-12-31', device=device, label_horizon=horizon)
    data_valid = StockData(instrument=instruments,
                           start_time='2019-01-01',
                           end_time='2019-12-31', device=device, label_horizon=horizon)
    pool = AlphaPool(
        capacity=pool_capacity,
        stock_data=data,
        target=target,
        ic_lower_bound=None
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=not keep_best_only)

    name_prefix = f"kdd_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path=os.path.join(output_dir, 'checkpoints'),
        valid_data=data_valid,
        valid_target=target,
        keep_best_only=keep_best_only,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log=os.path.join(output_dir, 'tensorboard'),
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train one KDD AlphaGen factor pool.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instruments', default='csi300')
    parser.add_argument('--pool-capacity', type=int, choices=(10, 20, 50, 100), default=10)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--device', dest='device_name', default='auto')
    parser.add_argument('--output-dir', default='out')
    parser.add_argument('--keep-best-only', action='store_true')
    main(**vars(parser.parse_args()))
