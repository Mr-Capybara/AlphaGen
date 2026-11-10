"""Run the frozen CSI 300 Table 2 configuration and evaluate selected pools."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, stdev
from uuid import uuid4

import numpy as np
import torch

from alphagen.config import OPERATORS
from alphagen.data.expression import Constant, Expression, Feature, Ref
from alphagen.models.alpha_pool import AlphaPool
from alphagen_qlib.stock_data import FeatureType, StockData
from train_maskable_ppo import main as train_one


OPERATORS_BY_NAME = {operator.__name__: operator for operator in [Constant, *OPERATORS]}
FEATURES_BY_NAME = {f'_field_{field.name.lower()}': field for field in FeatureType}


def parse_expression(expression):
    def convert(node):
        if isinstance(node, ast.Name) and node.id in FEATURES_BY_NAME:
            return Feature(FEATURES_BY_NAME[node.id])
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -convert(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            operator = OPERATORS_BY_NAME.get(node.func.id)
            if operator is not None:
                return operator(*(convert(arg) for arg in node.args))
        raise ValueError(f'Invalid AlphaGen expression: {expression}')

    parsed = convert(ast.parse(re.sub(r'\$(\w+)', r'_field_\1', expression), mode='eval').body)
    if not isinstance(parsed, Expression):
        raise ValueError(f'Invalid AlphaGen expression: {expression}')
    return parsed


def best_checkpoint(directory):
    [metrics_path] = directory.glob('checkpoints/*/best_valid.json')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if not all(math.isfinite(metrics[key]) for key in ('ic', 'rank_ic')):
        raise ValueError(f'Non-finite validation metrics in {metrics_path}')
    return metrics, metrics_path.with_name('best_pool.json')


def training(config, output):
    for capacity, steps in config['training_steps'].items():
        for seed in config['seeds']:
            run = output / f'pool-{capacity}-seed-{seed}'
            if run.exists():
                best_checkpoint(run)
                continue
            temporary = run.with_name(f'{run.name}.work-{uuid4().hex[:8]}')
            train_one(seed=seed, pool_capacity=int(capacity), steps=steps,
                      device_name=config['device'], output_dir=str(temporary), keep_best_only=True)
            best_checkpoint(temporary)
            temporary.rename(run)


def testing(config, output):
    records = {int(capacity): [best_checkpoint(output / f'pool-{capacity}-seed-{seed}')
                              for seed in config['seeds']] for capacity in config['training_steps']}
    validation_ic = {capacity: mean(metrics['ic'] for metrics, _ in items)
                     for capacity, items in records.items()}
    capacity = max(validation_ic, key=lambda size: (validation_ic[size], -size))
    data = StockData('csi300', '2020-01-01', '2021-12-31', device=torch.device(config['device']))
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    per_seed = []
    for seed, (validation, checkpoint) in zip(config['seeds'], records[capacity]):
        pool = json.loads(checkpoint.read_text(encoding='utf-8'))
        exprs = [parse_expression(expr) for expr in pool['exprs']]
        weights = np.asarray(pool['weights'], dtype=float)
        if not exprs or weights.shape != (len(exprs),) or not np.isfinite(weights).all():
            raise ValueError(f'Invalid factor pool: {checkpoint}')
        ic, rank_ic = AlphaPool.evaluate_weighted_pool(exprs, weights, data, target)
        if not math.isfinite(ic) or not math.isfinite(rank_ic):
            raise ValueError(f'Non-finite test metrics for seed {seed}')
        per_seed.append({'seed': seed, 'validation': validation, 'ic': ic, 'rank_ic': rank_ic})
    summary = {key: {'mean': mean(row[key] for row in per_seed),
                     'sample_std': stdev(row[key] for row in per_seed)} for key in ('ic', 'rank_ic')}
    (output / 'table2-csi300.json').write_text(json.dumps({
        'selected_capacity': capacity, 'mean_validation_ic': validation_ic,
        'per_seed': per_seed, 'test_summary': summary,
    }, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'test'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    snapshot = Path(config['qlib_dir']).resolve()
    manifest_hash = hashlib.sha256((snapshot / 'snapshot.json').read_bytes()).hexdigest()
    if manifest_hash != config['snapshot_manifest_sha256']:
        parser.error('The Qlib snapshot manifest does not match the experiment configuration')
    os.environ['ALPHAGEN_QLIB_DIR'] = str(snapshot)
    receipt = args.output / 'config.json'
    if receipt.is_file():
        if receipt.read_bytes() != config_bytes:
            parser.error('Output already contains a different experiment configuration')
    elif args.stage == 'train':
        args.output.mkdir(parents=True, exist_ok=False)
        receipt.write_bytes(config_bytes)
    else:
        parser.error('Complete the training stage before evaluating the test set')
    if args.stage == 'train':
        training(config, args.output)
    else:
        testing(config, args.output)


if __name__ == '__main__':
    main()
