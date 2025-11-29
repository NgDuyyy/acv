from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

HISTORY_PATH = Path('result/log_history/training_history.csv')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training/validation curves from history CSV.')
    parser.add_argument(
        '--history',
        type=Path,
        default=HISTORY_PATH,
        help='Đường dẫn tới training_history.csv (mặc định: result/log_history/training_history.csv).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Nếu chỉ định, lưu biểu đồ ra file thay vì hiển thị cửa sổ.',
    )
    return parser.parse_args()


def _detect_csv_format(csv_path: Path) -> Tuple[str, str]:
    """Infer delimiter and decimal symbol ('.' vs ',')."""

    with csv_path.open('r', encoding='utf-8') as handle:
        header = handle.readline()
    if ';' in header:
        return ';', ','
    return ',', '.'


def _load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'Không tìm thấy file lịch sử: {csv_path}')

    sep, decimal = _detect_csv_format(csv_path)
    df = pd.read_csv(csv_path, sep=sep, decimal=decimal)
    numeric_cols = [col for col in df.columns if col != 'epoch']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['epoch'])
    return df.sort_values('epoch')


def _determine_scale(series: pd.Series, threshold: float = 1e4) -> Tuple[float, str]:
    series = series.dropna().abs()
    if series.empty:
        return 1.0, ''
    median = series.median()
    if median < threshold:
        return 1.0, ''
    power = int(math.floor(math.log10(median)))
    scale = 10 ** power
    return float(scale), f' (÷1e{power})'


def _plot_curves(df: pd.DataFrame) -> plt.Figure:
    loss_scale, loss_suffix = _determine_scale(pd.concat([df['train_loss'], df['val_loss']]))
    bleu_scale, bleu_suffix = _determine_scale(df['val_bleu4'], threshold=100)

    fig, (ax_loss, ax_bleu) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax_loss.plot(df['epoch'], df['train_loss'] / loss_scale, label='Train Loss', color='#1f77b4')
    ax_loss.plot(df['epoch'], df['val_loss'] / loss_scale, label='Validation Loss', color='#ff7f0e')
    ax_loss.set_title('Training & Validation Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel(f'Loss{loss_suffix}')
    ax_loss.grid(alpha=0.2)
    ax_loss.legend()

    ax_bleu.plot(df['epoch'], df['val_bleu4'] / bleu_scale, label='Validation BLEU-4', color='#2ca02c')
    ax_bleu.set_title('BLEU-4 Score over Epochs')
    ax_bleu.set_xlabel('Epoch')
    ax_bleu.set_ylabel(f'BLEU-4{bleu_suffix}')
    ax_bleu.grid(alpha=0.2)

    return fig


def main() -> None:
    args = _parse_args()
    history = _load_history(args.history)
    fig = _plot_curves(history)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f'Đã lưu biểu đồ tại {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
