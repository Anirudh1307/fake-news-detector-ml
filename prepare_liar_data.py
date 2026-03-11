import argparse
from pathlib import Path

import pandas as pd

TSV_COLUMNS = [
    'id',
    'label',
    'statement',
    'subject',
    'speaker',
    'speaker_job_title',
    'state_info',
    'party_affiliation',
    'barely_true_counts',
    'false_counts',
    'half_true_counts',
    'mostly_true_counts',
    'pants_on_fire_counts',
    'context',
]

FAKE_LABELS = {'false', 'barely-true', 'pants-fire'}
REAL_LABELS = {'true', 'mostly-true'}
HALF_TRUE_LABEL = 'half-true'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LIAR train/valid/test TSV files into Fake.csv and True.csv.'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing train.tsv, valid.tsv, and test.tsv (default: data).',
    )
    parser.add_argument(
        '--half-true',
        choices=['drop', 'fake', 'real'],
        default='drop',
        help='How to map half-true labels (default: drop).',
    )
    parser.add_argument(
        '--fake-out',
        default='data/Fake.csv',
        help='Output path for fake rows (default: data/Fake.csv).',
    )
    parser.add_argument(
        '--true-out',
        default='data/True.csv',
        help='Output path for real rows (default: data/True.csv).',
    )
    return parser.parse_args()


def load_liar_splits(data_dir):
    split_paths = [Path(data_dir) / name for name in ('train.tsv', 'valid.tsv', 'test.tsv')]
    missing = [str(path) for path in split_paths if not path.exists()]
    if missing:
        missing_text = ', '.join(missing)
        raise FileNotFoundError(f"Missing required TSV files: {missing_text}")

    frames = [pd.read_csv(path, sep='\t', names=TSV_COLUMNS) for path in split_paths]
    return pd.concat(frames, ignore_index=True)


def map_binary_labels(df, half_true_mode):
    labels = df['label'].astype(str).str.strip().str.lower()
    fake_labels = set(FAKE_LABELS)
    real_labels = set(REAL_LABELS)

    if half_true_mode == 'fake':
        fake_labels.add(HALF_TRUE_LABEL)
    elif half_true_mode == 'real':
        real_labels.add(HALF_TRUE_LABEL)

    usable = labels.isin(fake_labels.union(real_labels))
    mapped = df[usable].copy()
    mapped['is_fake'] = labels[usable].isin(fake_labels).astype(int)
    return mapped


def build_output_frames(df):
    output = df.copy()
    output['statement'] = output['statement'].fillna('').astype(str).str.strip()
    output['context'] = output['context'].fillna('').astype(str).str.strip()
    output['title'] = output['statement']
    output['text'] = (output['statement'] + ' ' + output['context']).str.strip()

    output = output[(output['title'] != '') & (output['text'] != '')].copy()
    output = output.drop_duplicates(subset=['title', 'text'])

    fake_df = output[output['is_fake'] == 1][['title', 'text']].copy()
    true_df = output[output['is_fake'] == 0][['title', 'text']].copy()
    return fake_df, true_df


def main():
    args = parse_args()
    df = load_liar_splits(args.data_dir)
    mapped = map_binary_labels(df, args.half_true)
    fake_df, true_df = build_output_frames(mapped)

    Path(args.fake_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.true_out).parent.mkdir(parents=True, exist_ok=True)
    fake_df.to_csv(args.fake_out, index=False)
    true_df.to_csv(args.true_out, index=False)

    original_label_counts = df['label'].value_counts().to_dict()
    print(f"Loaded rows: {len(df)}")
    print(f"Original label counts: {original_label_counts}")
    print(f"half-true mode: {args.half_true}")
    print(f"Saved fake rows: {len(fake_df)} -> {args.fake_out}")
    print(f"Saved real rows: {len(true_df)} -> {args.true_out}")


if __name__ == '__main__':
    main()
