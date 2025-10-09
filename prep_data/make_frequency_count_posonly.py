"""Converts token/lemma+POS frequency counts into POS-only frequency counts,
mostly relevant for analyzing morphosyntactic patterns based on ngrams."""
import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file', help='frequency file, tab-separated, where '
                        'last column contains frequency counts and all '
                        'preceding columns are treated as ngrams of format '
                        'word:POSTAG, as output by `make_frequency_count.pos`')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--out-file', help='output file name; if not provided, '
                        'corresponds to {in_file}_posonly.txt')
    args = parser.parse_args()

    tqdm.pandas()

    # Load file
    df = pd.read_csv(args.in_file, sep='\t', quoting=3, header=None)

    # Strip down to POS tags only
    for col in df.columns[:-1]:
        df[col] = df[col].progress_apply(lambda x: x.split(':')[-1])

    # Sum up frequencies
    df = df.groupby(list(df.columns[:-1])).sum().reset_index()

    # Write output file
    if args.out_file is None:
        args.out_file = f'{Path(args.in_file).stem}_posonly.txt'
    out_path = os.path.join(args.out_dir, args.out_file)
    df.to_csv(out_path, sep='\t', quoting=3, header=None, index=None)

if __name__ == '__main__':
    main()