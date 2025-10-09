"""This script computes distributional neighborhood density (as average of
cosines between target and its top NNs) from a precomputed similarity matrix."""
import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from utils import read_pandas

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file', help='pandas DF with precomputed similarity '
                        'matrix from w2v, indices+columns as target words')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--out_file', help='output file name; if not provided, '
                        'corresponds to {in_file}_density_k{k}.parquet')
    parser.add_argument('-k', help='number of top nearest neighbors to use; '
                        'defaults to 10', type=int, default=10)
    args = parser.parse_args()

    df = read_pandas(args.in_file)
    assert df.shape[0] == df.shape[1]

    """
    rows = []
    for w in tqdm(df.columns):
        row = df[[w]].drop(index=w)\
                     .sort_values(by=w, ascending=False)[:args.k]\
                     .agg(['mean', 'std']).T
        rows.append(row)
    out_df = pd.concat(rows)"""

    # Get cosines
    rows = {}
    for w in tqdm(df.columns):
        row = df[[w]].drop(index=w)\
              .sort_values(by=w, ascending=False)[:args.k][w]\
              .to_list()
        rows[w] = row

    # Compute mean+std
    out_df = pd.DataFrame.from_dict(rows, orient='index')
    out_df[['mean', 'std']] = out_df.agg(['mean', 'std'], axis=1)

    # Prepare output
    out_cols = ['mean', 'std'] + list(out_df.columns)[:-2]
    out_df = out_df[out_cols]
    out_df = out_df.rename(columns={c: str(c) for c in out_cols})

    if not args.out_file:
        args.out_file = f'{Path(args.in_file).stem}_density_k{args.k}.parquet'
    out_path = os.path.join(args.out_dir, args.out_file)
    out_df.to_parquet(out_path)

if __name__ == '__main__':
    main()