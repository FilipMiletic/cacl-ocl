"""This script compares the distribution of neighborhood densities for the same
target across two corpora (= two sets of cosine similarities between target and
its neighbors), computes Mann-Whitney--U over those, and outputs a summary
comparison (mean+std of density for each corpus; statistics)."""
import argparse
import os
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm
from utils import read_pandas

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file1', help='path to neighborhood density data '
                        'from corpus 1, output by `w2v_neighb_density.py`: '
                        'dataframe with `mean`, `std`, and individual cosines')
    parser.add_argument('in_file2', help='path to neighborhood density data '
                        'from corpus 2')
    parser.add_argument('out_dir', help='path to output directory')
    parser.add_argument('--out_file', help='output file name; if not specified '
                        'defaults to {in_file1}_compare.parquet')
    args = parser.parse_args()

    tqdm.pandas()

    # Load files
    df1 = read_pandas(args.in_file1)
    df2 = read_pandas(args.in_file2)
    assert df1.shape[1] == df2.shape[1]
    print('\n'.join(['Loaded dataframes:', args.in_file1, args.in_file2]))

    old_cols = ['mean', 'std']

    # Make new dataframe with cosine distributions as a list in a single column
    df1['cosines'] = df1.drop(columns=old_cols).apply(lambda x: list(x), axis=1)
    df2['cosines'] = df2.drop(columns=old_cols).apply(lambda x: list(x), axis=1)
    df = df1[['cosines']].join(df2[['cosines']], rsuffix='2', how='inner')

    # Run Mann-Whitney and tidy up
    df['mw'] = df.progress_apply(lambda x: mannwhitneyu(x.cosines, x.cosines2),
                                 axis=1)
    df[['stat', 'pval']] = pd.DataFrame(df['mw'].to_list(), index=df.index)

    # Join back with mean+std info
    old_df = df1[old_cols].join(df2[old_cols], lsuffix='1', rsuffix='2', how='inner')
    df = old_df.join(df[['stat', 'pval']])

    # Prepare output
    if not args.out_file:
        args.out_file = f'{Path(args.in_file1).stem}_compare.parquet'
    out_path = os.path.join(args.out_dir, args.out_file)
    df.to_parquet(out_path)
    print('Output written to:', out_path)

if __name__ == '__main__':
    main()