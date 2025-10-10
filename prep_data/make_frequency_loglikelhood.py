"""This scripts starts from raw frequency counts in two corpora and compares
them using log-likelihood."""
import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import read_pandas

tqdm.pandas()

def process_targets(w: str) -> tuple[str, str]:
    """Processes <lemma>:<POS> target into lemma and POS tag."""
    lemma = ':'.join(w.split(':')[:-1])
    postag = w.split(':')[-1]
    return lemma, postag


def process_ngrams(ngram: list[str]) -> tuple[str, str, str]:
    """Processes an iterable of <lemma>:<POS> targets (as in a list of ngrams)
    and returns space-joined full target, only lemmas, and only POS. Mainly
    to use on pandas dataframes via df[[col1, ..., coln]].apply(process_ngrams)

    Example:
        input:   ['the:DET', 'paper:NOUN']
        output:  ('the:DET paper:NOUN', 'the paper', 'DET NOUN').
    Args:
        ngram: iterable of <lemma>:<POS> targets to join together.
    Returns:
        Tuple of space-joined full target, lemma, POS tags.
    """
    lemmas = []
    postags = []
    for w in ngram:
        lemma, postag = process_targets(w)
        lemmas.append(lemma)
        postags.append(postag)

    target = ' '.join(ngram)
    lemma = ' '.join(lemmas)
    postag = ' '.join(postags)

    return target, lemma, postag


def load_counts(counts_file: str, fast: bool = False) -> pd.DataFrame:
    """Loads TSV counts file and processes it into a usable dataframe.

    Args:
        counts_file: Path to TSV file. Last column is assumed to contain counts,
            all preceding columns are considered as the target (2+grams if more
            than 1 column) in the format <lemma>:<POS>.
        fast: If True, skips target processing and only determines `freq`.
    Returns:
        Pandas dataframe with columns `target`, `lemma`, `pos`, `freq`."""
    # Load file
    df = read_pandas(counts_file, quoting=3, header=None)

    # Set up columns
    cols = df.columns
    df = df.rename(columns={cols[-1]: 'freq'}) # last column is frequency
    target_cols = list(df.columns)[:-1] # target(s) in preceding column(s)

    if not fast:
        # Figure out if 1-gram or more
        kw = dict(axis=1, result_type='expand')
        if len(target_cols) == 1:
            df = df.rename(columns={0: 'target'})
            df[['lemma', 'pos']] = df.progress_apply(lambda x: process_targets(x.target), **kw)
        elif len(target_cols) > 1:
            df[['target', 'lemma', 'pos']] = df[target_cols].progress_apply(process_ngrams, **kw)
        else:
            raise ValueError("Something's wrong with the dataframe.")

        # Reorder columns
        df = df[['target', 'lemma', 'pos', 'freq']]

    return df


def get_total_size(df: pd.DataFrame | str) -> int:
    """Computes total frequency from (path to) pandas DF with `freq` column."""
    if isinstance(df, str):
        print('Loading unigram counts for total size:\n', df)
        df = load_counts(df, fast=True)
    assert isinstance(df, pd.DataFrame)
    total_size = df['freq'].sum()
    return total_size


def get_ll(freq_target: int, freq_base: int, total_target: int,
           total_base: int) -> float:
    """Computes log-likelihood as per https://ucrel.lancs.ac.uk/llwizard.html

    Args:
        freq_target: Raw frequency of the word in target corpus.
        freq_base: Raw frequency of the word in base corpus.
        total_target: Total size of the target corpus.
        total_base: Total size of the base corpus.
    Returns:
        Computed log-likelihood.
    """

    # doesn't matter which is target/base
    a = freq_target
    b = freq_base
    c = total_target
    d = total_base

    # Calculate expected values
    E1 = c*(a+b) / (c+d)
    E2 = d*(a+b) / (c+d)

    # Calculate log-likelihood, ignoring observed frequencies of 0
    # (see Note 2 in link above)
    to_sum = []
    if a > 0:
        to_sum.append(a*np.log(a/E1))
    if b > 0:
        to_sum.append(b*np.log(b/E2))
    G2 = 2*sum(to_sum)

    return G2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('counts_file1', help='path to first word counts file, '
                        'expected as TSV file with frequency in last column, '
                        'as output by make_frequency_count.py')
    parser.add_argument('counts_file2', help='path to second word counts file')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--join', default='inner', choices=['inner', 'outer'],
                        help='how to join vocabularies being compared, one of '
                        '[inner, outer], defaults to `inner`')
    parser.add_argument('--out_name', help='output file name; if not provided, '
                        'defaults to {counts_file1}_{join}_ll.parquet')
    parser.add_argument('--unigram_file1', help='path to first unigram counts '
                        'file, used to calculate total corpus size if processing'
                        ' ngram counts')
    parser.add_argument('--unigram_file2', help='path to second unigram counts')
    args = parser.parse_args()

    # Load files with word counts
    print('\n'.join(['Loading counts:', args.counts_file1, args.counts_file2]))
    df1 = load_counts(args.counts_file1)
    df2 = load_counts(args.counts_file2)

    print('Starting to compute')

    # Get total corpus sizes
    t1_total = get_total_size(df1 if args.unigram_file1 is None else args.unigram_file1)
    t2_total = get_total_size(df2 if args.unigram_file2 is None else args.unigram_file2)

    # Normalize to frequencies per million
    df1['freq_norm'] = df1['freq'] / t1_total * 10**6
    df2['freq_norm'] = df2['freq'] / t2_total * 10**6

    # Merge DFs
    df = df1.merge(df2, how=args.join, on=['target', 'lemma', 'pos'], suffixes=['1', '2'])
    df = df.fillna(0)

    # Calculate frequency ratios
    df['ratio_t2'] = df['freq_norm2'] / df['freq_norm1']
    df['ratio_t1'] = df['freq_norm1'] / df['freq_norm2']

    # Replace results of zero division with np.nan
    df['ratio_t2'] = df['ratio_t2'].replace([np.inf, -np.inf], np.nan)
    df['ratio_t1'] = df['ratio_t1'].replace([np.inf, -np.inf], np.nan)

    # Calculate log-ratios
    # https://cass.lancs.ac.uk/log-ratio-an-informal-introduction/
    df['logratio_t1'] = df['ratio_t1'].apply(np.log2)
    df['logratio_t2'] = df['ratio_t2'].apply(np.log2)

    # Calculate log-likelihood
    df['ll'] = df.progress_apply(lambda x: get_ll(freq_target=x.freq2,
                                                  freq_base=x.freq1,
                                                  total_target=t2_total,
                                                  total_base=t1_total),
                                 axis=1)

    # Prepare output
    if args.out_name is None:
        args.out_name = f'{Path(args.counts_file1).stem}_{args.join}_ll.parquet'
    out_path = os.path.join(args.out_dir, args.out_name)

    df.to_parquet(out_path)
    print('Output written:', out_path)

if __name__ == '__main__':
    main()