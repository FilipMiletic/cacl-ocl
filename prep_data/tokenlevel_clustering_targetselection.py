"""This script selects targets for clustering based on previously computed
log-likelihood differences across corpora. Specifically, it applies different
possible filters (POS tags, minimum character length, alphabetical-only). It
selects top N words for each of t1 and t2, determining which of the two
based on the frequency ratio above/below 1, and then within that selection
sorting on log-likelihood."""
import argparse
import pandas as pd
from utils import read_pandas

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file', help='pandas dataframe with log-likelihood '
                        'info output by make_frequency_loglikelihood.py')
    parser.add_argument('out_file', help='output path to dump .txt list; '
                        'appends if file exists already')
    parser.add_argument('--topn', type=int, default=100, help='how many words '
                        'to keep for EACH side (topn highest + topn lowest)')
    parser.add_argument('--pos', type=str, help='POS tag, takes all if not set')
    parser.add_argument('--min_char', help='minimum (inclusive) number of chars '
                        'for targets, takes all if not set', type=int, default=0)
    parser.add_argument('--no_alpha_filter', action='store_false',
                        help='overrides filter keeping only alphabetic targets')
    args = parser.parse_args()

    # Load everything
    df = read_pandas(args.in_file)

    # POS filter
    if args.pos is not None:
        df = df[df['pos'] == args.pos]

    # Number of characters
    df['nchar'] = df['lemma'].str.len()
    df = df[df['nchar'] >= args.min_char]

    # Only alphabetic
    if args.no_alpha_filter: # default value is True
        df = df[df['lemma'].str.isalpha()]

    # More frequent in T1
    top_t1 = df[(df['ratio_t2'] < 1)].sort_values(by='ll', ascending=False)\
               ['target'][:args.topn].to_list()
    top_t2 = df[(df['ratio_t2'] > 1)].sort_values(by='ll', ascending=False)\
               ['target'][:args.topn].to_list()

    # Write output
    with open(args.out_file, 'a') as f:
        for w in top_t1 + top_t2:
            f.write(w+'\n')

    print('Done.')

if __name__ == '__main__':
    main()