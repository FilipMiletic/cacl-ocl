"""This script makes a TSV word frequency count, given a corpus file."""
import argparse
import os
from collections import Counter
from nltk.util import ngrams
from pathlib import Path
from smart_open import open
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file', help='one-sentence-per-line corpus file '
                        '(supports compressed formats)')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--ngrams', type=int, default=1, help='number of ngrams'
                        ' to use for frequency count')
    parser.add_argument('--out-file', help='output file name; if not provided, '
                        'corresponds to {in_file}_{ngrams}_freqs.txt')
    args = parser.parse_args()

    # Load data and count
    counts = Counter()
    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        # tokens = process_line(line)
        tokens = line.strip().split()
        if args.ngrams > 1:
            tokens = ['\t'.join(ngram) for ngram in ngrams(tokens, args.ngrams)]
        counts.update(tokens)

    # Prepare counts for writing
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    out_lines = []
    for word, count in counts:
        out_line = f'{word}\t{count}\n'
        out_lines.append(out_line)

    # Set up output file name + print
    if args.out_file is None:
        args.out_file = f'{Path(args.in_file).stem}_{args.ngrams}grams_freqs.txt'
    out_path = os.path.join(args.out_dir, args.out_file)
    with open(out_path, 'w') as f:
        f.writelines(out_lines)


if __name__ == '__main__':
    main()