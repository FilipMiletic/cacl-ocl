"""This script processes text chunks into one-sentence-per-line format.

As input, it takes a dataframe whose rows contain e.g. paragraphs or sentence
windows. It passes those texts through a spacy pipeline to segment the
sentences, and then tokenize + lemmatize + POS tag them.

As output, the script makes two files: one which is only tokenized; and one
which is lemmatized (+ optionally POS-tagged).
"""
import argparse
import os
import pandas as pd
import spacy
from utils import read_pandas
from tqdm import tqdm

def write_output(out_file, lines):
    with open(out_file, 'w') as f:
        for l in lines:
            f.write(l + '\n')
    print('Output written:', out_file)

def main():
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument('in_file', help='path to pandas DF with column `text`')
    parser.add_argument('out_dir', help='path to output directory')
    parser.add_argument('basename', help='basename for output files')
    parser.add_argument('--processes', help='number of processes, default is 1',
                        type=int, default=1)
    parser.add_argument('--no-pos', action='store_true', default=False,
                        help='do not write POS tags')
    parser.add_argument('--lowercase', action='store_true', default=False,
                        help='lowercase all tokens in output')
    parser.add_argument('--col_name', default='text', help='name of input DF '
                        'column with text to process, defaults to `text`')
    args = parser.parse_args()

    # Print config
    print(f'Processing with {args.processes} processes')
    print('Omitting POS tags:', args.no_pos)
    print('Lowercasing all tokens:', args.lowercase)
    print('Target column name:', args.col_name)

    # Load files
    df = read_pandas(args.in_file)
    texts = df[args.col_name].tolist()
    print(f'Loaded {len(texts)} sentence windows: {args.in_file}')

    # Process texts
    nlp = spacy.load('en_core_web_sm')
    docs = nlp.pipe(texts, n_process=args.processes)
    tokenized = []
    lemmatized = []
    for doc in tqdm(docs, total=len(texts)):
        for sent in doc.sents:
            tmp_tokenized = []
            tmp_lemmatized = []
            for tok in sent.as_doc():
                if args.lowercase:
                    token = tok.text.lower()
                    lemma = tok.lemma_.lower()
                else:
                    token = tok.text
                    lemma = tok.lemma_
                if not args.no_pos:
                    lemma += ':' + tok.pos_
                tmp_tokenized.append(token)
                tmp_lemmatized.append(lemma)
            tokenized.append(' '.join(tmp_tokenized))
            lemmatized.append(' '.join(tmp_lemmatized))

    # Set up file names
    tail = ''
    if args.no_pos:
        tail += '_nopos'
    if args.lowercase:
        tail += '_lower'

    out_tok = os.path.join(args.out_dir,
                           f'{args.basename}_tokenized{tail}.txt')
    out_lem = os.path.join(args.out_dir,
                           f'{args.basename}_lemmatized{tail}.txt')

    # Write output
    write_output(out_tok, tokenized)
    write_output(out_lem, lemmatized)


if __name__ == '__main__':
    main()