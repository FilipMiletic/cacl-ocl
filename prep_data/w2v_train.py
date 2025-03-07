"""Trains a word2vec model from one-sentence-per-line corpus."""
import argparse
import logging
import os
from gensim.models import word2vec
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file', help='one-sentence-per-line corpus file')
    parser.add_argument('out_dir', help='output directory for w2v model')
    parser.add_argument('--dims', help='vector dimensions, default 100',
                        default=100, type=int)
    parser.add_argument('--win', help='window around target, default 5',
                        default=5, type=int)
    parser.add_argument('--freq', help='minimum frequency, default 100',
                        default=100, type=int)
    parser.add_argument('--algo', help="one of ['cbow', 'sg'], default 'sg'",
                        default='sg', choices=['cbow', 'sg'])
    parser.add_argument('--work', help='number of workers, default 18',
                        default=18, type=int)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Load sentences
    sentences = word2vec.LineSentence(args.in_file)
    logging.info(f'Loaded training data: {args.in_file}')

    # Train model
    model = word2vec.Word2Vec(sentences,
                              vector_size=args.dims,
                              window=args.win,
                              min_count=args.freq,
                              sg=(1 if args.algo == 'sg' else 0),
                              workers=args.work)
    logging.info('Model trained.')

    # Save model
    out_name = Path(args.in_file).stem
    out_name += f'__d{args.dims}_w{args.win}_f{args.freq}_{args.algo}.model'
    out_path = os.path.join(args.out_dir, out_name)
    model.save(out_path)
    logging.info(f'Model saved: {out_path}')


if __name__ == '__main__':
    main()