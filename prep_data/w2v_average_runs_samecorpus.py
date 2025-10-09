"""This script takes multiple w2v models trained on the same corpus and
builds a similarity matrix for the whole vocabulary by averaging over
the similarities from all models."""
import argparse
import numpy as np
import os
import pandas as pd
from itertools import combinations
from gensim.models import Word2Vec
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_files', nargs='+', help='gensim word2vec models '
                        'whose similarities will be averaged together; they '
                        'must have identical vocabularies')
    parser.add_argument('out_dir', help='output directory to write out average '
                        'similarity matrix as pandas parquet file')
    parser.add_argument('--out_file', help='output file name; if not provided, '
                        'defaults to {model_files[0]}_sim_avg.parquet')
    parser.add_argument('--smaller', action='store_true', default=False,
                        help='saves output in np.float16 format')
    args = parser.parse_args()

    # Load individual models
    models = []
    vocabs = []
    for model_file in args.model_files:
        model = Word2Vec.load(model_file)
        models.append(model.wv.get_normed_vectors())
        vocabs.append(model.wv.index_to_key)
        print('Model loaded:', model_file)

    # Check that vocabularies are identical
    for vocab1, vocab2 in combinations(vocabs, 2):
        if vocab1 != vocab2:
            raise ValueError('Vocabularies not identical')
    print('All vocabularies identical.')

    # Compute similarity matrix for each run
    matrices = []
    for model in models:
        matrix = np.dot(model, model.T)
        matrices.append(matrix)

    # Get average similarity matrix
    mean_matrix = np.mean(matrices, axis=0)
    print('Mean similarity matrix computed.')

    # Put everything into pandas and write out
    df = pd.DataFrame(mean_matrix, index=vocabs[0], columns=vocabs[0])
    if args.smaller:
        df = df.astype(np.float16)
        print('Saving in np.float16')
    if not args.out_file:
        args.out_file = f'{Path(args.model_files[0]).stem}_sim_avg.parquet'
    out_path = os.path.join(args.out_dir, args.out_file)
    df.to_parquet(out_path)
    print('Output written:', out_path)

if __name__ == '__main__':
    main()