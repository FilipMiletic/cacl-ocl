"""This script takes w2v models trained on two different corpora and in multiple
runs, computes self-similarities across pairs trained in a given run, and
averages over those runs."""
import argparse
import os
import pandas as pd
from collections import defaultdict
from gensim.models import Word2Vec
from pathlib import Path
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm
from w2v_orthogonal_procrustes import procrustes_align

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_files1', nargs='+', required=True,
                        help='gensim word2vec models trained on the 1st corpus')
    parser.add_argument('--model_files2', nargs='+', required=True,
                        help='gensim word2vec models trained on the 2nd corpus')
    parser.add_argument('--out_dir', required=True, help='output directory to '
                        'write out similarities as pandas parquet file')
    parser.add_argument('--out_file', help='output file name; if not provided, '
                        'defaults to {model_files1[0]}_sim_align_avg.parquet')
    args = parser.parse_args()

    # Check there's the same number of models from both corpora
    assert len(args.model_files1) == len(args.model_files2)

    sims = defaultdict(list)
    for i in range(len(args.model_files1)):
        model1 = Word2Vec.load(args.model_files1[i])
        model2 = Word2Vec.load(args.model_files2[i])
        model2 = procrustes_align(model1, model2)
        print('Models loaded and aligned:',
              args.model_files1[i], args.model_files2[i])
        for w in tqdm(model1.wv.key_to_index):
            sim = 1-cosine(model1.wv[w], model2.wv[w])
            sims[w].append(sim)

    # Put everything into pandas and write out
    df = pd.DataFrame.from_dict(sims, orient='index')
    df = df.rename(columns={c: str(c) for c in df.columns})
    df['avg'] = df.mean(axis=1)
    if not args.out_file:
        args.out_file = f'{Path(args.model_files1[0]).stem}_sim_align_avg.parquet'
    out_path = os.path.join(args.out_dir, args.out_file)
    df.to_parquet(out_path)
    print('Output written:', out_path)

if __name__ == '__main__':
    main()