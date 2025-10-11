"""This script starts from a list of <lemma>:<POS> targets, for each looks up
sentences in one-sentence-per-line corpora, encodes them with a BERT-like model,
clusters them, and outputs a dataframe with the clustered sentences."""
import argparse
import os
import pandas as pd
import torch
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine, jensenshannon
from transformers import AutoTokenizer, AutoModel

MODEL = 'answerdotai/ModernBERT-base'
CACHE = '/resources/models/huggingface/hub/'


def load_inputs(lemmatized_path: str, tokenized_path: str,
                time_col: str) -> pd.DataFrame:
    """Loads one-sentence-per-line corpus with lemmatized + tokenized files.

    Args:
        lemmatized_path: One-sentence-per-line file with space-separated
            sequences of <lemma>:<POS>.
        tokenized_path: One-sentence-per-line file with space-separated tokens.
        time_col: Value to write into the `time` column.
    Returns:
        A pandas dataframe with `lemmatized`, `tokenized`, `time` columns.
    """
    with open(lemmatized_path, 'r') as f:
        lemmatized = f.readlines()
    with open(tokenized_path, 'r') as f:
        tokenized = f.readlines()

    lemmatized = [l.rstrip() for l in lemmatized]
    tokenized = [l.rstrip() for l in tokenized]
    assert len(lemmatized) == len(tokenized)

    df = pd.DataFrame({'lemmatized': lemmatized,
                       'tokenized': tokenized,
                       'time': time_col})
    return df


def get_target_forms(target: str, sents_lem: list, sents_tok: list) -> list:
    """Iterates over lemmatized + tokenized sentences to build the corresponding
    list of tokenized forms with which the target lemma is used.

    Args:
        target: Target lemma in <lemma>:<POS> format.
        sents_lem: Iterable of sentences containing the target, in <lemma>:<POS>
            format.
        sents_tok: Iterable of corresponding sentences in tokenized format.
    Returns:
        A list of tokens corresponding to the lemma-target in `sents_tok`.
    """
    target_forms = []
    for lemmas, tokens in zip(sents_lem, sents_tok):
        # Map lemmas onto tokens for a given sentence
        lemmas = lemmas.split()
        lemmas = [l for l in lemmas if not l.endswith(':SPACE')] # clean up
        tokens = tokens.split()
        assert len(lemmas) == len(tokens)
        pairs = set(zip(lemmas, tokens))

        # Iterate over lemma-token pairs in a given sentence
        for lemma, token in pairs:
            if lemma == target:
                target_forms.append(token)
                break # only one target occurrence per sentence

    return target_forms

def get_target_embeddings(target_forms, sents_tok, inputs, model_outs,
                          tokenizer) -> (list[torch.Tensor], list[int]):
    """Iterates over tokenized sentences + target forms used in them + full
    sentence-level embeddings, and extracts embeddings corresponding to the
    target for each sentence.

    Args:
        target_forms: List of target tokens as attested in individual sentences.
        sents_tok: List of tokenized sentences containing the target.
        inputs: Those sentences tokenized by the model's tokenizer.
        model_outs: Model outputs with embeddings for the whole sentence.
        tokenizer: The tokenizer used to get inputs.
    Returns:
        A list of target-level embeddings, and a list of indices of sentences
        corresponding to those embeddings (in case some need to be dropped).
    """
    embs = []
    emb_sent_ids = []
    everything = zip(target_forms, sents_tok, model_outs.last_hidden_state)

    for i, (target_form, sent, model_out) in enumerate(everything):
        # Get the index at which the target occurs in original sentence
        idx_in_input = sent.split(' ').index(target_form)

        # Try the fast solution
        try:
            # Get indices of model-tokenized tokens corresponding to the
            # target's index in the space-separated input sentence
            start, end = inputs.word_to_tokens(batch_or_word_index=i,
                                               word_index=idx_in_input)
        except TypeError:
            print('Type error:', target_form, i)
            continue
        # Get the model-tokenized token IDs at the indices found above
        target_tokens = inputs['input_ids'][i][start:end]

        # Check if fast solution worked, i.e. if token IDs map back to target;
        # if not, go the longer way by iterating over all model-made tokens
        if tokenizer.decode(target_tokens).strip() != target_form:
            # Iterate over the model-defined mapping of input words to tokens,
            # decode each sequence of tokens corresponding to a word,
            # and check if that corresponds to our target
            max_id = max([wid for wid in inputs.word_ids(i) if wid is not None])
            for word_id in range(max_id):
                start, end = inputs.word_to_tokens(i, word_id)
                target_tokens = inputs['input_ids'][i][start:end]
                if tokenizer.decode(target_tokens).strip() == target_form:
                    break
            else:
                # Didn't get to break -> target not found, skipping the sentence
                print('Target not found:', target_form, i)
                continue
        # Average possible subword embeddings and append the outputs
        emb = model_out[start:end].mean(axis=0)
        embs.append(emb)
        emb_sent_ids.append(i)

    return embs, emb_sent_ids

def get_centroid_dists(embs, kmeans) -> list:
    """Calculates distance from embedding to cluster centroid after k-means."""
    centroid_dists = []
    for emb, clust in zip(embs, kmeans.labels_):
        centroid = kmeans.cluster_centers_[clust]
        dist = cosine(emb, centroid)
        centroid_dists.append(dist)
    return centroid_dists

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lemmatized_path1', help='path to lemmatized corpus 1, '
                        'in one-sentence-per-line format')
    parser.add_argument('tokenized_path1', help='path to tokenized corpus 1')
    parser.add_argument('lemmatized_path2', help='path to lemmatized corpus 2')
    parser.add_argument('tokenized_path2', help='path to tokenized corpus 2')
    parser.add_argument('targets_path', help='file with <lemma>:<POS> targets, '
                        'one target per line')
    parser.add_argument('out_dir', help='path to output directory where '
                        'output files will be dumped for each target -- NOTE: '
                        'output file names correspond to targets')
    parser.add_argument('--sample', type=int, default=1000, help='maximum '
                        'sentences per target to sample from each corpus, '
                        'defaults to 1000')
    parser.add_argument('-k', type=int, default=8, help='number of clusters '
                        'for k-means, defaults to 8')
    args = parser.parse_args()

    # Load datasets
    df1 = load_inputs(args.lemmatized_path1, args.tokenized_path1, 't1')
    df2 = load_inputs(args.lemmatized_path2, args.tokenized_path2, 't2')
    df = pd.concat([df1, df2], ignore_index=True)
    print('\n'.join(['Loaded:', args.lemmatized_path1, args.lemmatized_path2]))

    # Map vocabulary to sentences
    print('Preparing vocabulary mapper... ', end='\r')
    mapper = defaultdict(list)
    for i, row in enumerate(df['lemmatized']):
        for lemma in row.split(' '):
            mapper[lemma].append(i)
    print('Preparing vocabulary mapper... Done.')

    # Load tokenizer and model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModel.from_pretrained(MODEL, cache_dir=CACHE).to(device)
    print('\n'.join(['Tokenizer and model loaded on '+device, MODEL]))

    # Load targets to be modeled
    with open(args.targets_path, 'r') as f:
        targets = f.readlines()
    targets = [t.rstrip() for t in targets]
    print(f'Loaded {len(targets)} targets.')

    for target in targets:
        print('Processing target:', target)

        # Extract and subsample sentences in which target is used
        sub_df = df.iloc[mapper[target]][['lemmatized', 'tokenized', 'time']]
        max1, max2 = sub_df['time'].value_counts()[['t1', 't2']]
        sample1 = sub_df[sub_df['time'] == 't1'].sample(min(args.sample, max1),
                                                        random_state=42)
        sample2 = sub_df[sub_df['time'] == 't2'].sample(min(args.sample, max2),
                                                        random_state=42)
        sub_df = pd.concat([sample1, sample2]).sort_index()
        sents_lem, sents_tok, times = sub_df.values.T

        # Map the actual (tokenized) forms with which the target is attested
        target_forms = get_target_forms(target, sents_lem, sents_tok)
        assert len(sents_lem) == len(sents_tok) == len(target_forms)

        # Tokenize
        inputs = tokenizer(list(sents_tok),
                           padding=True,
                           truncation=True,
                           max_length=256,
                           return_tensors='pt').to(device)

        # Embed sentences
        with torch.no_grad():
            model_outs = model(**inputs, output_hidden_states=False)

        # Get target embeddings
        embs, emb_sent_ids = get_target_embeddings(target_forms,
                                                   sents_tok,
                                                   inputs,
                                                   model_outs,
                                                   tokenizer)
        print(f'Embedded  {len(sents_tok)} sentences')
        print(f'Extracted {len(embs)} target embeddings')
        assert len(embs) == len(emb_sent_ids)

        # Run clustering
        embs = [e.cpu() for e in embs] # needs to happen for numpy
        kmeans = KMeans(n_clusters=args.k, random_state=42).fit(embs)
        centroid_dists = get_centroid_dists(embs, kmeans)

        # Process clustering
        out = pd.DataFrame({'sent_id': [mapper[target][i] for i in emb_sent_ids],
                            'sent': [sents_tok[i] for i in emb_sent_ids],
                            'clust': kmeans.labels_,
                            'dist': centroid_dists,
                            'time': [times[i] for i in emb_sent_ids]})
        out = out.sort_values(by=['clust', 'dist'])

        # Write output
        out_path = os.path.join(args.out_dir, target+'.parquet')
        out.reset_index(drop=True).to_parquet(out_path)
        print('Output written:', out_path)

if __name__ == '__main__':
    main()