"""This script trains a topic model over the Anthology corpus."""
import argparse
import logging
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from utils import read_pandas
from umap import UMAP

def prepare_docs(df_path: str) -> (list[str], pd.DataFrame):
    """Prepares documents for topic model training from corpus dataframe.

    Args:
        df_path: Path to pandas dataframe file with at least `title_abstract`
          column. Remaining columns can contain metadata and are concatenated
          to the output papers dataframe.

    Returns:
        A list (containing documents to pass to the topic model), and a pandas
        dataframe (with documents, after extra preprocessing to remove NaNs).
    """
    df = read_pandas(df_path)
    df = df.dropna(subset='title_abstract')
    # df = df.drop_duplicates(subset='title_abstract')
    docs = df['title_abstract'].to_list()
    assert all(isinstance(doc, str) for doc in docs)

    return docs, df

def train_model(docs: list[str], min_topic_size: int) -> BERTopic:
    """Trains a topic model.

    Params:
        docs: List of documents to pass to the topic model.
        min_topic_size: Minimum number of documents per topic.

    Returns:
        A BERTopic object.
    """
    # Initialize models
    # UMAP uses default BERTopic param value + fixes the random state
    umap_model = UMAP(n_neighbors=15, n_components=5,
                      min_dist=0.0, metric='cosine', random_state=42)
    representation_model = MaximalMarginalRelevance()

    # Train topic model
    topic_model = BERTopic(umap_model=umap_model,
                           representation_model=representation_model,
                           min_topic_size=min_topic_size,
                           calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(docs)

    # Reduce outliers (topic -1)
    new_topics = topic_model.reduce_outliers(docs,
                                             topics,
                                             probabilities=probs,
                                             strategy='probabilities')
    topic_model.update_topics(docs, topics=new_topics)

    return topic_model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='path to pandas DF with columns '
                        '`acl_id`, `time`, and `title_abstract`')
    parser.add_argument('output_model_file', help='path to dump topic model')
    parser.add_argument('output_papers_file', help='path to dump papers DF')
    parser.add_argument('--min_topic_size', type=int, default=100, help='min '
                        'number of documents per topic, defaults to 100')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                        level=logging.INFO)

    # Load corpus
    docs, df = prepare_docs(args.input_file)
    logging.info(f'Loaded corpus from: {args.input_file}')

    # Train model
    topic_model = train_model(docs, args.min_topic_size)
    logging.info('Trained topic model with min_topic_size '
                 f'{args.min_topic_size}')

    # Prepare document analysis
    papers_df = topic_model.get_document_info(docs, df=df)

    # Dump output files
    topic_model.save(args.output_model_file, serialization='pickle')
    logging.info(f'Saved topic model to: {args.output_model_file}')
    papers_df.to_parquet(args.output_papers_file)
    logging.info(f'Saved papers dataframe to: {args.output_papers_file}')

if __name__ == '__main__':
    main()
