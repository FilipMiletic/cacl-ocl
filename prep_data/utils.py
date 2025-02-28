import pandas as pd

def read_pandas(path: str, **kwargs) -> pd.DataFrame:
    """Reads pandas dataframe distinguishing between different file formats.

    Args:
        path: Path to pandas dataframe file. Based on ending, resolved as
          parquet (`.parquet`), CSV (`.csv`), TSV (`.tsv`), or otherwise pickle.
        kwargs: Keyword arguments passed to pandas read function.

    Returns:
        A pandas dataframe.
    """
    if path.endswith('.parquet'):
        df = pd.read_parquet(path, **kwargs)
    elif path.endswith('.csv'):
        df = pd.read_csv(path, **kwargs)
    elif path.endswith('.tsv'):
        df = pd.read_csv(path, sep='\t', **kwargs)
    else:
        df = pd.read_pickle(path, **kwargs)
    return df