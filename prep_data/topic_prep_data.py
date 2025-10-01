"""This script prepares data for topic modeling by merging existing DFs
across time periods and only keeping title+abstract info, to use for topics."""
import argparse
import pandas as pd
from pathlib import Path
from utils import read_pandas

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_files', nargs='+', help='corpus data in '
                        'pandas dataframes containing columns `acl_id`, '
                        '`title`, `abstract`. Assumes last underscore-leading '
                        'string in the filename to be time period')
    parser.add_argument('output_file', help='output parquet DF with columns '
                        '`acl_id`, `time`, `title_abstract`')
    args = parser.parse_args()

    dfs = []
    for input_file in args.input_files:
        # Figure out time information
        basename = Path(input_file).stem
        time = basename.split('_')[-1]
        assert time in ['t1', 't2'] # sanity check for our use case

        # Load and process DF
        df = read_pandas(input_file)
        df['time'] = time
        df['title_abstract'] = df['title'] + '. ' + df['abstract']
        cols = ['acl_id', 'time', 'title_abstract']
        dfs.append(df[cols])

    # Concatenate DFs
    df = pd.concat(dfs, ignore_index=True)

    # Write out final DF
    df.to_parquet(args.output_file)

if __name__ == '__main__':
    main()
