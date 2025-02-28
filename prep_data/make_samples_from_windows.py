import argparse
import logging
import os
import pandas as pd
from utils import read_pandas


def clean_output_df(df, cols):
    cols = [c for c in cols if c != 'ID'] + ['index']
    df = df.reset_index()[cols]
    df = df.rename(columns={'index': 'ID'})
    return df


def main():
    desc = ('Starting from corpus files mapping papers to time splits (`acl_id`'
            ' and `time` columns), this script repackages derived files (e.g. '
            'sentence windows files, containing minimally `acl_id`) into time-'
            'specific files.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('t1_file', help='input dataframe for T1')
    parser.add_argument('t2_file', help='input dataframe for T2')
    parser.add_argument('--raw-files', nargs='+', required=True,
                        help='dataframes to be repackaged')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument('--base-name', required=True,
                        help='base name for output file')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Load t1/t2 papers
    t1 = read_pandas(args.t1_file)
    t2 = read_pandas(args.t2_file)
    t1_papers = set(t1['acl_id'])
    t2_papers = set(t2['acl_id'])
    logging.info('Loaded t1 papers from: %s', args.t1_file)
    logging.info('Loaded t2 papers from: %s', args.t2_file)

    # Split raw (e.g. window) files
    t1_raw = pd.DataFrame()
    t2_raw = pd.DataFrame()
    cols = None

    for in_file in args.raw_files:
        logging.info('Loading file: %s', in_file)
        raw = read_pandas(in_file)
        if cols is None:
            cols = raw.columns
        else:
            cols = [c for c in cols if c in raw]

        t1_tmp = raw.loc[raw['acl_id'].isin(t1_papers)]
        t2_tmp = raw.loc[raw['acl_id'].isin(t2_papers)]
        t1_raw = pd.concat([t1_raw, t1_tmp], ignore_index=True)
        t2_raw = pd.concat([t2_raw, t2_tmp], ignore_index=True)
        logging.info('Repackaged data from: %s', in_file)

    # Clean up files
    t1_raw = clean_output_df(t1_raw, cols)
    t2_raw = clean_output_df(t2_raw, cols)

    # Write files
    t1_out_path = os.path.join(args.out_dir, args.base_name+'_t1.parquet')
    t2_out_path = os.path.join(args.out_dir, args.base_name+'_t2.parquet')
    t1_raw.to_parquet(t1_out_path, compression='gzip')
    logging.info('Output written: %s', t1_out_path)
    t2_raw.to_parquet(t2_out_path, compression='gzip')
    logging.info('Output written: %s', t2_out_path)


if __name__ == '__main__':
    main()