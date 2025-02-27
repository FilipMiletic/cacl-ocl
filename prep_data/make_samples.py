import argparse
import os
import pandas as pd


def get_acl_volume(acl_id: str) -> str:
    """Converts ACL paper ID into complete volume ID.
    See https://aclanthology.org/info/ids/ """
    if acl_id[0].isdigit():
        # new style:
        vol = acl_id.split('.')
        vol = '.'.join(vol[:2])
    elif acl_id[0] == 'W':
        # old style workshop
        vol = acl_id[:6]
    else:
        # old style conference
        vol = acl_id[:5]

    return vol


def read_pandas(path: str) -> pd.DataFrame:
    """Reads pandas DF, assuming parquet if ends with '.parquet' and
    pickle otherwise."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    return df


def main():
    desc = ('Make temporal samples of ACL papers, defined as follows: '
            't1: papers from 2020-2022; t2: papers from 07/2023-12/2024')
    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=desc, formatter_class=fmt)
    parser.add_argument('corpus_files', nargs='+', help='Pickle or parquet '
                        'dataframes with XML-derived paper info')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--base-name', help='Base name for output files',
                        default='cacl')
    parser.add_argument('--venues', help='DF mapping volume IDs to venues',
                        default='aclvolume2venue.parquet')
    args = parser.parse_args()

    # Load corpus files
    df = pd.DataFrame()
    cols = None
    for corpus_file in args.corpus_files:
        tmp = read_pandas(corpus_file)
        if cols is None:
            cols = tmp.columns
        else:
            cols = [c for c in cols if c in tmp]
        df = pd.concat([df, tmp], ignore_index=True)[cols]

    # Map volumes to venues
    mapping = read_pandas(args.venues)
    df['acl_volume'] = df['acl_id'].apply(get_acl_volume)
    df = df.merge(mapping, how='left')
    df = df.drop_duplicates(subset='acl_id')

    # Define time splits
    months = ['July', 'August', 'September', 'October', 'November',
                   'December', 'septempber', '20 August']
    df['time'] = 't0'
    df.loc[df['year'].isin(['2020', '2021', '2022']), 'time'] = 't1'
    df.loc[(df['year'] == '2023') & (df['month'].isin(months)), 'time'] = 't2'
    df.loc[(df['year'] == '2024'), 'time'] = 't2'

    print('All available papers')
    print(df['time'].value_counts(), '\n')

    # Get shared venues
    venues = {}
    for time in ['t1', 't2']:
        subvenues = df.loc[df['time'] == time][ ['venue1', 'venue2', 'venue3']]\
                      .melt().dropna()['value'].unique()
        venues[time] = set(subvenues)
    shared_venues = venues['t1'] & venues['t2']
    assert 'ws' not in shared_venues

    # Filter for time + shared venues
    df = df.loc[df['time'] != 't0']
    df = df.loc[(df['venue1'].isin(shared_venues)) |
                (df['venue2'].isin(shared_venues)) |
                (df['venue3'].isin(shared_venues))]
    print('Papers after filtering')
    print(df['time'].value_counts(), '\n')

    # Exclude frontmatter papers
    df = df.loc[~df['acl_id'].str.endswith('.0')]
    print('Papers after frontmatter exclusion')
    print(df['time'].value_counts())

    # Write time-specific dataframes
    for time in ['t1', 't2']:
        out_file = f'{args.base_name}_{time}.parquet'
        out_path = os.path.join(args.out_dir, out_file)
        df.loc[df['time'] == time].reset_index(drop=True).to_parquet(out_path)


if __name__ == '__main__':
    main()