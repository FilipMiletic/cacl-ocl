import argparse
import pandas as pd


def load_corpus_dfs(corpus_files: list[str]) -> pd.DataFrame:
    """Loads possibly multiple corpus DFs and merges them into one."""
    master_df = pd.DataFrame()
    for corpus_file in corpus_files:
        if corpus_file.endswith('.parquet'):
            tmp_df = pd.read_parquet(corpus_file)
        else:
            tmp_df = pd.read_pickle(corpus_file)
        tmp_df = tmp_df[['acl_id', 'full_text']]
        master_df = pd.concat([master_df, tmp_df], ignore_index=True)

    master_df = master_df.drop_duplicates(subset='acl_id')
    return master_df


def main():
    desc = 'Compares available ACL corpus files against the full Anthology '\
           'and finds papers that are completely missing or have empty text.'
    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=desc, formatter_class=fmt)
    parser.add_argument('corpus_files', nargs='+', help='pickle or parquet '\
                        'dataframes with columns `acl_id` and `full_text`')
    parser.add_argument('--anthology-file', help='ACL anthology dataframe',
                        default='anthology.parquet')
    parser.add_argument('--missing-file', help='output file for missing papers',
                        default='missing_papers.txt')
    parser.add_argument('--empty-file', help='output file for empty papers',
                        default='empty_papers.txt')
    args = parser.parse_args()

    anthology_df = pd.read_parquet(args.anthology_file)
    corpus_df = load_corpus_dfs(args.corpus_files)
    print('Loaded dataframes')

    all_ids = anthology_df['acl_id'].to_list()
    avail_ids = corpus_df['acl_id'].to_list()
    missing_ids = sorted(list(set(all_ids) - set(avail_ids)))
    empty_ids = corpus_df.loc[corpus_df['full_text'].isna()]['acl_id'].to_list()

    with open(args.missing_file, 'w') as f:
        f.writelines([paper+'\n' for paper in missing_ids])

    with open(args.empty_file, 'w') as f:
        f.writelines([paper+'\n' for paper in empty_ids])


if __name__ == '__main__':
    main()