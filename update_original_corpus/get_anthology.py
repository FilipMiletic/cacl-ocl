import argparse
import bibtexparser
import pandas as pd
import re
import urllib.request
from smart_open import open


def get_id_from_url(url: str) -> str:
    """Extracts ACL paper ID contained in Anthology URL"""
    acl_id = re.findall('(?<=https://aclanthology.org/).+?(?=/)', url)[0]
    assert len(acl_id) > 0
    return acl_id


def anthology_to_df(anthology_file: str) -> pd.DataFrame:
    """Converts Anthology bibtex file into dataframe."""
    with open(anthology_file, 'r') as f:
        bibtex_str = f.readlines()
    bibtex_str = ''.join(bibtex_str)

    bibtex_lib = bibtexparser.bparser.parse(bibtex_str)
    df = pd.DataFrame.from_dict(bibtex_lib.entries_dict, orient='index')

    df['acl_id'] = df['url'].apply(get_id_from_url)
    assert df['acl_id'].nunique() == df.shape[0]
    df.set_index('acl_id')

    return df


def main():
    desc = 'Downloads ACL Anthology BibTeX and converts it into Pandas DF.'
    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=desc, formatter_class=fmt)
    parser.add_argument('--anthology-url', help='URL to Anthology Bib file',
                        default='https://aclanthology.org/anthology.bib.gz')
    parser.add_argument('--anthology-bib', help='local file for Anthology Bib',
                        default='anthology.bib.gz')
    parser.add_argument('--anthology-out', help='local file for Anthology DF',
                        default='anthology.parquet')
    args = parser.parse_args()

    # Download ACL Anthology file
    urllib.request.urlretrieve(args.anthology_url, args.anthology_bib)

    # Convert anthology bib to dataframe
    df = anthology_to_df(args.anthology_bib)
    df.to_parquet(args.anthology_out, compression='gzip')


if __name__ == '__main__':
    main()