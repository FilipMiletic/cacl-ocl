import argparse
import os
import pandas as pd
from bs4 import BeautifulSoup


def mapping_to_dataframe(mapping: dict[str, list[str]]) -> pd.DataFrame:
    """Converts volume-to-venue dictionary into pandas dataframe.

    Args:
        mapping: Dictionary mapping ACL volumes to lists of venues.
    Returns:
        Pandas dataframe with columns `acl_volume` and `venue{1,2,3}`.
    """
    columns = ['venue1', 'venue2', 'venue3']
    df = pd.DataFrame.from_dict(mapping, orient='index', columns=columns)
    df = df.reset_index().rename(columns={'index': 'acl_volume'})
    return df


def main():
    desc = ('This script maps ACL anthology volumes to venues, e.g.: P00-1 -> '
            'acl; 2024.acl-long -> acl; 2024.lrec-main -> lrec, coling. It uses'
            ' ACL anthology XML files which should be available here: '
            'https://github.com/acl-org/acl-anthology/ -> data/xml')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('path_to_xml_files', help='path to Anthology XLMs')
    parser.add_argument('out_file', help='path to output parquet file')
    args = parser.parse_args()

    # Note to selves: we used Anthology XMLs as downloaded by acl-anthology-py:
    # ~/.local/share/acl-anthology-py/git/acl-org-acl-anthology-git/data/xml
    in_files = os.listdir(args.path_to_xml_files)
    in_files = [f for f in in_files if f.endswith('.xml')]

    mapping = {}
    for in_file in in_files:
        with open(os.path.join(args.path_to_xml_files, in_file), 'r') as f:
            soup = BeautifulSoup(f, 'xml')

        # Find collection info -- should be unique in file
        # <collection id="2024.acl">
        collection = soup.find_all('collection')
        assert len(collection) == 1
        collection = collection[0]
        coll_id = collection['id']

        # Find volume info -- possibly multiple within a collection
        # <volume id="long" ...>
        volumes = collection.find_all('volume')
        for volume in volumes:
            vol_id = volume['id']
            # Old-style workshops identifiers: add lead zero for volume ID
            # https://aclanthology.org/info/ids/
            if coll_id.startswith('W'):
                vol_id = f'{int(vol_id):02d}'
            # TODO: Fix special case of C69 and D19

            # Extract venues associated with the volume -- possibly multiple
            # <venue>acl</venue>
            venues = []
            for venue in volume.find_all('venue'):
                venues.append(venue.text)

            # Make full volume identifier (collection + volume) and map it
            # 2024.acl-long -> acl
            identifier = '-'.join([coll_id, vol_id])
            mapping[identifier] = venues

    # Write dataframe
    df = mapping_to_dataframe(mapping)
    df.to_parquet(args.out_file, compression='gzip')


if __name__ == '__main__':
    main()