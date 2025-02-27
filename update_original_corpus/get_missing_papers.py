import argparse
import logging
import os
import random
import urllib.request
from time import sleep
from urllib.error import HTTPError


def load_existing_pdfs(pdf_dirs: list[str] | None) -> set[str]:
    """Loads list of paper IDs already available as PDF files."""
    pdfs = set()
    if pdf_dirs is not None:
        for pdf_dir in pdf_dirs:
            pdfs.update(os.listdir(pdf_dir))
        pdfs = {f[:-4] for f in pdfs if f.endswith('.pdf')}

    return pdfs


def load_missing_papers(in_files: list[str]) -> set[str]:
    """Loads list of missing paper IDs from one-paper-per-line files."""
    papers = set()
    for in_file in in_files:
        with open(in_file, 'r') as f:
            papers.update(f.readlines())
    papers = {p.rstrip() for p in papers}
    return papers


def download_paper(acl_id: str, out_dir: str) -> None:
    """Downloads paper PDF from anthology into `out_dir`."""
    url = f'https://aclanthology.org/{acl_id}.pdf'
    out_file = os.path.join(out_dir, f'{acl_id}.pdf')
    try:
        urllib.request.urlretrieve(url, out_file)
    except HTTPError as e:
        logging.warning(f'{acl_id}\t{e}')
    else:
        logging.info(f'{acl_id}\tDone')


def main():
    desc = 'Downloads paper PDFs from ACL anthology based on a list of '\
           'ACL paper IDs, skipping papers whose PDFs already exist locally.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('in_files', type=str, nargs='+',
                        help='files containing one-per-line ACL paper IDs')
    parser.add_argument('--pdf-dirs', type=str, nargs='+',
                        help='directories containing already downloaded PDFs')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='directory where to download new PDFs')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                        level=logging.INFO)

    missing_papers = load_missing_papers(args.in_files)
    existing_pdfs = load_existing_pdfs(args.pdf_dirs)
    to_download = missing_papers - existing_pdfs
    logging.info(f'Loading paper IDs from files: {args.in_files}')
    logging.info(f'Checking against PDFs in directories: {args.pdf_dirs}')
    logging.info(f'Downloading {len(to_download)} PDFs to: {args.out_dir}')

    for acl_id in to_download:
        download_paper(acl_id, args.out_dir)
        sleep(random.randint(0, 3))
    logging.info('Done.')

if __name__ == '__main__':
    main()