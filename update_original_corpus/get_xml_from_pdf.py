import argparse
import logging
import os
from grobid.client import GrobidClient


def process_pdf(pdf_file: str, out_dir: str, client: GrobidClient) -> None:
    """Converts PDF into XML file."""
    xml_file = f'{os.path.basename(pdf_file)[:-4]}.tei.xml'
    xml_file = os.path.join(out_dir, xml_file)

    if os.path.isfile(xml_file):
        logging.warning(f'Skipping existing file: {xml_file}')
    else:
        rsp = client.serve(service='processFulltextDocument', pdf_file=pdf_file)
        text = rsp[0].content
        with open(xml_file, 'wb') as f:
            f.write(text)


def main():
    desc = 'Converts paper PDFs into XML files using GROBID.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('in_dir', help='input directory with paper PDFs')
    parser.add_argument('out_dir', help='output directory to write XMLs')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                        level=logging.INFO)

    pdf_files = os.listdir(args.in_dir)
    pdf_files = [f for f in pdf_files if f.endswith('.pdf')]

    logging.info(f'Processing {len(pdf_files)} PDF files.')
    logging.info(f'Input directory:  {args.in_dir}')
    logging.info(f'Output directory: {args.out_dir}')

    # Grobid server needs to be running on Docker
    client = GrobidClient(port='8070')

    for i, pdf_file in enumerate(pdf_files):
        pdf_file = os.path.join(args.in_dir, pdf_file)
        process_pdf(pdf_file, args.out_dir, client)
        if i % 100 == 0:
            logging.info(f'Processed {i} files.')

    logging.info(f'PDF to XML conversion done.')

if __name__ == '__main__':
    main()
