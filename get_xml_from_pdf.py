import argparse
import os
from grobid.client import GrobidClient

def process_pdfs(in_dir, out_dir):
    
    # Grobid client needs to be running on Docker
    client = GrobidClient(port='8070')

    pdf_files = os.listdir(in_dir)
    pdf_files = [f for f in pdf_files if f.endswith('.pdf')]

    print(f'Processing {len(pdf_files)} PDF files.')
    print(f'Input directory:  {in_dir}')
    print(f'Output directory: {out_dir}')
    
    for i, pdf_file in enumerate(pdf_files):
        rsp = client.serve(service='processFulltextDocument',
                           pdf_file=os.path.join(in_dir, pdf_file))
        text = rsp[0].content

        xml_file = f'{pdf_file[:-4]}.tei.xml'
        xml_file = os.path.join(out_dir, xml_file)
        if not os.path.isfile(xml_file):
            with open(xml_file, 'wb') as f:
                f.write(text)
        else:
            print(f'Skipping existing file: {xml_file}')
            
        if i % 100 == 0:
            print(f'Processed {i} files.')
            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', help='input directory with paper PDFs')
    parser.add_argument('out_dir', help='output directory to write XMLs')
    args = parser.parse_args()

    process_pdfs(args.in_dir, args.out_dir)

if __name__ == '__main__':
    main()
