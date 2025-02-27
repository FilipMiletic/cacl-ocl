existing_corpus=path_to_parquet_file.parquet
pdf_dir=path_to_download_pdfs/
xml_dir=path_to_extract_xmls/
dfs_dir=path_to_dump_dataframes/
log_file=corpus_log.txt

# Download ACL anthology and convert it into dataframe
# Outputs by default: anthology.bib.gz (raw download); anthology.parquet (parsed into Pandas dataframe)
python get_anthology.py

# Compare existing corpus version(s) against anthology to find missing papers
# Note: multiple corpus files can be passed
# Outputs by default: missing_papers.txt (entirely absent); empty_papers.txt (present but with null full_text)
python find_missing_papers.py $existing_corpus

# Download PDFs of missing papers, optionally double-checking against already available PDF files
# Note: you can pass multiple input files (with paper IDs) and multiple directories with existing PDF files
python get_missing_papers.py missing_papers.txt --pdf-dirs $pdf_dir --out-dir $pdf_dir 2> $log_file

# Uncomment to also download empty papers (limited likelihood that they have become available)
# python get_missing_papers.py empty_papers.txt --pdf-dirs $pdf_dir --out-dir $pdf_dir 2> $log_file

# Convert PDFs into XML files
# Start Docker container for GROBID server in detached mode
docker run -d --name grobid --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.1
# TODO: Test if there needs to be forced wait/sleep time for GROBID server to come on
# Run PDF to XML conversion
python get_xml_from_pdf.py $pdf_dir $xml_dir 2> $log_file
# Stop GROBID container
docker stop grobid

# Convert XML into Pandas dataframe
python parse_grobid_extraction.py $xml_dir $dfs_dir/papers_content.parquet

# Add metadata to new corpus
# TODO

# Merge corpus dataframes and update Semantic Scholar info
# TODO