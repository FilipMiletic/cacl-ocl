"""This script converts existing one-sentence-per-line file to lowercase.
It checks for existence of POS tags and doesn't touch them."""
import argparse
import os
from tqdm import tqdm
from smart_open import open
from pathlib import Path


def safe_token_lowercase(token: str) -> str:
    """Lowercases a (possibly tagged) token, preserving POS capitalization."""
    parts = token.split(':')
    try:
        # standard tagged case ("model:NOUN")
        base, pos = parts
        out = ':'.join([base.lower(), pos])
    except ValueError:
        if len(parts) == 1:
            # no tagging ("model")
            out = parts[0].lower()
        else:
            # multiple colons ("some:noise:NOUN", "::PUNCT")
            out = []
            for part in parts[:-1]:
                out.append(part.lower())
            out.append(parts[-1])
            out = ':'.join(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file', help='one-sentence-per-line file')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--out-file', help='output file name; if not provided, '
                        'corresponds to {in_file}_lower.txt')
    parser.add_argument('--ignore-pos', help='lowercase everything without '
                        'checking for POS tags to preserve capitalization',
                        action='store_true', default=False)
    args = parser.parse_args()

    # Load file
    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    # Process file
    out_lines = []
    if args.ignore_pos:
        for line in tqdm(lines):
            out_lines.append(line.lower())
    else:
        for line in tqdm(lines):
            out_line = []
            for token in line.rstrip().split():
                out_line.append(safe_token_lowercase(token))
            out_lines.append(' '.join(out_line) + '\n')

    # Write output
    if args.out_file is None:
        args.out_file = Path(args.in_file).stem + '_lower.txt'
    out_path = os.path.join(args.out_dir, args.out_file)
    with open(out_path, 'w') as f:
        f.writelines(out_lines)


if __name__ == '__main__':
    main()