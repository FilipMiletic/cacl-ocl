import pandas as pd
import spacy
from huggingface_hub.utils import tqdm
from tabulate import tabulate


def return_sentences(text):
    # use spacy to split the text into sentences
    # download en_core_web_sm model using: python -m spacy download en_core_web_sm

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def remove_non_english_sentences(sentences):
    # use package langid to detect the language of the sentence
    from langid.langid import LanguageIdentifier, model
    lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    english_sentences = []
    for sentence in sentences:
        lang, prob = lang_identifier.classify(sentence)
        if lang == "en":
            english_sentences.append(sentence)
    return english_sentences


def create_sentence_windows(sentences, window_size):
    # create windows of sentences
    windows = []
    for i in range(0, len(sentences), window_size):
        window = sentences[i:i + window_size]
        windows.append(window)
    return windows

if __name__ == '__main__':
    old_papers = "acl-publication-info.74k.parquet"
    # find all rows where full text is empty or null or None

    # read in as a pandas dataframe
    df = pd.read_parquet(old_papers)
    empty_full_text = df[df.full_text.isnull() | df.full_text.eq("") | df.full_text.eq("None")]
    print("Number of rows with empty full text:", len(empty_full_text))
    print(tabulate(empty_full_text.head(), headers='keys', tablefmt='psql'))
    print(df.booktitle.value_counts())
    # drop all rows with empty full text
    df = df.dropna(subset=["full_text"])
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.head(500)
    new_dfs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        paper = row["full_text"]
        # Process the paper into sentences and windows
        sentences = return_sentences(paper)
        english_sentences = remove_non_english_sentences(sentences)
        windows = create_sentence_windows(english_sentences, 5)

        # Create new rows for each window
        for window in windows:
            # merge window sentences into a single string
            window = " ".join(window)
            new_row = row.to_dict()  # Convert the row to a dictionary
            new_row["text"] = window
            new_dfs.append(new_row)  # Append the new row to the list

    # Combine all rows into a new DataFrame
    full_df = pd.DataFrame(new_dfs)
    full_df["ID"] = range(len(full_df))
    full_df = pd.DataFrame(new_dfs)

    # Print the result
    print(tabulate(full_df.head(), headers='keys', tablefmt='psql'))
    # save the new dataframe
    full_df.to_parquet("acl-publication-info.74k.windows.parquet")

