import numpy as np
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
import pandas as pd
from tqdm import tqdm

stemmer = PorterStemmer()


def get_training_data(path_to_csv_file):
    """Builds feature vectors from a csv file."""
    csv_data = pd.read_csv(path_to_csv_file)
    return csv_data


def extract_vocab(texts, nb_documents=250):
    """Extract unique vocabulary from a sequence of texts."""
    token_set = set({})
    for text in tqdm(texts[:nb_documents]):
        try:
            tokens = word_tokenize(text)
            for word in tokens:
                if word.lower() not in stopwords.words("english"):
                    token_set.add(stemmer.stem(word.lower()))
        except:
            # If there is a problem, skip the text entirely... this try/except block does that...
            pass
    return token_set


if __name__ == "__main__":

    path_to_training_data = "./data/train.csv"
    path_to_testing_data = "./data/test.csv"

    print("> Loading the training data...")
    training_data = get_training_data(path_to_training_data)

    print(f"> {training_data.columns}")
    training_text = training_data["text"]

    print(f"> Total of {len(training_text)} documents available.")
    vocab = extract_vocab(training_text)

    print(f"> The total number of distinct words is {len(vocab)}.")
