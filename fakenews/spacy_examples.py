from build_feature_vectors import get_training_data

import spacy

path_to_training_data = "./data/train.csv"
path_to_testing_data = "./data/test.csv"

if __name__ == "__main__":

    # Build a natural language processing tool.
    nlp = spacy.load("en_core_web_sm")

    # Load text data.
    print("> Loading the training data...")
    training_data = get_training_data(path_to_training_data)

    print(f"> {training_data.columns}")
    training_text = training_data["text"].tolist()
