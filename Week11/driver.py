import glob, csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

train_filename = "imdb_tr.csv"
unigram_output_file = "unigram.output.txt"
unigram_tfidf_output_file = "unigramtfidf.output.txt"
bigram_output_file = "bigram.output.txt"
bigram_tfidf_output_file = "bigramtfidf.output.txt"

# PROD
# train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
# test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation
# negative_path = "/neg/"
# positive_path = "/pos/"
# stop_words_file_path = "./stopwords.en.txt"

# DEV
train_path = ".\\resource\\lib\\publicdata\\aclImdb\\train\\" # use terminal to ls files under this directory
test_path = ".\\imdb_te.csv" # test data for grade evaluation
negative_path = "\\neg\\"
positive_path = "\\pos\\"
stop_words_file_path = ".\\stopwords.en.txt"


def load_stop_words(stop_words_file_path):
    stop_words = []
    stop_words_file = open(stop_words_file_path, "rb")
    for word in stop_words_file:
        word = word.replace("\n", "")
        stop_words.append(word)
    stop_words_file.close()

    return stop_words


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    # Implement this module to extract
    # and combine text files under train_path directory into
    # imdb_tr.csv. Each text file in train_path should be stored
    # as a row in imdb_tr.csv. And imdb_tr.csv should have two
    # columns, "text" and label

    # Create and open file for output
    output_file_path = outpath + name
    output_file = open(output_file_path, "wb")
    output_csv_writer = csv.DictWriter(output_file, fieldnames=['row_number', 'text', 'polarity'], delimiter=',')
    output_csv_writer.writeheader()
    output_row = 0

    # Process Negative feedback entries. Polarity: 0
    negative_files = glob.glob(inpath + negative_path + "*.txt")
    for file_path in negative_files:
        file = open(file_path, "rb")
        text = file.readline().replace('<br />', '')  # Remove <br /> tags
        file.close()
        output_csv_writer.writerow({'row_number': output_row, 'text': text, 'polarity': 0})
        output_row += 1

    # Process Positive feedback entries. Polarity: 1
    positive_files = glob.glob(inpath + positive_path + "*.txt")
    for file_path in positive_files:
        file = open(file_path, "rb")
        text = file.readline().replace('<br />', '')  # Remove <br /> tags
        file.close()
        output_csv_writer.writerow({'row_number': output_row, 'text': text, 'polarity': 1})
        output_row += 1

    output_file.close()
    pass


def sgd_unigram(train_data, test_data, stop_words):

    # Initializes the Count Vectorizer
    train_counter = CountVectorizer(decode_error='ignore', ngram_range=(1, 1), stop_words=stop_words)

    # Creates the vocabulary of train data
    train_vocabulary = train_counter.fit_transform(train_data.text)

    # Train the SGD classifier with this training data
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(train_vocabulary, train_data.polarity)

    # Creates the vocabulary of test data
    test_counter = CountVectorizer(decode_error='ignore', ngram_range=(1, 1), stop_words=stop_words, vocabulary=train_counter.vocabulary_)
    test_vocabulary = test_counter.fit_transform(test_data.text)
    test_predictions = clf.predict(test_vocabulary)

    return test_predictions


def sgd_bigram(train_data, test_data, stop_words):

    # Initializes the Count Vectorizer
    train_counter = CountVectorizer(decode_error='ignore', ngram_range=(1, 2), stop_words=stop_words)

    # Creates the vocabulary of train data
    train_vocabulary = train_counter.fit_transform(train_data.text)

    # Train the SGD classifier with this training data
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(train_vocabulary, train_data.polarity)

    # Creates the vocabulary of test data
    test_counter = CountVectorizer(decode_error='ignore', ngram_range=(1, 2), stop_words=stop_words, vocabulary=train_counter.vocabulary_)
    test_vocabulary = test_counter.fit_transform(test_data.text)
    test_predictions = clf.predict(test_vocabulary)

    return test_predictions


def sgd_unigram_tridf(train_data, test_data, stop_words):

    # Initializes the Count Vectorizer
    train_counter = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 1), stop_words=stop_words)

    # Creates the vocabulary of train data
    train_vocabulary = train_counter.fit_transform(train_data.text)

    # Train the SGD classifier with this training data
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(train_vocabulary, train_data.polarity)

    # Creates the vocabulary of test data
    test_counter = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 1), stop_words=stop_words, vocabulary=train_counter.vocabulary_)
    test_vocabulary = test_counter.fit_transform(test_data.text)
    test_predictions = clf.predict(test_vocabulary)

    return test_predictions


def sgd_bigram_tridf(train_data, test_data, stop_words):

    # Initializes the Count Vectorizer
    train_counter = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2), stop_words=stop_words)

    # Creates the vocabulary of train data
    train_vocabulary = train_counter.fit_transform(train_data.text)

    # Train the SGD classifier with this training data
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(train_vocabulary, train_data.polarity)

    # Creates the vocabulary of test data
    test_counter = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2), stop_words=stop_words, vocabulary=train_counter.vocabulary_)
    test_vocabulary = test_counter.fit_transform(test_data.text)
    test_predictions = clf.predict(test_vocabulary)

    return test_predictions


def print_status(data, predictions):
    total = 0.0
    errors = 0.0
    for i in range(len(data)):
        if data[i] != predictions[i]:
            errors += 1
        total += 1

    print "Prevision_rate: " + str((1 - (errors / total)) * 100) + "% (" + str(errors) + "/" + str(total) + ")"


if __name__ == "__main__":

    # Import data to Train Data csv
    # imdb_data_preprocess(train_path)

    # Read csv files to memory
    train_data = pd.read_csv(train_filename, delimiter=',')
    test_data = pd.read_csv(test_path, delimiter=',')

    # Load Stop Words
    stop_words = load_stop_words(stop_words_file_path)

    # train a SGD classifier using unigram representation,
    # predict sentiments on imdb_te.csv, and write output to
    # unigram.output.txt
    # unigram_predictions = sgd_unigram(train_data, test_data, stop_words)
    # np.savetxt(unigram_output_file, unigram_predictions, fmt='%.0f')

    # train a SGD classifier using bigram representation,
    # predict sentiments on imdb_te.csv, and write output to
    # bigram.output.txt
    bigram_predictions = sgd_bigram(train_data, test_data, stop_words)
    np.savetxt(bigram_output_file, bigram_predictions, fmt='%.0f')

    # train a SGD classifier using unigram representation
    # with tf-idf, predict sentiments on imdb_te.csv, and write
    # output to unigramtfidf.output.txt
    # unigram_tridf_predictions = sgd_unigram_tridf(train_data, test_data, stop_words)
    # np.savetxt(unigram_tfidf_output_file, unigram_tridf_predictions, fmt='%.0f')

    # train a SGD classifier using bigram representation
    # with tf-idf, predict sentiments on imdb_te.csv, and write
    # output to bigramtfidf.output.txt
    bigram_tridf_predictions = sgd_bigram_tridf(train_data, test_data, stop_words)
    np.savetxt(bigram_tfidf_output_file, bigram_tridf_predictions, fmt='%.0f')

    # Training Performance:
    unigram_predictions = sgd_unigram(train_data, train_data, stop_words)
    bigram_predictions = sgd_bigram(train_data, train_data, stop_words)
    unigram_tridf_predictions = sgd_unigram_tridf(train_data, train_data, stop_words)
    bigram_tridf_predictions = sgd_bigram_tridf(train_data, train_data, stop_words)

    print "unigram_prediction: "
    print_status(train_data.polarity, unigram_predictions)
    print "bigram_prediction: "
    print_status(train_data.polarity, bigram_predictions)
    print "unigram_tridf_prediction :"
    print_status(train_data.polarity, unigram_tridf_predictions)
    print "bigram_tridf_prediction: "
    print_status(train_data.polarity, bigram_tridf_predictions)

    pass
