import os
import math as m
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def get_stopwords():
    """
    This function is used to extract stopwords from 'Stopword-List.txt' file.

    It reads each line from the file, and if the line is not empty, it appends the line to the stopwords list.
    The function continues this process until it reaches the end of the file. Assumes the file is in your current working directory.

    Returns:
        stopwords (list): A list of stopwords extracted from the file.
    """

    stopwords = []
    with open('Stopword-List.txt', 'r') as f: # the 'Stopword-List.txt' file is opened in read mode
        while True:
            text = f.readline() # each line from the file is read one by one
            if not text: # if the line read is empty (which means end of file), the loop is broken
                break
            stopwords.append(text) # else append the read line to the stopwords list

    stopwords = [c.rstrip(' \n') for c in stopwords if c != '\n'] # a new list is created from stopwords, excluding any newline characters. Newline characters are also removed from the strings.
    return stopwords


def get_docIDs():
    """
    This function is used to extract document IDs based on the names of the files in the 'ResearchPapers' directory.

    It gets the current working directory and lists all the files in the 'ResearchPapers' directory. 
    It then extracts the document IDs from the names of these files, sorts them, and returns the sorted list.
    Assumes the 'ResearchPapers' folder is in your current working directory.

    Returns:
        docID (list): A sorted list of document IDs extracted from the file names in the 'ResearchPapers' directory.
    """

    curr_dir = os.getcwd() # get the current directory
    docID = [int(c.rstrip('.txt')) for c in os.listdir(curr_dir + '\ResearchPapers')] # extract the docIDs from the names of the files in the ResearchPapers directory
    docID.sort()
    return docID


def calculate_TF(total_tokens):
    """
    This function calculates the Term Frequency weights for the terms and saves it in a DataFrame.

    Args:
        total_tokens (list): A list of processed tokens.

    Returns:
        tf (DataFrame): A pandas DataFrame representing the Term Frequency weights.
    """

    tf = {} # declare an empty dictionary for the term frequency weights
    porter_stemmer = PorterStemmer() # initialize the stemmer
    doc = get_docIDs() # get the docIDs

    for i, tokens in enumerate(total_tokens): # loop through each token in total_tokens, and then loop through each word in the token
        for word in tokens:
            word = porter_stemmer.stem(word) # stem the word
            if word[-1] == "'": # if the word ends with an apostrophe, remove it
                word = word.rstrip("'")
            if word in tf: # if the word is already in the index, add the docID to the index
                if doc[i] in tf[word]:
                    freq = tf[word][doc[i]]
                    tf[word][doc[i]] = freq + 1
                else:
                    tf[word][doc[i]] = 1
            else: # add the word in the index along with the docID and the frequency
                tf[word] = {doc[i]: 1}

    for word in tf.keys(): # calculate the log term frequency weights
        for doc in tf[word].keys():
            tf[word][doc] = 1 + m.log(tf[word][doc], 10) # normalize the term frequency weights

    tf = pd.DataFrame(tf) # convert the dictionary to a Pandas DataFrame
    tf = tf.transpose() # since the DataFrame will be in the form of words as columns and docIDs as rows, we transpose it to have docIDs as columns and words as rows
    tf.fillna(0, inplace=True) # fill the NaN values with 0

    print("Term Frequency Weights created")
    return tf


def calculate_IDF(tf):
    """
    This function calculates the Inverse Document Frequency weights for the terms and saves it in a DataFrame.

    Args:
        total_tokens (list): A list of processed tokens.

    Returns:
        idf (DataFrame): A pandas DataFrame representing the Inverse Document Frequency weights.
    """

    df = {} # a dictionary to store the document frequency of each word
    idf = {} # a dictionary to store the inverse document frequency of each word
    doc = get_docIDs() # get the docIDs

    for keys in tf.index: # loop through each word in the index
        frequency = list(tf.loc[keys].value_counts()) # get the frequency of each word in the document
        df[keys] = len(doc) - frequency[0] # calculate the document frequency of each word

    for keys in tf.index: # loop through each word in the document frequency index
        idf[keys] = m.log(len(doc)/df[keys], 10) # calculate the inverse document frequency of each word

    idf = pd.DataFrame(idf, index=[0]) # convert the dictionary to a Pandas DataFrame

    print("Inverse Document Frequency Weights calculated")
    return idf


def preprocessing():
    """
    This function is used to preprocess the text files in the 'ResearchPapers' directory.

    It reads each file, tokenizes the text, removes punctuation and converts the text to lowercase. 
    It also splits the tokens at '.' and '-'. Assumes the 'ResearchPapers' folder is in your current working directory.

    Returns:
        total_tokens (list): A list of preprocessed tokens from all the files.
    """

    total_tokens = [] # an empty list to store the tokens from all the files
    doc = get_docIDs() # get the docIDs
    stopwords = get_stopwords() # get the stopwords
    stemmer = PorterStemmer() # create a stemmer object

    for i in doc: # iterate through each doc
        tokens = []
        with open('ResearchPapers/' + str(i) + '.txt', 'r') as f: # open the file corresponding to the current document ID
            while True:
                text = f.readline() # read a line from the file
                if not text: # if the line is empty (which means end of file), break the loop
                    break
                tokens += word_tokenize(text) # tokenize the line and add the tokens to the list

        j = 0
        while j < len(tokens): # loop through each token
            if tokens[j] not in stopwords and len(tokens[j]) <= 45: # filter out the stopwords and tokens with length greater than 45
                # remove symbols and numbers from the start and end of the token and also apply case folding
                tokens[j] = tokens[j].strip('0123456789!@#$%^&*()-_=+[{]}\|;:\'",<.>/?`~').casefold()
                if '.' in tokens[j]: # if '.' exists in a word, split the word at that point and add the splitted words at the end of the tokens list while removing the original word
                    word = tokens[j].split('.')
                    del tokens[j]
                    tokens.extend(word)
                elif '-' in tokens[j]: # do the same for words with '-'
                    word = tokens[j].split('-')
                    del tokens[j]
                    tokens.extend(word)
            j += 1 # move the index forward
        tokens = [stemmer.stem(c) for c in tokens if c.isalpha() and c not in stopwords and len(c) >= 2] # filter out any strings that contain symbols, numbers, etc.
        total_tokens.append(tokens) # add the processed tokens as a seperate list. Did this to keep track of which tokens appear in which docs (needed to construct indexes). List at index 0 indicate tokens found in doc 1 and so on.

    return total_tokens


def calculate_TFIDF(TF, IDF):
    '''
    This function calculates the TF-IDF weights for the terms.

    Args:
        TF (DataFrame): The Term Frequency weights.
        IDF (DataFrame): The Inverse Document Frequency weights.

    Returns:
        vectors (DataFrame): A pandas DataFrame representing the TF-IDF weights.
    '''

    vectors = pd.DataFrame(index=TF.index, columns=TF.columns) # create a DataFrame with the same index and columns as the TF DataFrame
    for term in TF.index: # loop through each term in the index
        if pd.isna(term): # special case of 'nan' term
            vectors.loc[term] = TF.loc[term] * IDF.loc[0, 'nan']
        else: # calculate the TF-IDF weights
            vectors.loc[term] = TF.loc[term] * IDF.loc[0, term]
    vectors.transpose() # transpose the DataFrame

    print("TF-IDF Weights calculated")
    return vectors


def save_weights():
    """
    This function calls the preprocessing function for the processed terms, calculates the TF and IDF weights for each term, using the calculate_TF and calculate_IDF function, and then saves the DataFrame to 'tf.csv' and 'idf.csv' respectively.

    The preprocessing function is expected to return a list of tokens.
    The calculate_TF function takes the tokens as input and returns a DataFrame where each row represent the TF weights for each term.
    The calculate_IDF function also takes the tokens as input and returns a DataFrame where each value represent the IDF weight for each term.

    The output files 'tf-idf.csv' and 'idf.csv' contain the comma seperated values of the term frequency and inverse document frequency weights for each term.
    """

    tokens = preprocessing() # preprocessing function is called, returns the processed tokens
    tf = calculate_TF(tokens) # calculate_TF function is called, returns the TF weights
    idf = calculate_IDF(tf) # create_positional_index function is called, returns the IDF weights
    tf_idf = calculate_TFIDF(tf, idf) # calculate the TF-IDF weights

    tf_idf.to_csv('tf-idf.csv') # output TF-IDF DataFrame to CSV including the index
    print("TF-IDF Weights saved")

    idf.to_csv('idf.csv') # output IDF DataFrame to CSV including the index
    print("Inverse Document Frequency Weights saved")


def main():
    if (not os.path.isfile('tf-idf.csv') or not os.path.isfile('idf.csv')): # check if the indexes already exist, if they don't, call the save_indexes function
        save_weights()
    else:
        print("Weights are already calculated")


if __name__ == '__main__':
    main() # execute the main function