import os
import math as m
import numpy as np
import pandas as pd
import tkinter as tk
from pathlib import Path
from nltk.stem import PorterStemmer
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


def extract_weights():
    """
    This function is used to extract the TF-IDF and the IDF weights from their respective csv files.

    Returns:
        TF-IDF (DataFrame): The extracted TF-IDF weights.
        IDF (DataFrame): The extracted IDF weights.
    """

    TF_IDF = pd.read_csv('tf-idf.csv', index_col=0)  # read TF-IDF DataFrame from CSV
    IDF = pd.read_csv('idf.csv', index_col=0) # read IDF DataFrame from CSV

    return TF_IDF, IDF


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


def calculate_QueryVector(query, IDF):
    """
    This function calculates the query vector based on the query terms and the IDF weights.

    Args:
        query (list): A list of query terms.
        IDF (DataFrame): The Inverse Document Frequency weights.

    Returns:
        query_vector (dictionary): The query vector based on the query terms and the IDF weights.
    """

    query_vector = {} # create a list of zeros with the length of the columns in the IDF DataFrame

    for term in IDF.columns: # loop through each term in the IDF columns
        if term in query: # if the term is in the query
            query_vector[term] = 1 + m.log(query.count(term), 10) # calculate the log term frequency weight
            query_vector[term] *= IDF.loc[0, term] # multiply the log term frequency weight by the IDF weight
        else:
            query_vector[term] = 0 # if the term is not in the query, set the weight to zero

    norm = sum([query_vector[x] ** 2 for x in query_vector.keys()]) ** 0.5 # calculate the norm of the query vector

    if norm != 0: # if the norm is not zero
        for term in query_vector:
            query_vector[term] = query_vector[term] / norm # normalize the query vector

    return query_vector


def QueryProcessing(query):
    '''
    This function processes the query by stemming the words, removing stopwords, and calculating the TF-IDF weights of the terms.
    
    Args:
        query (str): The query string to be processed.
    
    Returns:
        vectors (DataFrame): The DataFrame containing the TF-IDF weights of all the terms.
        query_vector (DataFrame): The DataFrame containing TF-IDF weights the query terms.
    '''

    vectors, IDF = extract_weights() # # read the TF-IDF and IDF DataFrames from CSV

    query_vector = calculate_QueryVector(query, IDF) # calculate the query vector
    query_vector = pd.DataFrame(query_vector, index=[0]) # create a DataFrame from the query vector

    return vectors, query_vector


def find_sim(query):
    '''
    This function calculates the similarity scores between the query vector and the document vectors.

    Args:
        query (string): The query string to be processed.

    Returns:
        scores (string): a string containing the document IDs with similarity scores greater than or equal to 0.05, sorted and ranked with respect to their scores.
    '''

    query = query.split() # split the query into words
    porter_stemmer = PorterStemmer() # initialize the stemmer
    stopwords = get_stopwords() # get the stopwords
    doc = get_docIDs() # get the document IDs

    query = [porter_stemmer.stem(word).rstrip("'").casefold() for word in query if word not in stopwords] # stem the words in the query and remove the stopwords
    
    vectors , query_vector = QueryProcessing(query) # calculate the query vector

    score = {} # create a dictionary to store the similarity scores
    for docID in doc:
        score[docID] = np.dot(vectors[str(docID)], query_vector.transpose()) # calculate the similarity score for each document

    score = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)} # sort the similarity scores in descending order
    score = {k: score[k] for k in score if score[k] >= 0.05} # remove any documents with a similarity score less than 0.05

    score = [k for k in score.keys()]
    score = ' '.join(map(str, score))

    return score


def process_query():
    """

    This function retrieves a user's query from a GUI text entry field, processes the query, and displays the result in a GUI label.
    
    """

    query = enter_query.get() # get the query from the text entry field
    result = find_sim(query)

    if result == '': # if no documents are found
        result = 'No documents found'

    output_label.configure(state='normal') # enable the output label
    output_label.delete(0.0, tk.END) # clear the output label
    output_label.insert(0.0, result) # insert the result into the output label
    output_label.configure(state='disabled') # again disable the output label


OUTPUT_PATH = Path(__file__).parent # get the output path
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\Hp\Desktop\Information Retrieval\Assignment 2\build\assets\frame0") # get the assets path

def relative_to_assets(path: str) -> Path: # function to get the relative path of the assets
    return ASSETS_PATH / Path(path)

window = Tk() # create a new window
window.geometry("700x400") # set the window size
window.configure(bg = "#FFFFFF") # set the window background color

# create a canvas to place the widgets
canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 400,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

# place the canvas at the top left corner of the window
canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    700.0,
    49.0,
    fill="#51ADD4",
    outline="")

# place the Vector Space Model text, that has been converted to an image, at the given co-ordinates on the canvas
vsm_text_image = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    130.0,
    24.0,
    image=vsm_text_image
)

# place the image at the given co-ordinates for the query entry box on the canvas
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    350.0,
    130.5,
    image=entry_image_1
)

# place the text entry field at the given co-ordinates on the canvas with the foreground and background matching the image for the entry box
enter_query = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
enter_query.place(
    x=52.0,
    y=110.0,
    width=596.0,
    height=39.0
)

# place the image at the given co-ordinates for the output label box on the canvas
output_label_image = PhotoImage(
    file=relative_to_assets("entry_2.png"))
output_label_bg = canvas.create_image(
    350.0,
    313.5,
    image=output_label_image
)

# create the output label at the given co-ordinates on the canvas with the foreground and background matching the image for the output label box
output_label = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
output_label.place(
    x=52.0,
    y=293.0,
    width=596.0,
    height=39.0
)

# disable the output label so that it cannot be edited
output_label.configure(
    state='disabled',
    font=('Monteserrat', 10)
)

# place the "Enter Query" text, that has been converted to an image, above the query entry box
image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    100.0,
    88.0,
    image=image_image_2
)

# place the "Result" text, that has been converted to an image, above the output label box
image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    80.0,
    275.0,
    image=image_image_3
)

# place the "Search" button, that has been converted to an image, at the given co-ordinates on the canvas
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))

# create the button that is to be placed on the canvas for the searching, pressing this button will call the process_query function
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=process_query,
    relief="flat"
)
button_1.place(
    x=176.0,
    y=166.0,
    width=348.0,
    height=41.0
)

# place the "Exit" button, that has been converted to an image, at the given co-ordinates on the canvas, pressing this button will close the window
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=window.destroy,
    relief="flat"
)
button_2.place(
    x=176.0,
    y=222.0,
    width=348.0,
    height=41.0
)

# place the image at the given co-ordinates for the logo on the canvas
image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    657.0,
    24.0,
    image=image_image_4
)
window.resizable(False, False) # make the window non-resizable


if __name__ == "__main__":
    window.mainloop() # run the window