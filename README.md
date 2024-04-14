# Vector Space Retrieval Model
This project implements a Vector Space Model (VSM) for Information Retrieval using the Term Frequency-Inverse Document Frequency (TF-IDF) weighting scheme. The VSM is a classical approach to document retrieval in information retrieval systems, where documents and queries are represented as vectors in a high-dimensional space, and similarity between documents and queries is computed using cosine similarity.

## Features
* Supports free word queries.
* Utilizes the TF-IDF weighting scheme for retrieval.
* Ranks document based on the similarity scores calculated using Cosine Similarity.
* Provides a simple GUI interface for user interaction.

## Getting Started
To run the information retrieval system, follow these steps:

* Ensure you have Python 3.12 installed.
* Install NLTK and tkinter 'pip install NLTK' and 'pip install tkinter' repectively.
* Make sure Stopword-List.txt and the Research Paper directory containing all the documents is in your current working directory.
* Update the path in the ASSET PATH to point to the asset folder.
* Run the files in an IDE.
* Run this command to download the tokennizer nltk.download('punkt')
* Run the 'weights_calculation.py' script first using 'python weights_calculation.py' to create and save the weights.
* Then run the 'query_processing.py' using 'python query_processing.py' for queries.
* Use the tkinter GUI interface to input queries and press 'Search' button to retrieve the required document IDs.
* Press the 'Exit' button to exit the program.

## License
 This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
* This project was inspired by information retrieval concepts and algorithms.
* Special thanks to the developers of NLTK for providing essential natural language processing tools.
* Special thanks to Tom Schimansky for developing CustomTkinter that was used for making the GUI.