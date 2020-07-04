from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import regex
from flask import Flask, request, jsonify, render_template
import nltk
# nltk.download('stopwords')

def read_article(data):

    article = data.split(". ")
    sentences = []
    for sentence in article:
        review = regex.sub("[^A-Za-z0-9]",' ', sentence)
        sentences.append(review.replace("[^a-zA-Z]", " ").split(" "))        
    sentences.pop()     
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words) #makes a vector of len all_words
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - nltk.cluster.util.cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=10):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)
    
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
#     print("\n\n---------------\nIndexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    # print("\n")
    # print("*"*140)
    # print("\n\nSUMMARY: \n---------\n\n", ". ".join(summarize_text))
    a = ". ".join(summarize_text)
    return a

#----------FLASK-----------------------------#

app = Flask(__name__)
@app.route('/templates', methods =['POST'])
def original_text_form():
		text = request.form['input_text']
		number_of_sent = request.form['num_sentences']
# 		print("TEXT:\n",text)
		summary = generate_summary(text,int(number_of_sent))
# 		print("*"*30)
# 		print(summary)
		return render_template('index1.html', title = "Summarizer", original_text = text, output_summary = summary, num_sentences = 5)

@app.route('/')
def homepage():
	title = "TEXT summarizer"
	return render_template('index1.html', title = title)

if __name__ == "__main__":
	app.debug = True
	app.run()
