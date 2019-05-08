import string
import re
import json
import pandas as pd
import csv
import jellyfish
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import os,glob


with open('dataset.json', 'r') as f:#read the dataset for learning
    data = json.load(f)
df = pd.DataFrame(data)
# print(df)

def pre_process(text):
    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # remove special characters and digit
    text = re.sub("(\\d|\\W)+", " ", text)

    return text


df['text'] = df['file.contenu']
df['text'] = df['text'].apply(lambda x: pre_process(x))
# show second text for fun
#df['text'][2]


from sklearn.feature_extraction.text import CountVectorizer

def get_stop_words(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

# load a set of stop words
stopwords = get_stop_words("stopwords_fr.txt")

# get the text column
docs = df['text'].tolist()
# print(docs[:20])


# create a vocabulary of words,
# ignore words that appear in 85% of documents,
# eliminate stop words
cv = CountVectorizer(max_df=0.85, stop_words=stopwords)

word_count_vector = cv.fit_transform(docs)  # create the voc and returns the term doc matrix
#list(cv.vocabulary_.keys())[:10]
word_count_vector.shape  # no of doc in our dataset and voc size

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)#calc the IDF
tfidf_transformer.idf_

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=100000):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

# you only needs to do this once
feature_names=cv.get_feature_names()

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform(docs))


#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
# print(sorted_items)
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items)
#for key in keywords:
    #print(key,keywords[key])

# generate cloudtag 
def CloudTag(dic, gTitle):
    wordcloud = WordCloud(width=800, height=800,
                        background_color='white', normalize_plurals=False,max_words=50,
                        min_font_size=10).generate_from_frequencies(dic)

# plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.title(gTitle)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()
    # plt.savefig(gTitle+'.png')



# dictFile = open('ana.csv').read().splitlines()
dir = r'venv/test'
dictFile = os.listdir(dir) #reading all dictionaries form a path

lst = [os.path.splitext(x)[0] for x in dictFile]

# filtering tfidf with the dictionaries
max_dist = 2
new_keywords = []


for file2 in dictFile:
   icdFile = open(r'venv/test/' + file2).read().splitlines()
   for key in keywords:
       terms = key
       tf_idf = keywords[key]
       min_dissim = max_dist
       for wrdFile in icdFile:
           cmpFile = jellyfish.levenshtein_distance(terms, wrdFile)
           if (cmpFile < min_dissim):
               min_dissim = cmpFile
           if min_dissim == 0:
               break
       new_tf_idf = tf_idf + ((max_dist - min_dissim) / max_dist)
       new_keywords.append((terms, new_tf_idf))
   # print(new_keywords)
       d = dict(new_keywords)
   # print(d)

   # # print("hello")
   printTag = CloudTag(d, file2)
   #     # printTag = CloudTag(d)
   #     printTag





