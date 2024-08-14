# The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together.

# (390^2+180^2+50^2)^0.5

from sklearn.feature_extraction.text import CountVectorizer #TfidfVectorizer
import pandas as pd


# Create the Document Term Matrix 文件向量化
count_vectorizer = CountVectorizer(stop_words='english') #多一行?
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.

doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), #像是取得 各座標(字眼)的名稱
                  index=['doc_trump', 'doc_election', 'doc_putin'])

# print(df)


# used the TfidfVectorizer() instead of CountVectorizer(), because it would have downweighted words that occur frequently across docuemnts.

# use cosine_similarity() to get the final output.
cosine_similarity(df, df)


### Soft Cosine Similarity
# 語意上的相似 words similar in meaning should be treated as similar
# Soft cosines can be a great feature if you want to use a similarity metric that can help in clustering or classification of documents. 

documents = [doc_trump, doc_election, doc_putin, doc_soup, doc_noodles, doc_dosa]

# https://www.machinelearningplus.com/nlp/gensim-tutorial/
import gensim
# upgrade gensim if you can't import softcossim
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
print(gensim.__version__)


# Download the FastText model
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')


# soft cosines, you need the dictionary (a map of word to unique id), the corpus (word counts)

# Prepare a dictionary and a corpus.
dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

# Prepare the similarity matrix
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

# Convert the sentences into 【bag-of-words vectors】.
sent_1 = dictionary.doc2bow(simple_preprocess(doc_trump))
sent_2 = dictionary.doc2bow(simple_preprocess(doc_election))
sent_3 = dictionary.doc2bow(simple_preprocess(doc_putin))
sent_4 = dictionary.doc2bow(simple_preprocess(doc_soup))
sent_5 = dictionary.doc2bow(simple_preprocess(doc_noodles))
sent_6 = dictionary.doc2bow(simple_preprocess(doc_dosa))

sentences = [sent_1, sent_2, sent_3, sent_4, sent_5, sent_6]
# Compute soft cosine similarity
# print(softcossim(sent_1, sent_2, similarity_matrix))

import numpy as np
import pandas as pd

def create_soft_cossim_matrix(sentences):
    len_array = np.arange(len(sentences))
    xx, yy = np.meshgrid(len_array, len_array)
    cossim_mat = pd.DataFrame([[round(softcossim(sentences[i],sentences[j], similarity_matrix) ,2) for i, j in zip(x,y)] for y, x in zip(xx, yy)])
    return cossim_mat

soft_cosine_similarity_matrix(sentences)