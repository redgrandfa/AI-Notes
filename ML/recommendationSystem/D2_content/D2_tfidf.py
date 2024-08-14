# https://towardsdatascience.com/building-a-content-based-recommender-system-for-hotels-in-seattle-d724f0a32070
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import random
import plotly.graph_objs as go #pip3 install plotly
# import plotly.plotly as py #The plotly.plotly module is 【deprecated】, please install the chart-studio package and use the chart_studio.plotly module instead. 
import chart_studio.plotly as py
import cufflinks #pip3 install cufflinks
pd.options.display.max_columns = 30

from IPython.core.interactiveshell import InteractiveShell #pip3 install IPython
import plotly.figure_factory as ff 

InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='solar')
# df = pd.read_csv('Seattle_Hotels.csv', encoding="latin-1")
df = pd.read_csv('recommendationSystem/D2/Seattle_Hotels.csv', encoding="latin-1")

# print('We have ', len(df), 'hotels in the data')
# print(df.head())
# 避免編碼問題 在終端機印 而非輸出
# chcp 65001
# python .\recommendationSystem\D2\D2_tfidf.py



# def print_description(index):
#     example = df[df.index == index][['desc', 'name']].values[0]
#     if len(example) > 0:
#         print(example[0])
#         print('Name:', example[1])

# print_description(10)

def get_top_n_words(corpus, n=None): # corpus中譯 = 語料庫
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# #作圖
# common_words = get_top_n_words(df['desc'], 20)
# df1 = pd.DataFrame(common_words, columns = ['desc' , 'count'])
# df1.groupby('desc').sum()['count'].sort_values().iplot(kind='barh', yTitle='Count', linecolor='black', title='Top 20 words in hotel description before removing stop words')


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['desc_clean'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index)