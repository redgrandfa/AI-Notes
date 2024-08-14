# 非內建的library? module?
# pip3 install rake_nltk
# pip3 install scikit-learn
import numpy as np
import pandas as pd

from rake_nltk import Rake

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# 內建的library? module?
from ast import literal_eval
import os

# 1 匯入資料 做成一個DataFrame
    # 相對路徑是從工作目錄開始找，可能需要先查看根目錄為何
    # import os
    # current_directory = os.getcwd()
    # print(current_directory)

print(os.getcwd())
t='recommendationSystem/D2_content/'
movies = pd.read_csv(t+'tmdb_5000_movies.csv') 
# 欄位有：budget genres homepage id(主鍵) keywords original_language original_title overview popularity production_companies

credits = pd.read_csv(t+'tmdb_5000_credits.csv')
#  欄位有：movie_id(主鍵)、title、cast、crew

credits.columns = ['id', 'title', 'cast', 'crew'] #把movie_id 改名成id

alldata = movies.merge(credits, on = 'id')
# print(alldata.head())
# print(alldata.columns)


#2 資料清理 
import warnings
warnings.filterwarnings('ignore') # 忽略警告

# Trim dataset to include relevant features
# dataframe 欄位篩選語法
df = alldata[['id', 'original_title', 'genres', 'keywords', 'overview', 'original_language', 'cast', 'crew']] 

# Parse stringed list features into python objects
# literal_eval是將字串當程式碼語法執行嗎?  本來是JSON字串 可拿來轉python objects?   
features = ['keywords', 'genres', 'cast', 'crew']
for i in features:
    df[i] = alldata[i].apply(literal_eval) #https://www.gushiciku.cn/pl/pIl0/zh-tw
    
# print(df)
# print('_______')

# Extract list of genres
def list_genres(x):
    l = [d['name'] for d in x]
    return(l)
df['genres'] = df['genres'].apply(list_genres)

# Extract top 3 cast members
def list_cast(x):
    l = [d['name'] for d in x]
    if len(l) > 3:
        l = l[:3]
    return(l)
df['cast'] = df['cast'].apply(list_cast)

# Extract top 5 keywords
def list_keywords(x):
    l = [d['name'] for d in x]
    if len(l) > 5:
        l = l[:5]
    return(l)
df['keywords'] = df['keywords'].apply(list_keywords)

# Extract director
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
df['director'] = df['crew'].apply(get_director)
# Drop the now unnecessary crew feature
df = df.drop('crew', axis = 1)

# 剛整理完的四個欄位 全小寫+去空白
# Clean features of spaces and lowercase all to ensure uniques
def clean_feat(x):
    if isinstance(x, list):
        return [i.lower().replace(" ","") for i in x]
    else:
        if isinstance(x, str):
            return x.lower().replace(" ", "")
        else:
            return ''
features = ['keywords', 'genres', 'cast', 'director']
for i in features:
    df[i] = df[i].apply(clean_feat)



# 3 Missing Values: NaN用空字串填充

# 3-1 印出來觀察
# missing = df.columns[df.isnull().any()] 
    # df.columns是Index物件，加上[布林陣列]，會剩下 是True的索引

# print( df[missing].isnull().sum().to_frame() )
    # missing是 有缺值的欄位們所成的Index 剩下 
    # .sum() => true算1，false算0 

# 3-2 (看見 overview欄 有三個NaN)
# From overview，replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

# 我：整張表格填充也可吧?
# df = df.fillna('') 
# print( df.isnull().sum() )


# 4 【擷取文章的詞】
        # use genres, keywords, overview, cast, and director
        #    to create a 【bag of words】 column ( plotwords).

# 初始化Rake()時，會報ERROR說需要 nltk的一些東西...想辦法debug補齊
import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt') 


# Initialize empty column 用來儲存擷取得到的東西
df['plotwords'] = ''

# 找出一文章中的所有關鍵字，並且 去除停用詞(stop words)如 a、the...；標點符號(puntuation characters) 
# function to get keywords from a text
# def get_keywords(x):
#     plot = x
def get_keywords(plot):
    # initialize Rake，預設using 【english stopwords】 from NLTK, and all 【punctuation characters】
    rake = Rake()

    # extract keywords from text
    rake.extract_keywords_from_text(plot)

    # get dictionary with keywords and scores，每個字有幾分
    scores = rake.get_word_degrees() 
    # print('scores:')
    # print(scores)
    
    # return new keywords as list, 【ignoring scores】
    return(list(scores.keys()))



# Apply function to generate keywords
df['plotwords'] = df['overview'].apply(get_keywords)
# print(df.head())
# print(df['plotwords'])

# 5  一個item =>  一個document 整理出 它的BOW (bag of words)  (每個文件 整理出它擁有的 詞集合 )
df_keys = pd.DataFrame() 
df_keys['title'] = df['original_title']
df_keys['keywords'] = '' #欄值全空

def bag_words(x):
    # print( x )
    return(' '.join(x['genres']) + ' ' 
        + ' '.join(x['keywords']) + ' ' 
        + ' '.join(x['cast']) + ' ' 
        + ' '.join(x['director']) + ' ' 
        + ' '.join(x['plotwords']))

df_keys['keywords'] = df.apply(bag_words, axis = 1) # df.apply會迭代每一列
# print(df_keys.head())


### 【重點是從這開始，前面都在...示範整理資料...】

# 6 【把每個item 變成一個 向量】
# create  matrix 

# 6-1 count詞
cv = CountVectorizer() #from sklearn.feature_extraction.text  
# print(cv.get_feature_names())
cv_mx = cv.fit_transform(df_keys['keywords']) 
# print(cv_mx)
# print(type(cv_mx)) #<class 'scipy.sparse._csr.csr_matrix'> 稀疏矩陣

# 6-2  TF-IDF版  https://ithelp.ithome.com.tw/articles/10228815
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
tf = TfidfVectorizer( #from sklearn.feature_extraction.text 
    analyzer='word' #和預設一樣
    #, ngram_range=(1, 3) # => 一元詞~三元詞 ，預設(1,1)
    , min_df=0 #df是document frequency  https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer
    , stop_words='english')
tfidf_matrix = tf.fit_transform( df_keys['keywords'] )
# print(tfidf_matrix)


#需要的話，可將稀疏矩陣 印成表格觀察

# 變稠密(dense)語法
# dense_matrix = sparse_matrix.toarray()
# dense_matrix = sparse_matrix.todense()

# df_dense = pd.DataFrame(
#     cv_mx.toarray()
#     , columns=cv.get_feature_names() )  # feature = 向量各座標軸是在計哪個'詞'
#     #,  index=['doc_trump', 'doc_election', 'doc_putin'] )
# df_dense2 = pd.DataFrame(
#     tfidf_matrix.todense()
#     , columns=tf.get_feature_names() ) 
# print( df_dense.columns )
# print( df_dense.columns )
# print( df_dense.index ) #電影標題 沒有進入矩陣中，後面須用 索引值 對應，去知道這向量屬於哪一部電影
# 欄是 各詞
# 列是 各文件索引值 


# 7 做出 相似度矩陣(正方形、pairwise關係)

# 7-1 Cosine Similarity
# (0,3,2) ˙ (1,2,4) = 0*1 + 3*2 + 2*4
# 內積：(0,3,2)．(1,2,4) = 0*1 + 3*2 + 2*4

# 對邊、鄰邊、斜邊
# cosine = 鄰邊/斜邊
# cos0 度 = 1  (0度 = 同向 = 極相似)
# cos90度 = 0

# from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_cv = cosine_similarity(cv_mx, cv_mx) # 正方矩陣 兩軸都是一群文件 
# print(cosine_sim_cv)

# cosine_sim_tf = cosine_similarity(tfidf_matrix) 
# print(cosine_sim_tf)

# from sklearn.metrics.pairwise import linear_kernel
# cosine_similarities = linear_kernel(tfidf_matrix)
# print(cosine_similarities) # 實際印出來，長得和cosine_similarity一樣

# 7-2 Jaccard Similarity
# 原理 https://pyshark.com/jaccard-similarity-and-jaccard-distance-in-python/#calculate-jaccard-similarity-in-python
# 兩個set的Jaccard Similarity 可以這樣算：
# def jaccard_similarity(A, B): 
#     # nominator = A.intersection(B)
#     nominator = A & B

#     # denominator = A.union(B)
#     denominator = A | B

#     similarity = len(nominator)/len(denominator)
#     return similarity



# from sklearn.metrics import pairwise_distances
# cos_dist_cv = pairwise_distances(cv_mx , metric='cosine') 
# print(cos_dist_cv)

# cos_dist_tf = pairwise_distances(tfidf_matrix, metric='cosine') 
# print(cos_dist_tf)

# cv_mx_dense = cv_mx.todense()
# jaccard_dist_cv = pairwise_distances(cv_mx_dense, metric='jaccard') #scipy distance metrics do not support sparse matrices
# print(jaccard_dist_cv)

# tfidf_matrix_dense = tfidf_matrix.todense()
# jaccard_dist_tf = pairwise_distances(tfidf_matrix_dense, metric='jaccard') 
# print(jaccard_dist_tf)


# 8 推薦系統，input是 一個電影title ，output是多個相關的電影title

# 但 正方矩陣 只有item的索引值； 需要 title -> 索引值 -> 相似度矩陣才能找到指定電影那一列 
# 做一個Series，列索引 是 電影title，值是 電影索引值
# create list of indices for later matching
indices = pd.Series(df_keys.index, index = df_keys['title'])
# print(indices)

def recommend_movie(title, n = 10, sim_matrix = cosine_sim_cv):
# distance是 1-similarity，排序要改由小至大
# def recommend_movie(title, n = 10, dist_matrix = jaccard_dist_tf):
    # retrieve matching movie title index
    if title not in indices.index: 
    # if title not in titles:
        print("Movie not in database.")
        return
    else:
        idx = indices[title]
        # idx = titles.index(title)
    
    # 排序
    scores = pd.Series(sim_matrix[idx]).sort_values(ascending = False)
    # scores = pd.Series(dist_matrix[idx]).sort_values()
    
    # top n most similar movies indexes
    # use 1:n because 0 is the same movie entered
    top_n_idx = list(scores.iloc[1:n].index)
        
    return df_keys['title'].iloc[top_n_idx]


# 前10
# print( recommend_movie('Avatar') )
# 前5
# print( recommend_movie('Toy Story', n = 5) )

