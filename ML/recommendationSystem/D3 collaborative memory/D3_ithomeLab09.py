# coding=utf-8 
import pandas as pd

input_data_file_movie = "recommendationSystem/D3/ml-100k/u.item"
input_data_file_rating = "recommendationSystem/D3./ml-100k/u.data"

movie = pd.read_csv(input_data_file_movie, sep='|', encoding='ISO-8859-1', names=['movie_id', 'movie_title'], usecols = [0,1,])
rating = pd.read_csv(input_data_file_rating, sep='\t', encoding='ISO-8859-1', names=["user_id","movie_id","rating"], usecols = [0,1,2])
#print(movie.head())
#print(rating.head())

data = pd.merge(movie,rating)
# print(data.head())



# A.  ITEM-ITEM 協同過濾相似性(Similarity)
# 先做出USER-ITEM Matrix：pivot成 列是user_id， 欄是movie_title
pivot_table_U_I = data.pivot_table(index = ["user_id"],columns = ["movie_title"],values = "rating")
print(pivot_table_U_I.head(10))

# 計算和"Bad Boys (1995)"的item相似度排行 (相似度使用correlation)
movie_watched = pivot_table_U_I["Bad Boys (1995)"]
similarity_with_other_movies = pivot_table_U_I.corrwith(movie_watched)
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
print( similarity_with_other_movies.head() )


# B.  USER-USER 協同過濾相似性(Similarity)
# 先做出ITEM-USER Matrix：pivot成 列是movie_title，欄是user_id
pivot_table_I_U = data.pivot_table(index =["movie_title"],columns = ["user_id"],values = "rating")
print(pivot_table_I_U.head(10))

# 計算和"user_id=10"的user相似度排行 (相似度使用correlation)
target_user = pivot_table_I_U[10]
similarity_with_other_users = pivot_table_I_U.corrwith(target_user)
similarity_with_other_users = similarity_with_other_users.sort_values(ascending=False)
print( similarity_with_other_users.head() )



#針對單一使用者瀏覽商品時，作出即時的推薦，
# 如果，要執行所有的使用者的推薦清單，採用平行計算，因為每一個相似性的計算都是可以獨立執行的。
# ---------

# item based 為何說  比較容易推薦到長尾(long tail)的商品? 
#     假設在瀏覽一個不很熱銷，但固定有剛需(ex.衛生紙)的商品

#     跟此商品相似的也會是長尾

#     跟此item最像的幾個商品，前幾高分？

#     會有跟你類似的一群人都有經常固定買的商品，雖然不是所有人都在買


# user based 為何說 是推薦最多使用者購買的商品，所以，常會推薦熱銷品。
#     user based ⇒ 跟此user最像的幾個人，前幾愛買的？

print('----------------------------')


import numpy as np
from sklearn.neighbors import NearestNeighbors

# us_canada_user_rating_matrix = ??
# us_canada_user_rating_pivot = ??

# 瞎猜
us_canada_user_rating_matrix = rating 
# us_canada_user_rating_matrix = data  # 使model_knn.fit 報 ValueError: could not convert string to float: 'Toy Story (1995)'
us_canada_user_rating_pivot = pivot_table_U_I 

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
print(type(model_knn)) # <class 'sklearn.neighbors._unsupervised.NearestNeighbors'>
print(model_knn) #NearestNeighbors(algorithm='brute', metric='cosine')

# us_canada_user_rating_matrix 即商品的評價表(User-Item Rating Table)
model_knn.fit(us_canada_user_rating_matrix)
#UserWarning: X does not have valid feature names, but NearestNeighbors was fitted with feature names

query_index=0
distances, indices = model_knn.kneighbors(np.array(us_canada_user_rating_pivot.iloc[query_index, :]).reshape(1, -1), n_neighbors = 6)

#NearestNeighbors does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values