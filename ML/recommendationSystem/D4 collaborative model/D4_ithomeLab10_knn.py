# k-Nearest Neighbors
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
# 其中 n_neighbors = 6，就是選出6項最相似的商品。
# 這語法 是tuple?

#NearestNeighbors does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values


# 梯度下降法(Stochastic Gradient Descent, SGD)：這是神經網路求解的方式，分別對qi、pu作偏微分，逐步調整權重，直到損失函數收斂(converge)為止。
# 交替最小平方法(Alternating Least Squares, ALS)：先固定pu，再對qi作偏微分，之後，再固定qi，再對pu作偏微分，如此交替，直到收斂為止。
# ALS方法計算比較快，通常，會使用ALS。


