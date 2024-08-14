# Dataset.load_builtin()
# Dataset.load_from_file()
# Dataset.load_from_df()

# line_format is a string that stores the order of the data with field names separated by a space, as in "item user rating".
# sep is used to specify separator between fields, such as ','.
# rating_scale is used to specify the rating scale. The default is (1, 5).
# skip_lines is used to indicate the number of lines to skip at the beginning of the file. The default is 0.


# load_data.py

import pandas as pd
from surprise import Dataset
from surprise import Reader

# This is the same data that was plotted for similarity earlier
# with one new user "E" who has rated only movie 1
ratings_dict = {
    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
}

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))

# Loads Pandas dataframe
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
# Loads the builtin Movielens-100k data
movielens = Dataset.load_builtin('ml-100k')


# recommender.py
from surprise import KNNWithMeans

sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)

# from load_data import data # 同一份.py檔就不需匯入
# from recommender import algo # 同一份.py檔就不需匯入
trainingSet = data.build_full_trainset()

algo.fit(trainingSet)
# Computing the cosine similarity matrix...
# Done computing similarity matrix.
# <surprise.prediction_algorithms.knns.KNNWithMeans object at 0x7f04fec56898>

prediction = algo.predict('E', 2)
print(prediction.est) # 4.15

# 以上 分三小塊：演算法、資料匯入到記憶體、演算法fit資料+估計
print('---------------------')
print('---Tuning the Algorithm Parameters (GridSearchCV)---')
# 演算法外面多包一個GridSearchCV，sim_options各參數變成陣列
# cv = cross-validation 演算法要交叉驗證 ⇒ ex.切五組資料，輪值到的那組當test set，其他當train set


from surprise import Dataset
from surprise.model_selection import GridSearchCV
data = Dataset.load_builtin("ml-100k")


print('---KNN---')
from surprise import KNNWithMeans
sim_options = {
    "name": ["msd", "cosine"], #Mean squared difference
    "min_support": [3, 4, 5], 
    "user_based": [False, True],
}
# name contains the similarity metric to use. Options are cosine, msd, pearson, or pearson_baseline. The default is msd.
# user_based is a boolean that tells whether the approach will be user-based or item-based. The default is True, which means the user-based approach will be used.
# min_support is the minimum number of common items needed between users to consider them for similarity. For the item-based approach, this corresponds to the minimum number of common users for two items.

param_grid = {"sim_options": sim_options}

gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
gs.fit(data)
#會跑次數 = 參數所有組合數 * cv組數

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
# 0.9434791128171457
# {'sim_options': {'name': 'msd', 'min_support': 3, 'user_based': False}}


print('---------------------')
print('---SVD algorithm---') 
from surprise import SVD

param_grid = {
    "n_epochs": [5, 10],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.4, 0.6]
}
# n_epochs 
#   is the number of iterations of SGD, 
#   which is basically an iterative method used in statistics to minimize a function.

# lr_all 
#   is the learning rate for all parameters, 
#   which is a parameter that decides 【how much the parameters are adjusted】 in each iteration.

# reg_all 
#   is the regularization term for all parameters, 
#   which is a penalty term added to prevent overfitting.
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
# 0.9642278631521038
# {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}
# 算得比knn快很多

# 演算法還有 SVD++ 、NMF