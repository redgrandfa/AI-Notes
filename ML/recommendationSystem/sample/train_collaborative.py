# from load_data import data_forCollaborative
from load_data import data
from load_data import df
from load_data import test
# from load_data import test
# from load_data import test
# from load_data import test
# from load_data import test

x_train = df[['userid', 'itemid']]
y_train = df[['rating']]

x_test = test[['userid', 'itemid']]
y_test = test['rating']

data_forCollaborative = data

from surprise import KNNWithMeans
from surprise import SVD

print('---Tuning the Algorithm Parameters (GridSearchCV)---')
from surprise.model_selection import GridSearchCV
# # 矩陣分解演算法有
# # SVD(Netflix prize competition)
# # PCA and its variations, NMF,
# # Autoencoders(Neural Networks)

# print('------------KNN algorithm------------') #屬memory-based

# sim_options_KNN = {
#     "name": ["msd", "cosine"],
#     "min_support": [3, 4, 5], 
#     "user_based": [False, True],
# }
# param_grid_KNN = {"sim_options": sim_options_KNN}
# gs_KNN = GridSearchCV(KNNWithMeans, param_grid_KNN, measures=["rmse"], cv=3) #, "mae"
# gs_KNN.fit(data_forCollaborative)
# # gs_KNN.fit(x_train, y_train)
# print(gs_KNN.best_score["rmse"])
# print(gs_KNN.best_params["rmse"])


# print('------------SVD algorithm------------') 
# param_grid_SVD = {
#     "n_epochs": [5, 10],
#     "lr_all": [0.002, 0.005],
#     "reg_all": [0.4, 0.6]
# }
# gs_SVD = GridSearchCV(SVD, param_grid_SVD, measures=["rmse"], cv=3) #, "mae"
# gs_SVD.fit(data_forCollaborative)
# # gs_SVD.fit(x_train,y_train)
# print(gs_SVD.best_score["rmse"])
# print(gs_SVD.best_params["rmse"])



# ===============
# https://surprise.readthedocs.io/en/stable/trainset.html
# trainingSet = data_forCollaborative.build_full_trainset()
# print(trainingSet) #<surprise.trainset.Trainset
# Dataset.folds() method or the DatasetAutoFolds.build_full_trainset() method.
# You can think of a Dataset as the raw data, and Trainsets as higher-level data where useful methods are defined
# a Dataset may be comprised of multiple Trainsets (e.g. when doing cross validation).


# sim_options_KNN = gs_KNN.best_params["rmse"]
sim_options_KNN =  {'name': 'msd', 'min_support': 3, 'user_based': False}
algo_KNN = KNNWithMeans(sim_options= sim_options_KNN )
# algo_KNN.fit(trainingSet)
# algo_KNN.fit(x_train,y_train)# 參數不符

# sim_options_SVD = gs_SVD.best_params["rmse"]
sim_options_SVD = {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}
algo_SVD = SVD(sim_options=sim_options_SVD)
# algo_SVD.fit(trainingSet)
algo_SVD.fit(x_train,y_train)# 參數不符

prediction_KNN = algo_KNN.predict(x_test)
print(prediction_KNN.est) 
prediction_SVD = algo_SVD.predict(x_test)
print(prediction_SVD.est) 

# pred1 = model1.predict(x_test)
# rmse = np.sqrt(np.mean((pred1 - y_test.to_numpy())**2))


def compute_single_prediction(userid, itemid, similarity_mtx, utility):
    user_rating = utility.iloc[:,userid-1]
    item_similarity = similarity_mtx[itemid-1]
    numerate = np.dot(user_rating, item_similarity)
    denom = item_similarity[user_rating > 0].sum()
            
    if denom == 0 or numerate == 0:
        return user_rating[user_rating>0].mean()
    
    return numerate / denom

def compute_all_prediction(test_set, pred_func, similarity_mtx, utility, **kwargs):
    pred = []
    for data in test_set:
        res = pred_func(userid = data[0], #第一欄
                        itemid = data[1], #第二欄
                        similarity_mtx = similarity_mtx, 
                        utility = utility, 
                        **kwargs)
        pred.append(res)
    return pred


# #====混合

# chart_val = []
# w = np.linspace(0,1,21)

# for i in w:
#     pred4 = pred1*i + pred2*(1-i)
# 		rmse = np.sqrt(np.mean((pred4 - y_test.to_numpy())**2)) #手刻rmse誤差
#     chart_val.append([i, rmse])

# chart_val_np = np.array(chart_val)

# import matplotlib.pyplot as plt
# plt.plot(chart_val_np[:, 0], chart_val_np[:,1])