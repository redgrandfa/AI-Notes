# pip install scikit-surprise
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# 載入內建 movielens-100k 資料集
# surprise.Dataset
data = Dataset.load_builtin('ml-100k') 
# Dataset ml-100k could not be found. Do you want to download it? [Y/n] 終端機互動

# 切分為訓練及測試資料 
trainset, testset = train_test_split(data, test_size=.25) #(25%用來測試)
# print(type(trainset)) #<class 'surprise.trainset.Trainset'>
# print(trainset) # surprise.trainset.Trainset object at 0x000001F65E8E8430>
# print(type(testset)) # <class 'list'>
# print(testset) #印出超長的List


# 使用 SVD 演算法
algo = SVD()
# print(type(algo)) # <class 'surprise.prediction_algorithms.matrix_factorization.SVD'>
# print(algo) #<surprise.prediction_algorithms.matrix_factorization.SVD object at 0x0000029658A5C730>

# 訓練
algo.fit(trainset)
# print(algo) #<surprise.prediction_algorithms.matrix_factorization.SVD object at 0x0000029658A5C730>

# 測試
predictions = algo.test(testset)
# print(type(predictions)) #<class 'list'>
# print(predictions) #印出類似如下，猜是稀疏矩陣 各格的估值
    # Prediction(
    #     uid='497', 
    #     iid='242', 
    #     r_ui=1.0, 
    #     est=3.371643595815592, 
    #     details={'was_impossible': False}),
    # Prediction(
    #     uid='328', 
    #     iid='380',
    #     r_ui=3.0,
    #     est=3.207159558235514, 
    #     details={'was_impossible': False}), 


# 計算 RMSE
# surprise.accuracy
accuracy.rmse(predictions)  #RMSE: 0.9402