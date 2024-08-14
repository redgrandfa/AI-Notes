import numpy as np
import pandas as pd

# ---- load data
# 下載electronics dataset #https://www.kaggle.com/saurav9786/amazon-product-reviews
df = pd.read_csv("recommendationSystem/data/ratings_Electronics.csv", delimiter = ',', names = ['userId', 'productId', 'rating', 'timestamp'])
df=df.loc[:19999, ] #文章只取2萬筆資料



# 檢查 have any negative values, null values、值範圍，是否需要Normalizing
print(df.tail())
print(df.info()) #每個欄的 名稱、non-null數量、 型別
print(df.describe()) #最大～最小值、四分位數、標準差

# ----EDA (Exploratory Data Analysis)
import matplotlib.pyplot as plt
import seaborn as sns #based on matplotlib，資料視覺化用的(It provides a high-level interface for drawing attractive and informative statistical graphics.)
#%matplotlib inline #这一句是IPython的魔法函数，可以通过命令行的语法形式来访问它们。
#可以在IPython编译器里直接使用，作用是内嵌画图，省略掉plt.show()这一步，直接显示图像。
#如果不加这一句的话，我们在画图结束之后需要加上plt.show()才可以显示图像。

print('----想看1~5分各出現幾次')  #analyzing the number of ratings vs ratings [1,2,3,4,5 stars] 
plt.figure(figsize=(10,6))
sns.countplot(x='rating', data=df) #??
plt.xlabel('Rating', fontsize=12)
plt.xlabel('Total Users', fontsize=12)
plt.title('Number of Each Rating', fontsize=15)
plt.show() #冒出一個新視窗顯示圖表 >> 看出 More than 5000 users gave the rating for the products as 5 stars.
# 如果label中文，可能出現錯誤：C:\Users\user\AppData\Local\Programs\Python\Python310\lib\tkinter\__init__.py:839: UserWarning: Glyph 27599 (\N{CJK UNIFIED IDEOGRAPH-6BCF}) missing from current font.
#   func(*args)


# analyze the distribution of the number of ratings and mean ratings recorded for each product.
tmp = df.groupby('productId')['rating']
df_product_grp=pd.DataFrame({
    'Number of Rating':tmp.count(), #各產品被評次數
    'Mean Rating':tmp.mean() #各產品被評分均值
})

df_product_grp = df_product_grp.sort_values('Number of Rating' , ascending=False)
print(df_product_grp)
# >> most of the ratings are between 0 and 200,
# most of the products have a mean rating of 5.

# observe the relationship between the Number of Ratings and Mean Rating => 兩者間的scatterplot散佈圖...沒提供語法


print('------訓練/測試資料------')
from surprise import Reader, Dataset

reader=Reader()
surprise_data=Dataset.load_from_df(df[['userId','productId','rating']], reader)  #少timestamp一欄
print(surprise_data)# <surprise.dataset.DatasetAutoFolds object>
# print(surprise_data.df) # Dataframe物件( userId   productId  rating)
# print(surprise_data.raw_ratings) # list of tuple(uid, iid, rating, timestamp)

from surprise.model_selection import train_test_split
trainset,testset = train_test_split(surprise_data, test_size=.3, random_state=10)
print(trainset) # <surprise.trainset.Trainset object>
print(testset[0:3]) # list of tuple ( uid , iid , rating)

print('------封裝調參過程------')
from surprise.model_selection import GridSearchCV
def find_best_model(algo , param_grid , data):
    gs = GridSearchCV(algo, param_grid, measures=["rmse"], cv=3)
    gs.fit(data)
    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"]) 
    print(gs.best_estimator["rmse"]) 
    return gs

print('------KNN(最近鄰------')
from surprise import KNNWithMeans
sim_options={
    "name":["pearson_baseline"], #"msd","cosine","pearson",
    "min_support":[4], #3,,5
    "user_based":[True], #
}
params={
    'k':[30, 46],#range(30,50,1), # The (max) number of neighbors to take into account. Default is 40.
    "sim_optionns":sim_options
}
clf=find_best_model(KNNWithMeans, params, surprise_data)#clf=classifier的缩写
#{'k': 30, 'sim_optionns': ...}  
# <surprise.prediction_algorithms.knns.KNNWithMeans object => 看型別可知 .best_estimator會直接得到一個最佳參數的algo
# knn=clf.best_estimator["rmse"]


print('------矩陣分解：SVD(奇異值分解)------')
from surprise import SVD
from surprise.model_selection.validation import cross_validate
# from surprise import accuracy

params={
    "n_epochs": [20], #5, 10, 15,
    "lr_all": [0.005], #0.002, 
    "reg_all": [0.4], #, 0.6
}
clf = find_best_model(SVD, params, surprise_data)
#1.3701966355698645
# {'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.4}
# <surprise.prediction_algorithms.matrix_factorization.SVD object



# print(' 這一段是簡述後面的過程?')
# svd=SVD() # 已確定參數的SVD演算法，做交叉驗證
# cross_validate(svd, surprise_data, measures=['RMSE','MAE'] , cv=5, verbose=True) 

# trainset=surprise_data.build_full_trainset() 
# resultant_matrix = svd.fit(trainset)

# df[df['userId']=='AKM1MP6P0OYPR'] #看真實值
# pred = svd.predict(uid='AKM1MP6P0OYPR',iid='0132793040',r_ui=None)
# print(pred.est) #看預測直
#------

print('最後：訓練→觀察誤差→若對誤差滿意→推薦')
svd= clf.best_estimator['rmse'] #從調好參的SVD繼續
class collab_filtering_based_recommender_model:
    def __init__(self, algo, trainset, testset, data) :
        self.algo = algo
        self.trainset = trainset
        self.testset = testset
        self.data = data #傳進來的是 'DatasetAutoFolds' object

    def fit_and_predict(self): 
        self.algo.fit(self.trainset) #會回傳自己
        # 想這樣寫但不能通
        # prediction = self.algo.predict(self.testset) # AlgoBase.predict() missing 1 required positional argument: 'iid'
        # return prediction.est
        
        SE = 0
        for data in self.testset: # list of tuple ( uid , iid , rating)
            prediction = self.algo.predict(data[0], data[1])
            SE += ((prediction.est-data[2])**2)
        
        rmse = np.sqrt( SE/len(self.testset) ) 
        return rmse


    def cross_validate(self):
        t= cross_validate(self.algo, self.data ,measures=['RMSE','MAE'] , cv=5, verbose=True)
        # print(t) #是個dictionary
        return np.mean(t['test_rmse'])

    def recommend(self, user_id , n):
        # 指定user => 針對 原始資料的所有商品 的預設評分排序
        allProductId = self.data.df['productId'].unique()
        # print(type(allProductId)) #'numpy.ndarray'
        # print(allProductId) 

        predictionList=[]
        idx=0
        for iid in allProductId:
            pred = self.algo.predict(user_id, iid)
            if(idx<6):
                print(pred)
            predictionList.append( pred )
            idx=idx+1

        print('---排序---')
        predictionList.sort(reverse=True , key = lambda pred: pred.est)
        result = predictionList[:n]
        for pred in result:
            print(pred)
            # print(f'{pred.uid}\t{pred.iid}\t{pred.est}')
        return result
    
        # class SVD 
            # |      pu(numpy array of size (n_users, n_factors)): The user factors (only
            # |          exists if ``fit()`` has been called)
            # |      qi(numpy array of size (n_items, n_factors)): The item factors (only
            # |          exists if ``fit()`` has been called)
            # |      bu(numpy array of size (n_users)): The user biases (only
            # |          exists if ``fit()`` has been called)
            # |      bi(numpy array of size (n_items)): The item biases (only
            # |          exists if ``fit()`` has been called)
            # print(self.algo.pu)
            # print(self.algo.qi)
            # print(self.algo.bu)
            # print(self.algo.bi)
            # print(self.algo.pu.dot(self.algo.qi.transpose()))



col_fil_svd=collab_filtering_based_recommender_model(svd, trainset,testset, surprise_data)

print('70%訓練，30%測試')
svd_rmse = col_fil_svd.fit_and_predict()
print(svd_rmse) #1.3624269244536127

print('交叉驗證 看誤差')
svd_cv_rmse = col_fil_svd.cross_validate()
print(svd_cv_rmse) #1.3621884964255158

print('指定user，推薦五個商品')
result_svd_user1 = col_fil_svd.recommend(user_id='ANTN61S4L7WG9', n=5)
result_svd_user2 = col_fil_svd.recommend(user_id='AYNAH993VDECT', n=5)
result_svd_user3 = col_fil_svd.recommend(user_id='A18YMFFW974QS', n=5)
# 結果 不同user對每個itme的預測評分都長一樣，應該有問題




#觀察
# SVD (Singular Value Decomposition) model has 
#   a test RMSE score of 1.362 and 
#   cross validation (CV) RMSE score of 1.366.
# reduced RMSE score compared to KNN which is 1.41


#  7824482 records.
# memory exceeded error.
# from 
#     df = pd.read_csv(myfile,sep='\t') # memory error
# to 
#     df = pd.read_csv(myfile,sep='\t',low_memory=False)

# reduced records to 20K


# Collaborative Filtering gives 
#     solid recommendation systems
#     requires fewer details than possible.



# can improve this recommendation engine using Deep Learning Techniques like adding RNN’s, CNN’s, 
# extra layers to train for much more accuracy. 
# And also Deep Hybrid Models Based Recommendation, 
# many neural building blocks can be integrated to formalize 
# more powerful and expressive models.