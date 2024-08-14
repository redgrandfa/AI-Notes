import pandas as pd
import numpy as np
# from copy import deepcopy

print('----load data')
data_path = 'recommendationSystem/data/ml-100k/'
train_path = data_path + 'u1.base'
test_path = data_path + 'u1.test'

# load train and test data
df = pd.read_csv(train_path, delimiter = '\t', names = ['userid', 'itemid', 'rating', 'timestamp'])
test = pd.read_csv(test_path, delimiter = '\t', names = ['userid', 'itemid', 'rating', 'timestamp'])

x_train = df[['userid', 'itemid']]
y_train = df[['rating']]

x_test = test[['userid', 'itemid']]
y_test = test['rating']


print('----前處理')
genre = pd.read_csv(data_path+'u.genre', delimiter = '|', names = ['genre', 'id']).genre.to_list()
print('genre：應該是 所有user表/item表中，genre欄可能的值')
print(genre)

occupation_col_names =  pd.read_csv(data_path+'u.occupation', delimiter = '|', names = ['occupation'])['occupation'].to_list()
print('occupation_col_names:應該是 所有user表，occupation欄可能的值')
print(occupation_col_names)

# perform one-hot encoding on the user's occupation column, and label encoding on the gender column
from sklearn import preprocessing
user = pd.read_csv(data_path+'u.user', delimiter = '|', names = ['id', 'age', 'gender', 'occupation', 'zip'])[['id', 'age', 'gender', 'occupation']]

print('----OneHotEncode前----')
print(user.head(3))
user[occupation_col_names] = preprocessing.OneHotEncoder(sparse = False).fit_transform(user.occupation.to_numpy().reshape(-1,1))
print('----OneHotEncode後----')
print(user.head(3)) # 欄位暴增

user['gender'] = preprocessing.LabelEncoder().fit_transform(user.gender)
user = user.drop(['occupation'], axis = 1)
print('----LabelEncode後----')
print(user.head(3)) #gender欄 M/F變 1/0

# notice that the genre is already in the one-hot encoding format in the movie dataset, 
# so we can simply load the movie data
item_col_names = ['movie id','movie title','release date','video release date','IMDb URL'] + genre
item = pd.read_csv(data_path+'u.item', delimiter = '|', names = item_col_names, encoding = 'latin1')[['movie id'] + genre]

# 把user/item資料join進來。 Next we merge the movie and user data with our train and test dataset
x_train = x_train.join(user.set_index('id'), on = 'userid').join(item.set_index('movie id'), on = 'itemid')
x_test = x_test.join(user.set_index('id'), on = 'userid').join(item.set_index('movie id'), on = 'itemid')
print('----join後----')
print( x_train.head() ) #可看見多出許多欄位


print('----model 1:XGBRegressor')
import xgboost as xgb
model1 = xgb.XGBRegressor(objective='reg:squarederror')
model1.fit(x_train, y_train) #和surprise不同

pred1 = model1.predict(x_test)
rmse = np.sqrt(np.mean((pred1 - y_test.to_numpy())**2))
print(f'content-based rmse = {rmse}')



print('----model 2: item-item collaborative filtering model ')
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
        res = pred_func(userid = data[0], 
                        itemid = data[1], 
                        similarity_mtx = similarity_mtx, 
                        utility = utility, 
                        **kwargs)
        pred.append(res)
    return pred


# construct the utility matrix
utility = df.pivot(index = 'itemid', columns = 'userid', values = 'rating') #item在表格左方，user在表格上方
utility = utility.fillna(0) #稀疏的填0

# calculate the similarity
from scipy.spatial.distance import pdist, squareform 
# pdist = Pairwise Distances
# squareform : Convert a vector-form distance vector to a square-form distance matrix

#cosine相似度
similarity_mtx = 1 - squareform(pdist(utility, 'cosine'))

pred2 = compute_all_prediction(test[['userid', 'itemid']].to_numpy(),
    compute_single_prediction,
    similarity_mtx,
    utility)

pred2 = np.array(pred2)
# 假的評分-真的評分(測資中)
rmse = np.sqrt(np.mean(
        (pred2 - y_test.to_numpy())**2
    ))
print(f'rmse of item-item collaborative filtering = {rmse}')




print('----hybrid proportion ')
chart_val = []

w = np.linspace(0,1,21)
# lowest = ( 0, 9999)
for i in w:
    pred4 = pred1*i + pred2*(1-i)
    rmse = np.sqrt(np.mean((pred4 - y_test.to_numpy())**2))
    chart_val.append([i, rmse])

chart_val_np = np.array(chart_val)
print(chart_val_np) #=> 選0.85

import matplotlib.pyplot as plt
plt.plot(chart_val_np[:, 0], chart_val_np[:,1]) #list切片的語法
plt.show()