# Dataset.load_builtin()
# Dataset.load_from_file()
# Dataset.load_from_df()

# line_format is a string that stores the order of the data with field names separated by a space, as in "item user rating".
# sep is used to specify separator between fields, such as ','.
# rating_scale is used to specify the rating scale. The default is (1, 5).
# skip_lines is used to indicate the number of lines to skip at the beginning of the file. The default is 0.


import pandas as pd
from surprise import Dataset

# data = Dataset.load_builtin('ml-100k')
# print(type(data)) #<class 'surprise.dataset.DatasetAutoFolds'>
# print(data) #<surprise.dataset.DatasetAutoFolds object
# data_forContentBased = data
# data_forCollaborative = data



# load data
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

data = pd.read_csv(data_path+'u.data', delimiter = '\t', names = ['userid', 'itemid', 'rating', 'timestamp'])

# load data
user = pd.read_csv(data_path+'u.user', delimiter = '|', names = ['id', 'age', 'gender', 'occupation', 'zip'])[['id', 'age', 'gender', 'occupation']]
genre = pd.read_csv(data_path+'u.genre', delimiter = '|', names = ['genre', 'id']).genre.to_list()
occupation_col_names =  pd.read_csv(data_path+'u.occupation', delimiter = '|', names = ['occupation'])['occupation'].to_list()


# perform one-hot encoding on the user's occupation column, and label encoding on the gender column
from sklearn import preprocessing

user[occupation_col_names] = preprocessing.OneHotEncoder(sparse = False).fit_transform(user.occupation.to_numpy().reshape(-1,1))
user['gender'] = preprocessing.LabelEncoder().fit_transform(user.gender)
user = user.drop(['occupation'], axis = 1)

# notice that the genre is already in the one-hot encoding format in the movie dataset, 
# so we can simply load the movie data
item_col_names = ['movie id','movie title','release date','video release date','IMDb URL'] + genre
item = pd.read_csv(data_path+'u.item', delimiter = '|', names = item_col_names, encoding = 'latin1')[['movie id'] + genre]

# Next we merge the movie and user data with our train and test dataset
x_train_join = x_train.join(user.set_index('id'), on = 'userid').join(item.set_index('movie id'), on = 'itemid')
x_test_join = x_test.join(user.set_index('id'), on = 'userid').join(item.set_index('movie id'), on = 'itemid')

#join應該只是把資料組裝過去。 如果不需要
# print( x_train_join.head() )
# print(x_test_join.head() )