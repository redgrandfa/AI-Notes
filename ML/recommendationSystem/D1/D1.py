import numpy as np
import pandas as pd

age = np.array( [ 17,17, 22,22 ] )
h = np.array( [ 1.60 , 1.80 ,  1.5 , 1.80 ] )
w = np.array( [44 , 50 , 50, 66] )
info = np.array( ['44' , '50' , '50', '66'] )

df = pd.DataFrame()
df['age'] = age
df['height'] = h
df['weight'] = w
### df['info'] = info

# df['bmi'] = w/h**2
df['bmi'] = df['weight']/df['height']**2

### def bag_words(x):
### # print(type(x))
### return(' '.join(x['info']))

### # df_keys = pd.DataFrame()
### # df_keys['keywords'] = df.apply(bag_words, axis = 1) # 迭代子 迭代 每一欄

### # cv = CountVectorizer() #from scikit-learn
### # cv_mx = cv.fit_transform(df_keys['keywords'])

# print(cv_mx)

print(df)
print('________')
# print(df.head(1))
# print(df.tail(1))
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(type(df.values))
# print(type(df.columns))
# print(df.columns)
# print(type(df.index))
# print(df.index)

# print(df.isnull().any())
# missing = df.columns[df.isnull().any()]
# print(missing)
# print(df[missing])


# print( df.isna() )
# print( df.duplicated() )
# print( df.apply( lambda x : x*2 ) )

# df2 =df.rename({'height':'hhh'} ,axis =1)
# df2 = df.drop('height' , axis = 1)
# print(df2)

# print(df.height)
# print(df['height'])
# print(df[ ['height','weight' ]  ])


# b1 = df['bmi'] >11 #⇒  布林陣列  #運算子 維度較小可以分配律
# # b2 = df[欄名].isin( someList ) #包含於
# print( b1 )
# #篩
# print(df[ b1  ] )

# print( df.loc[  2:3  , ['height' , 'bmi'] ])
# print( df.loc[  b1  , ['height' , 'bmi'] ])
# df.loc[  篩列用布林陣列 , 欄名或欄名List / 從左數起第幾欄 ]


# print(df.sort_values('bmi' ))
# print(df.sort_values('bmi' , ascending=True))
# print(df.sort_index( ascending = False ))
# print(df.sum())
# print(df.agg( 
# {
#     'height': 'sum', 
#     'bmi' : 'count' 
# }))

# print(df.groupby('age').agg( 
# {
#     'height': 'sum', 
#     'bmi' : 'count' 
# }))

# print(df.rank())
# print(df.rank(method = 'max' , ascending=False))
# print(df['bmi'].rank())
