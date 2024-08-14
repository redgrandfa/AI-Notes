from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# 輸出如下，猜測是把某一群當testset 剩下當trainset
#                   Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
# RMSE (testset)    0.9337  0.9324  0.9445  0.9373  0.9306  0.9357  0.0049  
# MAE (testset)     0.7357  0.7364  0.7440  0.7382  0.7336  0.7376  0.0035  
# Fit time          1.06    1.07    1.06    1.11    1.14    1.09    0.03    
# Test time         0.14    0.14    0.13    0.13    0.15    0.14    0.01  