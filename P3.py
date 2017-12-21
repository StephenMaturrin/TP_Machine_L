import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import  matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances



def clean_data(data):
    data = data.values

    toNan = lambda t: 2 if t == "?" else t
    vfunc = np.vectorize(toNan)
    data = vfunc(data)

    toBin = lambda t: 0 if t == "-" else t
    vfunc = np.vectorize(toBin)
    data = vfunc(data)

    toBin = lambda t: 1 if t == "+" else t
    vfunc = np.vectorize(toBin)
    data = vfunc(data)

    data = data[:, [1, 2, 7, 10, 13, 14, 15]]
    data = data.astype(np.float)

    res = data[:, [6]]
    data = data[:, [1, 2, 3, 4, 5]]

    arr = np.array(data)
    return data, res



warnings.filterwarnings("ignore")

X = pd.read_csv('./credit.data', sep='\t')

X1 =X
X = X.values
col_cat = [0, 3, 4, 5, 6, 8, 9, 11, 12]
X_cat = np.copy(X[:, col_cat])
for col_id in range(len(col_cat)):
    unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)
    X_cat[:, col_id] = val_idx
imp_cat = Imputer(missing_values=0, strategy='most_frequent')
X_cat[:, range(5)] = imp_cat.fit_transform(X_cat[:, range(5)])



X_cat_bin	=	OneHotEncoder().fit_transform(X_cat).toarray()

# print(X_cat_bin)


col_num = [1, 2, 7, 10, 13, 14]
X_num = np.copy(X[:, col_num])
X_num[X_num == '?'] = np.nan
X_num = X_num.astype(float)
imp_num = Imputer(missing_values=np.nan, strategy='mean')
X_num = imp_num.fit_transform(X_num)

# print(X_num)

scaler = preprocessing.StandardScaler().fit(X_num)
X_scaled = preprocessing.scale(X_num)


concat =  np.column_stack((X_scaled,X_cat_bin))



XX, res = clean_data(X1)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(concat, res)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)


print(" Random forest feature_importances : ")
print(clf.feature_importances_)
importances=clf.feature_importances_
sorted_idx	=	np.argsort(importances)[::-1]
print("sorted_idx")
print(sorted_idx)
features	=np.arange(0,	concat.shape[1])
print("features")
print(features)
padding	=	np.arange(concat.size/len(concat))	+	0.5
plt.barh(padding,	importances[sorted_idx],	align='center')
plt.yticks(padding,	features[sorted_idx])
plt.xlabel("Relative	Importance")
plt.title("Variable	Importance")
plt.show()



pca = PCA(n_components=12)

X = pca.fit(concat).transform(concat)

print("PCA  %")
print(pca.explained_variance_ratio_)


clfs = {
    'RandomForest': RandomForestClassifier(n_estimators=50),
    'Baging':BaggingClassifier(n_estimators=50,bootstrap_features=True,warm_start=True).fit(X,res),
    'NearestNeighbors': KNeighborsClassifier(n_neighbors=5).fit(X, res) ,
    'Gaussian': GaussianNB().fit(X, res),
    'CART': tree.DecisionTreeClassifier(criterion="gini").fit(X, res),
    'ID3': tree.DecisionTreeClassifier(criterion="entropy").fit(X, res),
    'AdaBoostClassifier Decision Stump': AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), algorithm="SAMME",
                                                            n_estimators=50).fit(X, res),
    'Multi-Layer Perceptron': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10),
                                            random_state=1).fit(X, res)

}


kf = KFold(n_splits=10, shuffle=True, random_state=0)
for i in clfs:
    clf = clfs[i]
    cv_acc = cross_val_score(clf, X, res, cv=kf)  # pour le calcul de l'accuracy
    print("Accuracy for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_acc), np.std(cv_acc)))
