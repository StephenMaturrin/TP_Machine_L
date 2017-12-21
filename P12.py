import numpy as np

np.set_printoptions(threshold=np.nan)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


warnings.filterwarnings("ignore")


def clean_data(data):
    data = data.values

    toNan = lambda t: np.nan if t == "?" else t
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
    data = data[~np.isnan(data).any(axis=1)]

    res = data[:, [6]]
    data = data[:, [1, 2, 3, 4, 5]]

    arr = np.array(data)
    return data, res


data = pd.read_csv('./credit.data', sep='\t')
X, res = clean_data(data)


scaler = preprocessing.StandardScaler().fit(X)
scalerm = MinMaxScaler().fit(X)
X_scalerm = scalerm.transform(X)
X_scaled = preprocessing.scale(X)

pca = PCA(n_components=3)

xr_1 = pca.fit(X_scaled).transform(X_scaled)

print("PCA 70 %")
print(pca.explained_variance_ratio_)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, res)

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
features	=np.arange(0,	X.shape[1])
print("features")
print(features)
padding	=	np.arange(X.size/len(X))	+	0.5
plt.barh(padding,	importances[sorted_idx],	align='center')
plt.yticks(padding,	features[sorted_idx])
plt.xlabel("Relative	Importance")
plt.title("Variable	Importance")
plt.show()


X = X




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

