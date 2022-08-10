from data_loader import UnsupervisedDataLoader
from kmean import kmean, kPlotter 


# It seems that although the iris data has 3 different class kmeans struggles to converge to 3
loader = UnsupervisedDataLoader("iris.data", columns_to_use = [0, 1, 2, 3])
#loader = UnsupervisedDataLoader("sample_data.csv", None)
x = loader.get_data()

ml = kmean(x, 3)
means, data = ml.fit()

kPlotter(data, means)