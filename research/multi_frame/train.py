import pickle
from sklearn.neighbors import KNeighborsClassifier

from dataset import Dataset
from options import Options

d = Dataset("data/frankmocap/ratio/")

score = d.train(KNeighborsClassifier(n_neighbors=1), 0.8)
print(score)

pickle.dump(d, open("training/weights/dataset_obj.pickle", "wb"))
