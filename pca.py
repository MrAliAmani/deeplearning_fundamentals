# TODO input_data, run
#  compare tensorboard visualization to autoencoder


import input_data
from sklearn import decomposition
from matplotlib import pyplot as plt


mnist = input_data.read_datasets("data/", one_hot=False)
pca = decomposition.PCA(n_components=2)
pca.fit(mnist.train.images)
pca_codes = pca.transform(mnist.test.images)

pca_recon = pca.inverse_transform(pca_codes[:1])
plt.imshow(pca_recon[0].reshape((28, 28)), cmap=plt.cm.gray)
