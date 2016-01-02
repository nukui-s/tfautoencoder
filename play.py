import numpy as np
import tensorflow as tf
from autoencoder import TFAutoEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

p1 = np.array([1,1,1,0,0,1,0,1,0,0,1])
p2 = np.array([0,1,0,1,0,0,1,1,0,1,0])
p3 = np.array([1,1,1,1,1,1,0,0,0,0,0])
patterns = [p1, p2, p3]
n = 10000
data = []
label = []
for _ in range(n):
    p = np.random.randint(100)%2
    d = patterns[p] + np.random.normal(size=[11], scale=0.5)
    data.append(d)
    label.append(p)
data = np.array(data)

#data = np.arange(12).reshape(4, 3)
ae = TFAutoEncoder(hidden_dim=2, lambda_w=0, num_cores=8, steps=1000,
                                continue_training=True, logdir="playlog", noising=True,
                                activate_func="relu", optimizer="adam", noise_stddev=0.1)
ae.fit(data)
R = ae.reconstruct(data)
F = ae.encode(data)

km = KMeans(2)
res = km.fit_predict(F)
nmi = normalized_mutual_info_score(res, label)
print(nmi)
