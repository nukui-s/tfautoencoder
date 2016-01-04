"""Simple demo code"""

import numpy as np
import tensorflow as tf

from tfautoencoder import TFAutoEncoder
from tfautoencoder import TFStackedAutoEncoder

#input: 20 dim 5000 data
data = np.arange(100000).reshape(5000, 20)

#basic autoencoder 20 dim -> 5 dim
ae = TFAutoEncoder(hidden_dim=5)
ae.fit(data)
encoded = ae.encode(data)
reconstructed = ae.reconstruct(data)

#stacked autoencoder 20 dim -> 10 dim -> 3dim
sae = TFStackedAutoEncoder(layer_units=[20, 10, 3])
sae.fit(data)
encoded = sae.encode(data)
