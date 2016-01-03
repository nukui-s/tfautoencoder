"""Simple demo code"""

import numpy as np
import tensorflow as tf

from autoencoder import TFAutoEncoder
from stackedautoencoder import TFStackedAutoEncoder

data = np.arange(100000).reshape(5000, 20)

#20 dim -> 5 dim
ae = TFAutoEncoder(hidden_dim=5)
ae.fit(data)
encoded = ae.encode(data)
reconstructed = ae.reconstruct(data)

#encode 20 dim -> 10 dim -> 3dim
sae = TFStackedAutoEncoder(layer_units=[20, 10, 3])
sae.fit(data)
encoded = sae.encode(data)
