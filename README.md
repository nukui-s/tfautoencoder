# tfautoencoder
Auto Encoder on Tensorflow

This package provides:
* Basic AutoEncoder
* Denoising AutoEncoder
* Stacked (Denoising) AutoEncoder

#Requirements
* Python 2.7 or 3.3+
* TensorFlow >= 0.6.0

#Installation

Clone git repository and run setup.py as the following commands.
```
$git clone https://github.com/nukui-s/tfautoencoder.git
$cd tfautoencoder
$sudo python setup.py install
```

#Basic Usage
```python
import numpy as np
from tfautoencoder import TFAutoEncoder

#prepare data as numpy array or pandas DataFrame
data = np.random.rand(1000,50)

#If you want to encode 50 dim into 10 dimension
ae = TFAutoEncoder(hidden_dim=10)
#fitting weight and bias in AutoEncoder
ae.fit(data)
#encode data
encoded = ae.encode(data)
#reconstuct data
data2 = ae.reconstruct(data)

#define Denoising AutoEncoder
dae = TFAutoEncoder(hidden_dim=10, noising=True)

#Stacked AutoEncoder with 100, 50, 30 layer units
sae = TFAutoEncoder(layer_units=[100, 50, 30])

```
