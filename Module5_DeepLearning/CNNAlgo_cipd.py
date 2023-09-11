import tensorflow as tf
from tensorflow._api.v1.keras import layers,dataset,module
import matplotlib as plt
import numpy as np
import pandas as pd

(X_train,y_train),(X_test,y_test)=dataset.cifar10.load_data()

print(X_train.shape)
