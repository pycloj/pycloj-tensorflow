(ns tensorflow.python.keras.api.-v1.keras.utils.Sequence
  "Base object for fitting to a sequence of data, such as a dataset.

  Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
  If you want to modify your dataset between epochs you may implement
  `on_epoch_end`.
  The method `__getitem__` should return a complete batch.

  Notes:

  `Sequence` are a safer way to do multiprocessing. This structure guarantees
  that the network will only train once
   on each sample per epoch which is not the case with generators.

  Examples:

  ```python
      from skimage.io import imread
      from skimage.transform import resize
      import numpy as np
      import math

      # Here, `x_set` is list of path to the images
      # and `y_set` are the associated classes.

      class CIFAR10Sequence(Sequence):

          def __init__(self, x_set, y_set, batch_size):
              self.x, self.y = x_set, y_set
              self.batch_size = batch_size

          def __len__(self):
              return math.ceil(len(self.x) / self.batch_size)

          def __getitem__(self, idx):
              batch_x = self.x[idx * self.batch_size:(idx + 1) *
              self.batch_size]
              batch_y = self.y[idx * self.batch_size:(idx + 1) *
              self.batch_size]

              return np.array([
                  resize(imread(file_name), (200, 200))
                     for file_name in batch_x]), np.array(batch_y)
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow.python.keras.api._v1.keras.utils"))

(defn Sequence 
  "Base object for fitting to a sequence of data, such as a dataset.

  Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
  If you want to modify your dataset between epochs you may implement
  `on_epoch_end`.
  The method `__getitem__` should return a complete batch.

  Notes:

  `Sequence` are a safer way to do multiprocessing. This structure guarantees
  that the network will only train once
   on each sample per epoch which is not the case with generators.

  Examples:

  ```python
      from skimage.io import imread
      from skimage.transform import resize
      import numpy as np
      import math

      # Here, `x_set` is list of path to the images
      # and `y_set` are the associated classes.

      class CIFAR10Sequence(Sequence):

          def __init__(self, x_set, y_set, batch_size):
              self.x, self.y = x_set, y_set
              self.batch_size = batch_size

          def __len__(self):
              return math.ceil(len(self.x) / self.batch_size)

          def __getitem__(self, idx):
              batch_x = self.x[idx * self.batch_size:(idx + 1) *
              self.batch_size]
              batch_y = self.y[idx * self.batch_size:(idx + 1) *
              self.batch_size]

              return np.array([
                  resize(imread(file_name), (200, 200))
                     for file_name in batch_x]), np.array(batch_y)
  ```
  "
  [  ]
  (py/call-attr utils "Sequence"  ))

(defn on-epoch-end 
  "Method called at the end of every epoch.
    "
  [ self  ]
  (py/call-attr self "on_epoch_end"  self  ))
