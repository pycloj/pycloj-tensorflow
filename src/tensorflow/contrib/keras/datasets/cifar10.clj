(ns tensorflow.contrib.keras.api.keras.datasets.cifar10
  "CIFAR10 small image classification dataset."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cifar10 (import-module "tensorflow.contrib.keras.api.keras.datasets.cifar10"))

(defn load-data 
  "Loads CIFAR10 dataset.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  "
  [  ]
  (py/call-attr cifar10 "load_data"  ))
