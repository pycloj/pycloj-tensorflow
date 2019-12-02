(ns tensorflow.contrib.keras.api.keras.datasets.boston-housing
  "Boston housing price regression dataset."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce boston-housing (import-module "tensorflow.contrib.keras.api.keras.datasets.boston_housing"))

(defn load-data 
  "Loads the Boston Housing dataset.

  Arguments:
      path: path where to cache the dataset locally
          (relative to ~/.keras/datasets).
      test_split: fraction of the data to reserve as test set.
      seed: Random seed for shuffling the data
          before computing the test split.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  "
  [ & {:keys [path test_split seed]} ]
   (py/call-attr-kw boston-housing "load_data" [] {:path path :test_split test_split :seed seed }))
