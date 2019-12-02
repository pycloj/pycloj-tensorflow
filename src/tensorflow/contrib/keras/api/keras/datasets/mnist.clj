(ns tensorflow.contrib.keras.api.keras.datasets.mnist
  "MNIST handwritten digits classification dataset."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mnist (import-module "tensorflow.contrib.keras.api.keras.datasets.mnist"))

(defn load-data 
  "Loads the MNIST dataset.

  Arguments:
      path: path where to cache the dataset locally
          (relative to ~/.keras/datasets).

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  License:
      Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
      which is a derivative work from original NIST datasets.
      MNIST dataset is made available under the terms of the
      [Creative Commons Attribution-Share Alike 3.0 license.](
      https://creativecommons.org/licenses/by-sa/3.0/)
  "
  [ & {:keys [path]} ]
   (py/call-attr-kw mnist "load_data" [] {:path path }))
