(ns tensorflow.python.keras.api.-v1.keras.datasets.cifar100
  "CIFAR100 small images classification dataset.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cifar100 (import-module "tensorflow.python.keras.api._v1.keras.datasets.cifar100"))

(defn load-data 
  "Loads CIFAR100 dataset.

  Arguments:
      label_mode: one of \"fine\", \"coarse\".

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  Raises:
      ValueError: in case of invalid `label_mode`.
  "
  [ & {:keys [label_mode]} ]
   (py/call-attr-kw cifar100 "load_data" [] {:label_mode label_mode }))
