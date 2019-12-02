(ns tensorflow.contrib.slim.python.slim.data.dataset.Dataset
  "Represents a Dataset specification."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dataset (import-module "tensorflow.contrib.slim.python.slim.data.dataset"))

(defn Dataset 
  "Represents a Dataset specification."
  [ data_sources reader decoder num_samples items_to_descriptions ]
  (py/call-attr dataset "Dataset"  data_sources reader decoder num_samples items_to_descriptions ))
