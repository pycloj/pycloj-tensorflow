(ns tensorflow.contrib.keras.api.keras.preprocessing.image.Iterator
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "tensorflow.contrib.keras.api.keras.preprocessing.image"))

(defn Iterator 
  ""
  [ n batch_size shuffle seed ]
  (py/call-attr image "Iterator"  n batch_size shuffle seed ))

(defn next 
  "For python 2.x.

        # Returns
            The next batch.
        "
  [ self  ]
  (py/call-attr self "next"  self  ))

(defn on-epoch-end 
  ""
  [ self  ]
  (py/call-attr self "on_epoch_end"  self  ))

(defn reset 
  ""
  [ self  ]
  (py/call-attr self "reset"  self  ))
