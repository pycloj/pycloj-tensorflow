(ns tensorflow.python.keras.api.-v1.keras.applications.nasnet
  "NASNet-A models for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce nasnet (import-module "tensorflow.python.keras.api._v1.keras.applications.nasnet"))

(defn NASNetLarge 
  ""
  [  ]
  (py/call-attr nasnet "NASNetLarge"  ))

(defn NASNetMobile 
  ""
  [  ]
  (py/call-attr nasnet "NASNetMobile"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr nasnet "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr nasnet "preprocess_input"  ))
