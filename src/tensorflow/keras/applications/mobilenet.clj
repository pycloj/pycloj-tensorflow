(ns tensorflow.python.keras.api.-v1.keras.applications.mobilenet
  "MobileNet v1 models for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mobilenet (import-module "tensorflow.python.keras.api._v1.keras.applications.mobilenet"))

(defn MobileNet 
  ""
  [  ]
  (py/call-attr mobilenet "MobileNet"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr mobilenet "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr mobilenet "preprocess_input"  ))
