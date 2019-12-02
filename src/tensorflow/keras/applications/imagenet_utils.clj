(ns tensorflow.python.keras.api.-v1.keras.applications.imagenet-utils
  "Utilities for ImageNet data preprocessing & prediction decoding.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce imagenet-utils (import-module "tensorflow.python.keras.api._v1.keras.applications.imagenet_utils"))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr imagenet-utils "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr imagenet-utils "preprocess_input"  ))
