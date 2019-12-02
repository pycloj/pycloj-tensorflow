(ns tensorflow.python.keras.api.-v1.keras.applications.vgg16
  "VGG16 model for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce vgg16 (import-module "tensorflow.python.keras.api._v1.keras.applications.vgg16"))

(defn VGG16 
  ""
  [  ]
  (py/call-attr vgg16 "VGG16"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr vgg16 "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr vgg16 "preprocess_input"  ))
