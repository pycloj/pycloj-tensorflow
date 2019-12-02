(ns tensorflow.contrib.keras.api.keras.applications.vgg16
  "VGG16 Keras application."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce vgg16 (import-module "tensorflow.contrib.keras.api.keras.applications.vgg16"))

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
