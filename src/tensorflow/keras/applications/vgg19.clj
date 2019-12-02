(ns tensorflow.python.keras.api.-v1.keras.applications.vgg19
  "VGG19 model for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce vgg19 (import-module "tensorflow.python.keras.api._v1.keras.applications.vgg19"))

(defn VGG19 
  ""
  [  ]
  (py/call-attr vgg19 "VGG19"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr vgg19 "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr vgg19 "preprocess_input"  ))
