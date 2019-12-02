(ns tensorflow.python.keras.api.-v1.keras.applications.resnet
  "ResNet models for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce resnet (import-module "tensorflow.python.keras.api._v1.keras.applications.resnet"))

(defn ResNet101 
  ""
  [  ]
  (py/call-attr resnet "ResNet101"  ))

(defn ResNet152 
  ""
  [  ]
  (py/call-attr resnet "ResNet152"  ))

(defn ResNet50 
  ""
  [  ]
  (py/call-attr resnet "ResNet50"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr resnet "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr resnet "preprocess_input"  ))
