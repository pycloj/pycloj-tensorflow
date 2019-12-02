(ns tensorflow.contrib.keras.api.keras.applications.resnet50
  "ResNet50 Keras application."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce resnet50 (import-module "tensorflow.contrib.keras.api.keras.applications.resnet50"))

(defn ResNet50 
  ""
  [  ]
  (py/call-attr resnet50 "ResNet50"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr resnet50 "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr resnet50 "preprocess_input"  ))
