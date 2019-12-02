(ns tensorflow.python.keras.api.-v1.keras.applications.resnet50
  "Public API for tf.keras.applications.resnet50 namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce resnet50 (import-module "tensorflow.python.keras.api._v1.keras.applications.resnet50"))

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
