(ns tensorflow.python.keras.api.-v1.keras.applications.resnet-v2
  "ResNet v2 models for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce resnet-v2 (import-module "tensorflow.python.keras.api._v1.keras.applications.resnet_v2"))

(defn ResNet101V2 
  ""
  [  ]
  (py/call-attr resnet-v2 "ResNet101V2"  ))

(defn ResNet152V2 
  ""
  [  ]
  (py/call-attr resnet-v2 "ResNet152V2"  ))

(defn ResNet50V2 
  ""
  [  ]
  (py/call-attr resnet-v2 "ResNet50V2"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr resnet-v2 "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr resnet-v2 "preprocess_input"  ))
