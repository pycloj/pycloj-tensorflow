(ns tensorflow.python.keras.api.-v1.keras.applications.inception-resnet-v2
  "Inception-ResNet V2 model for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce inception-resnet-v2 (import-module "tensorflow.python.keras.api._v1.keras.applications.inception_resnet_v2"))

(defn InceptionResNetV2 
  ""
  [  ]
  (py/call-attr inception-resnet-v2 "InceptionResNetV2"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr inception-resnet-v2 "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr inception-resnet-v2 "preprocess_input"  ))
