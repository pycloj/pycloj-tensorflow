(ns tensorflow.python.keras.api.-v1.keras.applications.mobilenet-v2
  "MobileNet v2 models for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mobilenet-v2 (import-module "tensorflow.python.keras.api._v1.keras.applications.mobilenet_v2"))

(defn MobileNetV2 
  ""
  [  ]
  (py/call-attr mobilenet-v2 "MobileNetV2"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr mobilenet-v2 "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr mobilenet-v2 "preprocess_input"  ))
