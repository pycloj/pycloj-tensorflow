(ns tensorflow.contrib.keras.api.keras.applications.inception-v3
  "Inception V3 Keras application."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce inception-v3 (import-module "tensorflow.contrib.keras.api.keras.applications.inception_v3"))

(defn InceptionV3 
  ""
  [  ]
  (py/call-attr inception-v3 "InceptionV3"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr inception-v3 "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr inception-v3 "preprocess_input"  ))
