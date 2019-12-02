(ns tensorflow.contrib.keras.api.keras.applications
  "Keras Applications are canned architectures with pre-trained weights."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce applications (import-module "tensorflow.contrib.keras.api.keras.applications"))

(defn InceptionV3 
  ""
  [  ]
  (py/call-attr applications "InceptionV3"  ))

(defn MobileNet 
  ""
  [  ]
  (py/call-attr applications "MobileNet"  ))

(defn ResNet50 
  ""
  [  ]
  (py/call-attr applications "ResNet50"  ))

(defn VGG16 
  ""
  [  ]
  (py/call-attr applications "VGG16"  ))

(defn VGG19 
  ""
  [  ]
  (py/call-attr applications "VGG19"  ))

(defn Xception 
  ""
  [  ]
  (py/call-attr applications "Xception"  ))
