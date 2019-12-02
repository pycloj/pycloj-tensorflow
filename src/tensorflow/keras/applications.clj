(ns tensorflow.python.keras.api.-v1.keras.applications
  "Keras Applications are canned architectures with pre-trained weights.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce applications (import-module "tensorflow.python.keras.api._v1.keras.applications"))

(defn DenseNet121 
  ""
  [  ]
  (py/call-attr applications "DenseNet121"  ))

(defn DenseNet169 
  ""
  [  ]
  (py/call-attr applications "DenseNet169"  ))

(defn DenseNet201 
  ""
  [  ]
  (py/call-attr applications "DenseNet201"  ))

(defn InceptionResNetV2 
  ""
  [  ]
  (py/call-attr applications "InceptionResNetV2"  ))

(defn InceptionV3 
  ""
  [  ]
  (py/call-attr applications "InceptionV3"  ))

(defn MobileNet 
  ""
  [  ]
  (py/call-attr applications "MobileNet"  ))

(defn MobileNetV2 
  ""
  [  ]
  (py/call-attr applications "MobileNetV2"  ))

(defn NASNetLarge 
  ""
  [  ]
  (py/call-attr applications "NASNetLarge"  ))

(defn NASNetMobile 
  ""
  [  ]
  (py/call-attr applications "NASNetMobile"  ))

(defn ResNet101 
  ""
  [  ]
  (py/call-attr applications "ResNet101"  ))

(defn ResNet101V2 
  ""
  [  ]
  (py/call-attr applications "ResNet101V2"  ))

(defn ResNet152 
  ""
  [  ]
  (py/call-attr applications "ResNet152"  ))

(defn ResNet152V2 
  ""
  [  ]
  (py/call-attr applications "ResNet152V2"  ))

(defn ResNet50 
  ""
  [  ]
  (py/call-attr applications "ResNet50"  ))

(defn ResNet50V2 
  ""
  [  ]
  (py/call-attr applications "ResNet50V2"  ))

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
