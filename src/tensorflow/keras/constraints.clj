(ns tensorflow.python.keras.api.-v1.keras.constraints
  "Constraints: functions that impose constraints on weight values.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "tensorflow.python.keras.api._v1.keras.constraints"))

(defn deserialize 
  ""
  [ config custom_objects ]
  (py/call-attr constraints "deserialize"  config custom_objects ))

(defn get 
  ""
  [ identifier ]
  (py/call-attr constraints "get"  identifier ))

(defn serialize 
  ""
  [ constraint ]
  (py/call-attr constraints "serialize"  constraint ))
