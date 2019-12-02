(ns tensorflow.python.keras.api.-v1.keras.regularizers
  "Built-in regularizers.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce regularizers (import-module "tensorflow.python.keras.api._v1.keras.regularizers"))

(defn deserialize 
  ""
  [ config custom_objects ]
  (py/call-attr regularizers "deserialize"  config custom_objects ))

(defn get 
  ""
  [ identifier ]
  (py/call-attr regularizers "get"  identifier ))

(defn l1 
  ""
  [ & {:keys [l]} ]
   (py/call-attr-kw regularizers "l1" [] {:l l }))

(defn l1-l2 
  ""
  [ & {:keys [l1 l2]} ]
   (py/call-attr-kw regularizers "l1_l2" [] {:l1 l1 :l2 l2 }))

(defn l2 
  ""
  [ & {:keys [l]} ]
   (py/call-attr-kw regularizers "l2" [] {:l l }))

(defn serialize 
  ""
  [ regularizer ]
  (py/call-attr regularizers "serialize"  regularizer ))
