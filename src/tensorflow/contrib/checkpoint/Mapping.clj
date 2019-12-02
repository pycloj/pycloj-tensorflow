(ns tensorflow.contrib.checkpoint.Mapping
  "An append-only trackable mapping data structure with string keys.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), named based on its keys.

  Note that once a key has been added, it may not be deleted or replaced. If
  names may not be unique, see `tf.contrib.checkpoint.UniqueNameTracker`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce checkpoint (import-module "tensorflow.contrib.checkpoint"))

(defn Mapping 
  "An append-only trackable mapping data structure with string keys.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), named based on its keys.

  Note that once a key has been added, it may not be deleted or replaced. If
  names may not be unique, see `tf.contrib.checkpoint.UniqueNameTracker`.
  "
  [  ]
  (py/call-attr checkpoint "Mapping"  ))

(defn get 
  "D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."
  [ self key default ]
  (py/call-attr self "get"  self key default ))

(defn items 
  "D.items() -> a set-like object providing a view on D's items"
  [ self  ]
  (py/call-attr self "items"  self  ))

(defn keys 
  "D.keys() -> a set-like object providing a view on D's keys"
  [ self  ]
  (py/call-attr self "keys"  self  ))

(defn layers 
  ""
  [ self ]
    (py/call-attr self "layers"))

(defn losses 
  "Aggregate losses from any `Layer` instances."
  [ self ]
    (py/call-attr self "losses"))

(defn non-trainable-variables 
  ""
  [ self ]
    (py/call-attr self "non_trainable_variables"))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr self "non_trainable_weights"))

(defn trainable 
  ""
  [ self ]
    (py/call-attr self "trainable"))

(defn trainable-variables 
  ""
  [ self ]
    (py/call-attr self "trainable_variables"))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr self "trainable_weights"))

(defn update 
  ""
  [ self  ]
  (py/call-attr self "update"  self  ))

(defn updates 
  "Aggregate updates from any `Layer` instances."
  [ self ]
    (py/call-attr self "updates"))

(defn values 
  "D.values() -> an object providing a view on D's values"
  [ self  ]
  (py/call-attr self "values"  self  ))

(defn variables 
  ""
  [ self ]
    (py/call-attr self "variables"))

(defn weights 
  ""
  [ self ]
    (py/call-attr self "weights"))
