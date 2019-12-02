(ns tensorflow.contrib.layers.python.layers.feature-column.InputLayer
  "An object-oriented version of `input_layer` that reuses variables."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce feature-column (import-module "tensorflow.contrib.layers.python.layers.feature_column"))

(defn InputLayer 
  "An object-oriented version of `input_layer` that reuses variables."
  [feature_columns weight_collections & {:keys [trainable cols_to_vars name create_scope_now]
                       :or {cols_to_vars None}} ]
    (py/call-attr-kw feature-column "InputLayer" [feature_columns weight_collections] {:trainable trainable :cols_to_vars cols_to_vars :name name :create_scope_now create_scope_now }))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn non-trainable-variables 
  ""
  [ self ]
    (py/call-attr self "non_trainable_variables"))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr self "non_trainable_weights"))

(defn trainable-variables 
  ""
  [ self ]
    (py/call-attr self "trainable_variables"))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr self "trainable_weights"))

(defn variables 
  ""
  [ self ]
    (py/call-attr self "variables"))

(defn weights 
  ""
  [ self ]
    (py/call-attr self "weights"))
