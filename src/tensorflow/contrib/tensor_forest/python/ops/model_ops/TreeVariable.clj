(ns tensorflow.contrib.tensor-forest.python.ops.model-ops.TreeVariable
  "A tree model."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce model-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.model_ops"))

(defn TreeVariable 
  "A tree model."
  [ params tree_config stats_handle name container ]
  (py/call-attr model-ops "TreeVariable"  params tree_config stats_handle name container ))

(defn initializer 
  ""
  [ self ]
    (py/call-attr self "initializer"))

(defn is-initialized 
  ""
  [ self  ]
  (py/call-attr self "is_initialized"  self  ))

(defn resource-handle 
  "Returns the resource handle associated with this Resource."
  [ self ]
    (py/call-attr self "resource_handle"))
