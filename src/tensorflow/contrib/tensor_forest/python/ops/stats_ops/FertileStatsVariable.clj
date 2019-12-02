(ns tensorflow.contrib.tensor-forest.python.ops.stats-ops.FertileStatsVariable
  "A Fertile stats variable."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stats-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.stats_ops"))

(defn FertileStatsVariable 
  "A Fertile stats variable."
  [ params stats_config name container ]
  (py/call-attr stats-ops "FertileStatsVariable"  params stats_config name container ))

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
