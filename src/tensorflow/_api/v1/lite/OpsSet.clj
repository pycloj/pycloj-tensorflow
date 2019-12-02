(ns tensorflow.-api.v1.lite.OpsSet
  "Enum class defining the sets of ops available to generate TFLite models.

  WARNING: Experimental interface, subject to change.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lite (import-module "tensorflow._api.v1.lite"))
(defn OpsSet 
  "Enum class defining the sets of ops available to generate TFLite models.

  WARNING: Experimental interface, subject to change.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw lite "OpsSet" [value names module qualname type] {:start start }))
