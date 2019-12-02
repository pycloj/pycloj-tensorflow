(ns tensorflow.contrib.distribute.UpdateContext
  "Context manager when you are in `update()` or `update_non_slot()`."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.contrib.distribute"))

(defn UpdateContext 
  "Context manager when you are in `update()` or `update_non_slot()`."
  [ device ]
  (py/call-attr distribute "UpdateContext"  device ))
