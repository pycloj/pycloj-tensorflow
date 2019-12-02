(ns tensorflow.contrib.labeled-tensor.python.ops.core.AxisOrderError
  "Error class for cases where there is no valid axis order."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce core (import-module "tensorflow.contrib.labeled_tensor.python.ops.core"))
