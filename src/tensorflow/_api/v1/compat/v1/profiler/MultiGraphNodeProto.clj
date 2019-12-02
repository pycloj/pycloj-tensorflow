(ns tensorflow.-api.v1.compat.v1.profiler.MultiGraphNodeProto
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce profiler (import-module "tensorflow._api.v1.compat.v1.profiler"))
