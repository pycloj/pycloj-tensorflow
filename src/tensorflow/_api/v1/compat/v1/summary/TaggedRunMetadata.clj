(ns tensorflow.-api.v1.compat.v1.summary.TaggedRunMetadata
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summary (import-module "tensorflow._api.v1.compat.v1.summary"))
