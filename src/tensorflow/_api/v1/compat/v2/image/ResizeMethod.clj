(ns tensorflow.-api.v1.compat.v2.image.ResizeMethod
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "tensorflow._api.v1.compat.v2.image"))

(defn ResizeMethod 
  ""
  [  ]
  (py/call-attr image "ResizeMethod"  ))
