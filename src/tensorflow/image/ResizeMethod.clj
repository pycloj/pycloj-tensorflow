(ns tensorflow.image.ResizeMethod
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "tensorflow.image"))

(defn ResizeMethod 
  ""
  [  ]
  (py/call-attr image "ResizeMethod"  ))
