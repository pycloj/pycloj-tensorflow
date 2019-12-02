(ns tensorflow.user-ops
  "Public API for tf.user_ops namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce user-ops (import-module "tensorflow.user_ops"))

(defn my-fact 
  "Example of overriding the generated code for an Op."
  [  ]
  (py/call-attr user-ops "my_fact"  ))
