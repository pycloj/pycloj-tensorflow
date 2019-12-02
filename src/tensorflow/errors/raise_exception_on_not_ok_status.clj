(ns tensorflow.errors.raise-exception-on-not-ok-status
  "Context manager to check for C API status."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce errors (import-module "tensorflow.errors"))

(defn raise-exception-on-not-ok-status 
  "Context manager to check for C API status."
  [  ]
  (py/call-attr errors "raise_exception_on_not_ok_status"  ))
