(ns tensorflow.-api.v1.compat.v1.errors
  "Exception types for TensorFlow errors.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce errors (import-module "tensorflow._api.v1.compat.v1.errors"))

(defn error-code-from-exception-type 
  ""
  [ cls ]
  (py/call-attr errors "error_code_from_exception_type"  cls ))

(defn exception-type-from-error-code 
  ""
  [ error_code ]
  (py/call-attr errors "exception_type_from_error_code"  error_code ))
