(ns tensorflow.python.platform.flags.UnrecognizedFlagError
  "Raised when a flag is unrecognized.

  Attributes:
    flagname: str, the name of the unrecognized flag.
    flagvalue: The value of the flag, empty if the flag is not defined.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn UnrecognizedFlagError 
  "Raised when a flag is unrecognized.

  Attributes:
    flagname: str, the name of the unrecognized flag.
    flagvalue: The value of the flag, empty if the flag is not defined.
  "
  [flagname & {:keys [flagvalue suggestions]
                       :or {suggestions None}} ]
    (py/call-attr-kw flags "UnrecognizedFlagError" [flagname] {:flagvalue flagvalue :suggestions suggestions }))
