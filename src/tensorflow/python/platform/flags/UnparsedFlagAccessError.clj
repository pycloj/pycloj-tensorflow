(ns tensorflow.python.platform.flags.UnparsedFlagAccessError
  "Raised when accessing the flag value from unparsed FlagValues."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))
