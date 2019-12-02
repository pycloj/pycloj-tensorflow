(ns tensorflow.python.platform.flags.FlagNameConflictsWithMethodError
  "Raised when a flag name conflicts with FlagValues methods."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))
