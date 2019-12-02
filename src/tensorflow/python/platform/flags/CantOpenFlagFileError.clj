(ns tensorflow.python.platform.flags.CantOpenFlagFileError
  "Raised when flagfile fails to open.

  E.g. the file doesn't exist, or has wrong permissions.
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
