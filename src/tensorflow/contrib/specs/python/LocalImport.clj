(ns tensorflow.contrib.specs.python.LocalImport
  "A class that allows us to temporarily import something.

  Attributes:
      frame: the frame in which the context manager was invocked
      names: a dictionary containing the new bindings
      old: variable bindings that have been shadowed by the import
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python (import-module "tensorflow.contrib.specs.python"))

(defn LocalImport 
  "A class that allows us to temporarily import something.

  Attributes:
      frame: the frame in which the context manager was invocked
      names: a dictionary containing the new bindings
      old: variable bindings that have been shadowed by the import
  "
  [ names ]
  (py/call-attr specs "LocalImport"  names ))
