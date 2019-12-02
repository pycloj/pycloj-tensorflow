(ns tensorflow.app
  "Generic entry point script.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce app (import-module "tensorflow.app"))

(defn run 
  "Runs the program with an optional 'main' function and 'argv' list."
  [ main argv ]
  (py/call-attr app "run"  main argv ))
