(ns tensorflow.contrib.specs.python.specs-lib.Composable
  "A composable function.

  This defines the operators common to all composable objects.
  Currently defines copmosition (via \"|\") and repeated application
  (via \"**\"), and maps addition (\"+\") and multiplication (\"*\")
  as \"(f + g)(x) = f(x) + g(x)\".
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs-lib (import-module "tensorflow.contrib.specs.python.specs_lib"))

(defn Composable 
  "A composable function.

  This defines the operators common to all composable objects.
  Currently defines copmosition (via \"|\") and repeated application
  (via \"**\"), and maps addition (\"+\") and multiplication (\"*\")
  as \"(f + g)(x) = f(x) + g(x)\".
  "
  [  ]
  (py/call-attr specs-lib "Composable"  ))
