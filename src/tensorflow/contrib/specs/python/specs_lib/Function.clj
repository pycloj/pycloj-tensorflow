(ns tensorflow.contrib.specs.python.specs-lib.Function
  "A composable function wrapper for a regular Python function.

  This overloads the regular __call__ operator for currying, i.e.,
  arguments passed to __call__ are remembered for the eventual
  function application.

  The final function application happens via the `of` method.
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

(defn Function 
  "A composable function wrapper for a regular Python function.

  This overloads the regular __call__ operator for currying, i.e.,
  arguments passed to __call__ are remembered for the eventual
  function application.

  The final function application happens via the `of` method.
  "
  [ f ]
  (py/call-attr specs-lib "Function"  f ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
