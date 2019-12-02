(ns tensorflow.contrib.specs.python.specs-lib.Composition
  "A function composition.

  This simply composes its two argument functions when
  applied to a final argument via `of`.
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

(defn Composition 
  "A function composition.

  This simply composes its two argument functions when
  applied to a final argument via `of`.
  "
  [ f g ]
  (py/call-attr specs-lib "Composition"  f g ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
