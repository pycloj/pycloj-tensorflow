(ns tensorflow.contrib.specs.python.Composition
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
(defonce python (import-module "tensorflow.contrib.specs.python"))

(defn Composition 
  "A function composition.

  This simply composes its two argument functions when
  applied to a final argument via `of`.
  "
  [ f g ]
  (py/call-attr specs "Composition"  f g ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
