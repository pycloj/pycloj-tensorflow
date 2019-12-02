(ns tensorflow.python.platform.flags.FloatParser
  "Parser of floating point values.

  Parsed value may be bounded to a given upper and lower bound.
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

(defn FloatParser 
  "Parser of floating point values.

  Parsed value may be bounded to a given upper and lower bound.
  "
  [  ]
  (py/call-attr flags "FloatParser"  ))

(defn convert 
  "Returns the float value of argument."
  [ self argument ]
  (py/call-attr self "convert"  self argument ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn is-outside-bounds 
  "Returns whether the value is outside the bounds or not."
  [ self val ]
  (py/call-attr self "is_outside_bounds"  self val ))

(defn parse 
  "See base class."
  [ self argument ]
  (py/call-attr self "parse"  self argument ))
