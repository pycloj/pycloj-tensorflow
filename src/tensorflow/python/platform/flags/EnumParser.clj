(ns tensorflow.python.platform.flags.EnumParser
  "Parser of a string enum value (a string value from a given set)."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn EnumParser 
  "Parser of a string enum value (a string value from a given set)."
  [  ]
  (py/call-attr flags "EnumParser"  ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "Determines validity of argument and returns the correct element of enum.

    Args:
      argument: str, the supplied flag value.

    Returns:
      The first matching element from enum_values.

    Raises:
      ValueError: Raised when argument didn't match anything in enum.
    "
  [ self argument ]
  (py/call-attr self "parse"  self argument ))
