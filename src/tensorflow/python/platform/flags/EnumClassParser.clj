(ns tensorflow.python.platform.flags.EnumClassParser
  "Parser of an Enum class member."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn EnumClassParser 
  "Parser of an Enum class member."
  [  ]
  (py/call-attr flags "EnumClassParser"  ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "Determines validity of argument and returns the correct element of enum.

    Args:
      argument: str or Enum class member, the supplied flag value.

    Returns:
      The first matching Enum class member in Enum class.

    Raises:
      ValueError: Raised when argument didn't match anything in enum.
    "
  [ self argument ]
  (py/call-attr self "parse"  self argument ))
