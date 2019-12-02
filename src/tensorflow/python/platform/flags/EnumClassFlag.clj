(ns tensorflow.python.platform.flags.EnumClassFlag
  "Basic enum flag; its value is an enum class's member."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn EnumClassFlag 
  "Basic enum flag; its value is an enum class's member."
  [ name default help enum_class short_name ]
  (py/call-attr flags "EnumClassFlag"  name default help enum_class short_name ))

(defn flag-type 
  "Returns a str that describes the type of the flag.

    NOTE: we use strings, and not the types.*Type constants because
    our flags can have more exotic types, e.g., 'comma separated list
    of strings', 'whitespace separated list of strings', etc.
    "
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "Parses string and sets flag value.

    Args:
      argument: str or the correct flag value type, argument to be parsed.
    "
  [ self argument ]
  (py/call-attr self "parse"  self argument ))

(defn serialize 
  "Serializes the flag."
  [ self  ]
  (py/call-attr self "serialize"  self  ))

(defn unparse 
  ""
  [ self  ]
  (py/call-attr self "unparse"  self  ))

(defn value 
  ""
  [ self ]
    (py/call-attr self "value"))
