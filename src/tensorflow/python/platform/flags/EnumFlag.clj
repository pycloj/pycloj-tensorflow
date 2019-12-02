(ns tensorflow.python.platform.flags.EnumFlag
  "Basic enum flag; its value can be any string from list of enum_values."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))
(defn EnumFlag 
  "Basic enum flag; its value can be any string from list of enum_values."
  [name default help enum_values short_name  & {:keys [case_sensitive]} ]
    (py/call-attr-kw flags "EnumFlag" [name default help enum_values short_name] {:case_sensitive case_sensitive }))

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
