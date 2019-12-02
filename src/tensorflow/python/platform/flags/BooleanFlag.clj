(ns tensorflow.python.platform.flags.BooleanFlag
  "Basic boolean flag.

  Boolean flags do not take any arguments, and their value is either
  True (1) or False (0).  The false value is specified on the command
  line by prepending the word 'no' to either the long or the short flag
  name.

  For example, if a Boolean flag was created whose long name was
  'update' and whose short name was 'x', then this flag could be
  explicitly unset through either --noupdate or --nox.
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

(defn BooleanFlag 
  "Basic boolean flag.

  Boolean flags do not take any arguments, and their value is either
  True (1) or False (0).  The false value is specified on the command
  line by prepending the word 'no' to either the long or the short flag
  name.

  For example, if a Boolean flag was created whose long name was
  'update' and whose short name was 'x', then this flag could be
  explicitly unset through either --noupdate or --nox.
  "
  [ name default help short_name ]
  (py/call-attr flags "BooleanFlag"  name default help short_name ))

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
