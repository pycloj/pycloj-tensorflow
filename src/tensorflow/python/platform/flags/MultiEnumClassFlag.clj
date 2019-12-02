(ns tensorflow.python.platform.flags.MultiEnumClassFlag
  "A multi_enum_class flag.

  See the __doc__ for MultiFlag for most behaviors of this class.  In addition,
  this class knows how to handle enum.Enum instances as values for this flag
  type.
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

(defn MultiEnumClassFlag 
  "A multi_enum_class flag.

  See the __doc__ for MultiFlag for most behaviors of this class.  In addition,
  this class knows how to handle enum.Enum instances as values for this flag
  type.
  "
  [ name default help_string enum_class ]
  (py/call-attr flags "MultiEnumClassFlag"  name default help_string enum_class ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "Parses one or more arguments with the installed parser.

    Args:
      arguments: a single argument or a list of arguments (typically a
        list of default values); a single argument is converted
        internally into a list containing one item.
    "
  [ self arguments ]
  (py/call-attr self "parse"  self arguments ))

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
