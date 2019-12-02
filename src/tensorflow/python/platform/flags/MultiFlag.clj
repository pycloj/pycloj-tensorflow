(ns tensorflow.python.platform.flags.MultiFlag
  "A flag that can appear multiple time on the command-line.

  The value of such a flag is a list that contains the individual values
  from all the appearances of that flag on the command-line.

  See the __doc__ for Flag for most behavior of this class.  Only
  differences in behavior are described here:

    * The default value may be either a single value or an iterable of values.
      A single value is transformed into a single-item list of that value.

    * The value of the flag is always a list, even if the option was
      only supplied once, and even if the default value is a single
      value
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

(defn MultiFlag 
  "A flag that can appear multiple time on the command-line.

  The value of such a flag is a list that contains the individual values
  from all the appearances of that flag on the command-line.

  See the __doc__ for Flag for most behavior of this class.  Only
  differences in behavior are described here:

    * The default value may be either a single value or an iterable of values.
      A single value is transformed into a single-item list of that value.

    * The value of the flag is always a list, even if the option was
      only supplied once, and even if the default value is a single
      value
  "
  [  ]
  (py/call-attr flags "MultiFlag"  ))

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
