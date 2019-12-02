(ns tensorflow.python.platform.flags.WhitespaceSeparatedListParser
  "Parser for a whitespace-separated list of strings."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn WhitespaceSeparatedListParser 
  "Parser for a whitespace-separated list of strings."
  [  ]
  (py/call-attr flags "WhitespaceSeparatedListParser"  ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "Parses argument as whitespace-separated list of strings.

    It also parses argument as comma-separated list of strings if requested.

    Args:
      argument: string argument passed in the commandline.

    Returns:
      [str], the parsed flag value.
    "
  [ self argument ]
  (py/call-attr self "parse"  self argument ))
