(ns tensorflow.python.platform.flags.BaseListParser
  "Base class for a parser of lists of strings.

  To extend, inherit from this class; from the subclass __init__, call

      BaseListParser.__init__(self, token, name)

  where token is a character used to tokenize, and name is a description
  of the separator.
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

(defn BaseListParser 
  "Base class for a parser of lists of strings.

  To extend, inherit from this class; from the subclass __init__, call

      BaseListParser.__init__(self, token, name)

  where token is a character used to tokenize, and name is a description
  of the separator.
  "
  [  ]
  (py/call-attr flags "BaseListParser"  ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "See base class."
  [ self argument ]
  (py/call-attr self "parse"  self argument ))
