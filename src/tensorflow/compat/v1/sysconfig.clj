(ns tensorflow.-api.v1.compat.v1.sysconfig
  "System configuration library.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sysconfig (import-module "tensorflow._api.v1.compat.v1.sysconfig"))

(defn get-compile-flags 
  "Get the compilation flags for custom operators.

  Returns:
    The compilation flags.
  "
  [  ]
  (py/call-attr sysconfig "get_compile_flags"  ))

(defn get-include 
  "Get the directory containing the TensorFlow C++ header files.

  Returns:
    The directory as string.
  "
  [  ]
  (py/call-attr sysconfig "get_include"  ))

(defn get-lib 
  "Get the directory containing the TensorFlow framework library.

  Returns:
    The directory as string.
  "
  [  ]
  (py/call-attr sysconfig "get_lib"  ))

(defn get-link-flags 
  "Get the link flags for custom operators.

  Returns:
    The link flags.
  "
  [  ]
  (py/call-attr sysconfig "get_link_flags"  ))
