(ns tensorflow.contrib.util.loader
  "Utilities for loading op libraries.

@@load_op_library
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce loader (import-module "tensorflow.contrib.util.loader"))

(defn load-op-library 
  "Loads a contrib op library from the given path.

  NOTE(mrry): On Windows, we currently assume that some contrib op
  libraries are statically linked into the main TensorFlow Python
  extension DLL - use dynamically linked ops if the .so is present.

  Args:
    path: An absolute path to a shared object file.

  Returns:
    A Python module containing the Python wrappers for Ops defined in the
    plugin.
  "
  [ path ]
  (py/call-attr loader "load_op_library"  path ))
