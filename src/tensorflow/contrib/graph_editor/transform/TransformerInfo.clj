(ns tensorflow.contrib.graph-editor.transform.TransformerInfo
  "\"Contains information about the result of a transform operation."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce transform (import-module "tensorflow.contrib.graph_editor.transform"))

(defn TransformerInfo 
  "\"Contains information about the result of a transform operation."
  [ info ]
  (py/call-attr transform "TransformerInfo"  info ))

(defn original 
  "Return the original op/tensor corresponding to the transformed one.

    Note that the output of this function mimics the hierarchy
    of its input argument `transformed`.
    Given an iterable, it returns a list. Given an operation or a tensor,
    it will return an operation or a tensor.

    Args:
      transformed: the transformed tensor/operation.
      missing_fn: function handling the case where the counterpart
        cannot be found. By default, None is returned.
    Returns:
      the original tensor/operation (or None if no match is found).
    "
  [ self transformed missing_fn ]
  (py/call-attr self "original"  self transformed missing_fn ))

(defn transformed 
  "Return the transformed op/tensor corresponding to the original one.

    Note that the output of this function mimics the hierarchy
    of its input argument `original`.
    Given an iterable, it returns a list. Given an operation or a tensor,
    it will return an operation or a tensor.

    Args:
      original: the original tensor/operation.
      missing_fn: function handling the case where the counterpart
        cannot be found. By default, None is returned.
    Returns:
      the transformed tensor/operation (or None if no match is found).
    "
  [ self original missing_fn ]
  (py/call-attr self "transformed"  self original missing_fn ))
