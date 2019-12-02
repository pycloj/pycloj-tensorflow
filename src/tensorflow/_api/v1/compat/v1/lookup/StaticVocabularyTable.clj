(ns tensorflow.-api.v1.compat.v1.lookup.StaticVocabularyTable
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow._api.v1.compat.v1.lookup"))

(defn StaticVocabularyTable 
  ""
  [ initializer num_oov_buckets lookup_key_dtype name ]
  (py/call-attr lookup "StaticVocabularyTable"  initializer num_oov_buckets lookup_key_dtype name ))

(defn initializer 
  ""
  [ self ]
    (py/call-attr self "initializer"))

(defn key-dtype 
  "The table key dtype."
  [ self ]
    (py/call-attr self "key_dtype"))

(defn lookup 
  "Looks up `keys` in the table, outputs the corresponding values.

    It assigns out-of-vocabulary keys to buckets based in their hashes.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.

    Returns:
      A `SparseTensor` if keys are sparse, otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    "
  [ self keys name ]
  (py/call-attr self "lookup"  self keys name ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn resource-handle 
  ""
  [ self ]
    (py/call-attr self "resource_handle"))

(defn size 
  "Compute the number of elements in this table."
  [ self name ]
  (py/call-attr self "size"  self name ))

(defn value-dtype 
  "The table value dtype."
  [ self ]
    (py/call-attr self "value_dtype"))
