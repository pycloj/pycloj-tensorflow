(ns tensorflow.-api.v1.compat.v1.ragged.RaggedTensorValue
  "Represents the value of a `RaggedTensor`.

  Warning: `RaggedTensorValue` should only be used in graph mode; in
  eager mode, the `tf.RaggedTensor` class contains its value directly.

  See `tf.RaggedTensor` for a description of ragged tensors.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ragged (import-module "tensorflow._api.v1.compat.v1.ragged"))

(defn RaggedTensorValue 
  "Represents the value of a `RaggedTensor`.

  Warning: `RaggedTensorValue` should only be used in graph mode; in
  eager mode, the `tf.RaggedTensor` class contains its value directly.

  See `tf.RaggedTensor` for a description of ragged tensors.
  "
  [ values row_splits ]
  (py/call-attr ragged "RaggedTensorValue"  values row_splits ))

(defn dtype 
  "The numpy dtype of values in this tensor."
  [ self ]
    (py/call-attr self "dtype"))

(defn flat-values 
  "The innermost `values` array for this ragged tensor value."
  [ self ]
    (py/call-attr self "flat_values"))

(defn nested-row-splits 
  "The row_splits for all ragged dimensions in this ragged tensor value."
  [ self ]
    (py/call-attr self "nested_row_splits"))

(defn ragged-rank 
  "The number of ragged dimensions in this ragged tensor value."
  [ self ]
    (py/call-attr self "ragged_rank"))

(defn row-splits 
  "The split indices for the ragged tensor value."
  [ self ]
    (py/call-attr self "row_splits"))

(defn shape 
  "A tuple indicating the shape of this RaggedTensorValue."
  [ self ]
    (py/call-attr self "shape"))

(defn to-list 
  "Returns this ragged tensor value as a nested Python list."
  [ self  ]
  (py/call-attr self "to_list"  self  ))

(defn values 
  "The concatenated values for all rows in this tensor."
  [ self ]
    (py/call-attr self "values"))
