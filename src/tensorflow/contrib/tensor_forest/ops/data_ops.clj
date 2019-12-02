(ns tensorflow.contrib.tensor-forest.python.ops.data-ops
  "Ops for preprocessing data."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.data_ops"))

(defn CastToFloat 
  ""
  [ tensor ]
  (py/call-attr data-ops "CastToFloat"  tensor ))

(defn GetColumnName 
  ""
  [ column_key col_num ]
  (py/call-attr data-ops "GetColumnName"  column_key col_num ))

(defn ParseDataTensorOrDict 
  "Return a tensor to use for input data.

  The incoming features can be a dict where keys are the string names of the
  columns, which we turn into a single 2-D tensor.

  Args:
    data: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for input to tensor_forest, a keys tensor for the
    tf.Examples if they exist, and a list of the type of each column
    (e.g. continuous float, categorical).
  "
  [ data ]
  (py/call-attr data-ops "ParseDataTensorOrDict"  data ))

(defn ParseLabelTensorOrDict 
  "Return a tensor to use for input labels to tensor_forest.

  The incoming targets can be a dict where keys are the string names of the
  columns, which we turn into a single 1-D tensor for classification or
  2-D tensor for regression.

  Converts sparse tensors to dense ones.

  Args:
    labels: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for labels/outputs.
  "
  [ labels ]
  (py/call-attr data-ops "ParseLabelTensorOrDict"  labels ))
