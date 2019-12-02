(ns tensorflow.contrib.tensor-forest.python.ops.gen-tensor-forest-ops
  "Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_tensor_forest_ops.cc
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gen-tensor-forest-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.gen_tensor_forest_ops"))

(defn ReinterpretStringToFloat 
  "   Converts byte arrays represented by strings to 32-bit

     floating point numbers. The output numbers themselves are meaningless, and
     should only be used in == comparisons.

     input_data: A batch of string features as a 2-d tensor; `input_data[i][j]`
       gives the j-th feature of the i-th input.
     output_data: A tensor of the same shape as input_data but the values are
       float32.

  Args:
    input_data: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [ input_data name ]
  (py/call-attr gen-tensor-forest-ops "ReinterpretStringToFloat"  input_data name ))

(defn ScatterAddNdim 
  "  Add elements in deltas to mutable input according to indices.

    input: A N-dimensional float tensor to mutate.
    indices:= A 2-D int32 tensor. The size of dimension 0 is the number of
      deltas, the size of dimension 1 is the rank of the input.  `indices[i]`
      gives the coordinates of input that `deltas[i]` should add to.  If
      `indices[i]` does not fully specify a location (it has less indices than
      there are dimensions in `input`), it is assumed that they are start
      indices and that deltas contains enough values to fill in the remaining
      input dimensions.
    deltas: `deltas[i]` is the value to add to input at index indices[i][:]

  Args:
    input: A `Tensor` of type mutable `float32`.
    indices: A `Tensor` of type `int32`.
    deltas: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ input indices deltas name ]
  (py/call-attr gen-tensor-forest-ops "ScatterAddNdim"  input indices deltas name ))

(defn deprecated-endpoints 
  "Decorator for marking endpoints deprecated.

  This decorator does not print deprecation messages.
  TODO(annarev): eventually start printing deprecation warnings when
  @deprecation_endpoints decorator is added.

  Args:
    *args: Deprecated endpoint names.

  Returns:
    A function that takes symbol as an argument and adds
    _tf_deprecated_api_names to that symbol.
    _tf_deprecated_api_names would be set to a list of deprecated
    endpoint names for the symbol.
  "
  [  ]
  (py/call-attr gen-tensor-forest-ops "deprecated_endpoints"  ))

(defn reinterpret-string-to-float 
  "   Converts byte arrays represented by strings to 32-bit

     floating point numbers. The output numbers themselves are meaningless, and
     should only be used in == comparisons.

     input_data: A batch of string features as a 2-d tensor; `input_data[i][j]`
       gives the j-th feature of the i-th input.
     output_data: A tensor of the same shape as input_data but the values are
       float32.

  Args:
    input_data: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [ input_data name ]
  (py/call-attr gen-tensor-forest-ops "reinterpret_string_to_float"  input_data name ))

(defn reinterpret-string-to-float-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function reinterpret_string_to_float
  "
  [ input_data name ctx ]
  (py/call-attr gen-tensor-forest-ops "reinterpret_string_to_float_eager_fallback"  input_data name ctx ))

(defn scatter-add-ndim 
  "  Add elements in deltas to mutable input according to indices.

    input: A N-dimensional float tensor to mutate.
    indices:= A 2-D int32 tensor. The size of dimension 0 is the number of
      deltas, the size of dimension 1 is the rank of the input.  `indices[i]`
      gives the coordinates of input that `deltas[i]` should add to.  If
      `indices[i]` does not fully specify a location (it has less indices than
      there are dimensions in `input`), it is assumed that they are start
      indices and that deltas contains enough values to fill in the remaining
      input dimensions.
    deltas: `deltas[i]` is the value to add to input at index indices[i][:]

  Args:
    input: A `Tensor` of type mutable `float32`.
    indices: A `Tensor` of type `int32`.
    deltas: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ input indices deltas name ]
  (py/call-attr gen-tensor-forest-ops "scatter_add_ndim"  input indices deltas name ))

(defn scatter-add-ndim-eager-fallback 
  ""
  [ input indices deltas name ctx ]
  (py/call-attr gen-tensor-forest-ops "scatter_add_ndim_eager_fallback"  input indices deltas name ctx ))
