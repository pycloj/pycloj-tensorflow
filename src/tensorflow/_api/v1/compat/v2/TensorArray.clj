(ns tensorflow.-api.v1.compat.v2.TensorArray
  "Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.

  This class is meant to be used with dynamic iteration primitives such as
  `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  \"flow\" control flow dependencies.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))

(defn TensorArray 
  "Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.

  This class is meant to be used with dynamic iteration primitives such as
  `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  \"flow\" control flow dependencies.
  "
  [dtype size dynamic_size clear_after_read tensor_array_name handle flow & {:keys [infer_shape element_shape colocate_with_first_write_call name]
                       :or {element_shape None name None}} ]
    (py/call-attr-kw v2 "TensorArray" [dtype size dynamic_size clear_after_read tensor_array_name handle flow] {:infer_shape infer_shape :element_shape element_shape :colocate_with_first_write_call colocate_with_first_write_call :name name }))

(defn close 
  "Close the current TensorArray.

  **NOTE** The output of this function should be used.  If it is not, a warning will be logged.  To mark the output as used, call its .mark_used() method."
  [ self name ]
  (py/call-attr self "close"  self name ))

(defn concat 
  "Return the values in the TensorArray as a concatenated `Tensor`.

    All of the values must have been written, their ranks must match, and
    and their shapes must all match for all dimensions except the first.

    Args:
      name: A name for the operation (optional).

    Returns:
      All the tensors in the TensorArray concatenated into one tensor.
    "
  [ self name ]
  (py/call-attr self "concat"  self name ))

(defn dtype 
  "The data type of this TensorArray."
  [ self ]
    (py/call-attr self "dtype"))

(defn dynamic-size 
  "Python bool; if `True` the TensorArray can grow dynamically."
  [ self ]
    (py/call-attr self "dynamic_size"))

(defn element-shape 
  "The `tf.TensorShape` of elements in this TensorArray."
  [ self ]
    (py/call-attr self "element_shape"))

(defn flow 
  "The flow `Tensor` forcing ops leading to this TensorArray state."
  [ self ]
    (py/call-attr self "flow"))

(defn gather 
  "Return selected values in the TensorArray as a packed `Tensor`.

    All of selected values must have been written and their shapes
    must all match.

    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If
        the `TensorArray` is not dynamic, `max_value=size()`.
      name: A name for the operation (optional).

    Returns:
      The tensors in the `TensorArray` selected by `indices`, packed into one
      tensor.
    "
  [ self indices name ]
  (py/call-attr self "gather"  self indices name ))

(defn grad 
  ""
  [ self source flow name ]
  (py/call-attr self "grad"  self source flow name ))

(defn handle 
  "The reference to the TensorArray."
  [ self ]
    (py/call-attr self "handle"))

(defn identity 
  "Returns a TensorArray with the same content and properties.

    Returns:
      A new TensorArray object with flow that ensures the control dependencies
      from the contexts will become control dependencies for writes, reads, etc.
      Use this object all for subsequent operations.
    "
  [ self  ]
  (py/call-attr self "identity"  self  ))

(defn read 
  "Read the value at location `index` in the TensorArray.

    Args:
      index: 0-D.  int32 tensor with the index to read from.
      name: A name for the operation (optional).

    Returns:
      The tensor at index `index`.
    "
  [ self index name ]
  (py/call-attr self "read"  self index name ))

(defn scatter 
  "Scatter the values of a `Tensor` in specific indices of a `TensorArray`.

    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If
        the `TensorArray` is not dynamic, `max_value=size()`.
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unpack.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the scatter occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if the shape inference fails.
    

  **NOTE** The output of this function should be used.  If it is not, a warning will be logged.  To mark the output as used, call its .mark_used() method."
  [ self indices value name ]
  (py/call-attr self "scatter"  self indices value name ))

(defn size 
  "Return the size of the TensorArray."
  [ self name ]
  (py/call-attr self "size"  self name ))

(defn split 
  "Split the values of a `Tensor` into the TensorArray.

    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to split.
      lengths: 1-D.  int32 vector with the lengths to use when splitting
        `value` along its first dimension.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the split occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if the shape inference fails.
    

  **NOTE** The output of this function should be used.  If it is not, a warning will be logged.  To mark the output as used, call its .mark_used() method."
  [ self value lengths name ]
  (py/call-attr self "split"  self value lengths name ))

(defn stack 
  "Return the values in the TensorArray as a stacked `Tensor`.

    All of the values must have been written and their shapes must all match.
    If input shapes have rank-`R`, then output shape will have rank-`(R+1)`.

    Args:
      name: A name for the operation (optional).

    Returns:
      All the tensors in the TensorArray stacked into one tensor.
    "
  [ self name ]
  (py/call-attr self "stack"  self name ))

(defn unstack 
  "Unstack the values of a `Tensor` in the TensorArray.

    If input value shapes have rank-`R`, then the output TensorArray will
    contain elements whose shapes are rank-`(R-1)`.

    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unstack.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the unstack occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if the shape inference fails.
    

  **NOTE** The output of this function should be used.  If it is not, a warning will be logged.  To mark the output as used, call its .mark_used() method."
  [ self value name ]
  (py/call-attr self "unstack"  self value name ))

(defn write 
  "Write `value` into index `index` of the TensorArray.

    Args:
      index: 0-D.  int32 scalar with the index to write to.
      value: N-D.  Tensor of type `dtype`.  The Tensor to write to this index.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the write occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if there are more writers than specified.
    "
  [ self index value name ]
  (py/call-attr self "write"  self index value name ))
