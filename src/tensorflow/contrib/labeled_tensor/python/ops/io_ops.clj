(ns tensorflow.contrib.labeled-tensor.python.ops.io-ops
  "Input parsing code for LabeledTensors."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io-ops (import-module "tensorflow.contrib.labeled_tensor.python.ops.io_ops"))

(defn parse-example 
  "Parse `Example` protos into a `dict` of labeled tensors.

  See tf.parse_example.

  Args:
    serialized: A 1-D LabeledTensor of strings, a batch of binary serialized
      `Example` protos.
    features: A `dict` mapping feature keys to `labeled_tensor.FixedLenFeature`
      values.
    name: A name for this operation (optional).
    example_names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos in the batch.

  Returns:
    A `dict` mapping feature keys to `LabeledTensor` values. The single axis
    from `serialized` will be prepended to the axes provided by each feature.

  Raises:
    ValueError: if any feature is invalid.
  "
  [ serialized features name example_names ]
  (py/call-attr io-ops "parse_example"  serialized features name example_names ))

(defn parse-single-example 
  "Parses a single `Example` proto.

  See tf.parse_single_example.

  Args:
    serialized: A scalar string Tensor or LabeledTensor, a single serialized
      Example.
    features: A `dict` mapping feature keys to `labeled_tensor.FixedLenFeature`
      values.
    name: A name for this operation (optional).
    example_names: (Optional) A scalar string Tensor, the associated name.

  Returns:
    A `dict` mapping feature keys to `LabeledTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  "
  [ serialized features name example_names ]
  (py/call-attr io-ops "parse_single_example"  serialized features name example_names ))

(defn placeholder 
  "Create a placeholder for a labeled tensor.

  For example:

    lt.placeholder(tf.float32, ['batch', ('channel', ['r', 'g', 'b'])])

  See tf.compat.v1.placeholder for more details.

  Args:
    dtype: The type of elements in the tensor to be fed.
    axes: sequence of strings (denoting axes of unknown size) and/or objects
      convertable to lt.Axis to label the result.
    name: Optional op name.

  Returns:
    Placeholder labeled tensor.
  "
  [ dtype axes name ]
  (py/call-attr io-ops "placeholder"  dtype axes name ))
