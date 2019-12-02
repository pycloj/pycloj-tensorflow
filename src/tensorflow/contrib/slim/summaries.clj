(ns tensorflow.contrib.slim.python.slim.summaries
  "Contains helper functions for creating summaries.

This module contains various helper functions for quickly and easily adding
tensorflow summaries. These allow users to print summary values
automatically as they are computed and add prefixes to collections of summaries.

Example usage:

  import tensorflow as tf
  slim = tf.contrib.slim

  slim.summaries.add_histogram_summaries(slim.variables.get_model_variables())
  slim.summaries.add_scalar_summary(total_loss, 'Total Loss')
  slim.summaries.add_scalar_summary(learning_rate, 'Learning Rate')
  slim.summaries.add_histogram_summaries(my_tensors)
  slim.summaries.add_zero_fraction_summaries(my_tensors)
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summaries (import-module "tensorflow.contrib.slim.python.slim.summaries"))

(defn add-histogram-summaries 
  "Adds a histogram summary for each of the given tensors.

  Args:
    tensors: A list of variable or op tensors.
    prefix: An optional prefix for the summary names.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  "
  [ tensors prefix ]
  (py/call-attr summaries "add_histogram_summaries"  tensors prefix ))

(defn add-histogram-summary 
  "Adds a histogram summary for the given tensor.

  Args:
    tensor: A variable or op tensor.
    name: The optional name for the summary.
    prefix: An optional prefix for the summary names.

  Returns:
    A scalar `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  "
  [ tensor name prefix ]
  (py/call-attr summaries "add_histogram_summary"  tensor name prefix ))

(defn add-image-summaries 
  "Adds an image summary for each of the given tensors.

  Args:
    tensors: A list of variable or op tensors.
    prefix: An optional prefix for the summary names.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  "
  [ tensors prefix ]
  (py/call-attr summaries "add_image_summaries"  tensors prefix ))
(defn add-image-summary 
  "Adds an image summary for the given tensor.

  Args:
    tensor: a variable or op tensor with shape [batch,height,width,channels]
    name: the optional name for the summary.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    An image `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  "
  [tensor name prefix  & {:keys [print_summary]} ]
    (py/call-attr-kw summaries "add_image_summary" [tensor name prefix] {:print_summary print_summary }))
(defn add-scalar-summaries 
  "Adds a scalar summary for each of the given tensors.

  Args:
    tensors: a list of variable or op tensors.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  "
  [tensors prefix  & {:keys [print_summary]} ]
    (py/call-attr-kw summaries "add_scalar_summaries" [tensors prefix] {:print_summary print_summary }))
(defn add-scalar-summary 
  "Adds a scalar summary for the given tensor.

  Args:
    tensor: a variable or op tensor.
    name: the optional name for the summary.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    A scalar `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  "
  [tensor name prefix  & {:keys [print_summary]} ]
    (py/call-attr-kw summaries "add_scalar_summary" [tensor name prefix] {:print_summary print_summary }))

(defn add-zero-fraction-summaries 
  "Adds a scalar zero-fraction summary for each of the given tensors.

  Args:
    tensors: a list of variable or op tensors.
    prefix: An optional prefix for the summary names.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  "
  [ tensors prefix ]
  (py/call-attr summaries "add_zero_fraction_summaries"  tensors prefix ))
(defn add-zero-fraction-summary 
  "Adds a summary for the percentage of zero values in the given tensor.

  Args:
    tensor: a variable or op tensor.
    name: the optional name for the summary.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    A scalar `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  "
  [tensor name prefix  & {:keys [print_summary]} ]
    (py/call-attr-kw summaries "add_zero_fraction_summary" [tensor name prefix] {:print_summary print_summary }))
