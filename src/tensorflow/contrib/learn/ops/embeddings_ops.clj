(ns tensorflow.contrib.learn.python.learn.ops.embeddings-ops
  "TensorFlow Ops to work with embeddings (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

Note: categorical variables are handled via embeddings in many cases.
For example, in case of words.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce embeddings-ops (import-module "tensorflow.contrib.learn.python.learn.ops.embeddings_ops"))

(defn categorical-variable 
  "Creates an embedding for categorical variable with given number of classes. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.contrib.layers.embed_sequence` instead.

Args:
  tensor_in: Input tensor with class identifier (can be batch or
    N-dimensional).
  n_classes: Number of classes.
  embedding_size: Size of embedding vector to represent each class.
  name: Name of this categorical variable.
Returns:
  Tensor of input shape, with additional dimension for embedding.

Example:
  Calling categorical_variable([1, 2], 5, 10, \"my_cat\"), will return 2 x 10
  tensor, where each row is representation of the class."
  [ tensor_in n_classes embedding_size name ]
  (py/call-attr embeddings-ops "categorical_variable"  tensor_in n_classes embedding_size name ))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw embeddings-ops "deprecated" [date instructions] {:warn_once warn_once }))
(defn embedding-lookup 
  "Provides a N dimensional version of tf.embedding_lookup. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.embedding_lookup` instead.

Ids are flattened to a 1d tensor before being passed to embedding_lookup
then, they are unflattend to match the original ids shape plus an extra
leading dimension of the size of the embeddings.

Args:
  params: List of tensors of size D0 x D1 x ... x Dn-2 x Dn-1.
  ids: N-dimensional tensor of B0 x B1 x .. x Bn-2 x Bn-1.
    Must contain indexes into params.
  name: Optional name for the op.

Returns:
  A tensor of size B0 x B1 x .. x Bn-2 x Bn-1 x D1 x ... x Dn-2 x Dn-1
  containing the values from the params tensor(s) for indecies in ids.

Raises:
  ValueError: if some parameters are invalid."
  [params ids  & {:keys [name]} ]
    (py/call-attr-kw embeddings-ops "embedding_lookup" [params ids] {:name name }))
