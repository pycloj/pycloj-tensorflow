(ns tensorflow.name-scope
  "A context manager for use when defining a Python op.

  This context manager validates that the given `values` are from the
  same graph, makes that graph the default graph, and pushes a
  name scope in that graph (see
  `tf.Graph.name_scope`
  for more details on that).

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope(name, \"MyOp\", [a, b, c]) as scope:
      a = tf.convert_to_tensor(a, name=\"a\")
      b = tf.convert_to_tensor(b, name=\"b\")
      c = tf.convert_to_tensor(c, name=\"c\")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn name-scope 
  "A context manager for use when defining a Python op.

  This context manager validates that the given `values` are from the
  same graph, makes that graph the default graph, and pushes a
  name scope in that graph (see
  `tf.Graph.name_scope`
  for more details on that).

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope(name, \"MyOp\", [a, b, c]) as scope:
      a = tf.convert_to_tensor(a, name=\"a\")
      b = tf.convert_to_tensor(b, name=\"b\")
      c = tf.convert_to_tensor(c, name=\"c\")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```
  "
  [ name default_name values ]
  (py/call-attr tensorflow "name_scope"  name default_name values ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
