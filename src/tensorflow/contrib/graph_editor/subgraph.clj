(ns tensorflow.contrib.graph-editor.subgraph
  "SubGraphView: a subgraph view on an existing tf.Graph.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce subgraph (import-module "tensorflow.contrib.graph_editor.subgraph"))

(defn iteritems 
  "Return an iterator over the (key, value) pairs of a dictionary."
  [ d ]
  (py/call-attr subgraph "iteritems"  d ))

(defn make-view 
  "Create a SubGraphView from selected operations and passthrough tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or 2) (array of)
      `tf.Operation` 3) (array of) `tf.Tensor`. Those objects will be converted
      into a list of operations and a list of candidate for passthrough tensors.
    **kwargs: keyword graph is used 1) to check that the ops and ts are from
      the correct graph 2) for regular expression query
  Returns:
    A subgraph view.
  Raises:
    TypeError: if the optional keyword argument graph is not a `tf.Graph`
      or if an argument in args is not an (array of) `tf.Tensor`
      or an (array of) `tf.Operation` or a string or a regular expression.
    ValueError: if one of the keyword arguments is unexpected.
  "
  [  ]
  (py/call-attr subgraph "make_view"  ))

(defn make-view-from-scope 
  "Make a subgraph from a name scope.

  Args:
    scope: the name of the scope.
    graph: the `tf.Graph`.
  Returns:
    A subgraph view representing the given scope.
  "
  [ scope graph ]
  (py/call-attr subgraph "make_view_from_scope"  scope graph ))
