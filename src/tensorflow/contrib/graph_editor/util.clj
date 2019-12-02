(ns tensorflow.contrib.graph-editor.util
  "Utility functions for the graph_editor.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce util (import-module "tensorflow.contrib.graph_editor.util"))

(defn check-graphs 
  "Check that all the element in args belong to the same graph.

  Args:
    *args: a list of object with a obj.graph property.
  Raises:
    ValueError: if all the elements do not belong to the same graph.
  "
  [  ]
  (py/call-attr util "check_graphs"  ))

(defn concatenate-unique 
  "Add all the elements of `lb` to `la` if they are not there already.

  The elements added to `la` maintain ordering with respect to `lb`.

  Args:
    la: List of Python objects.
    lb: List of Python objects.
  Returns:
    `la`: The list `la` with missing elements from `lb`.
  "
  [ la lb ]
  (py/call-attr util "concatenate_unique"  la lb ))
(defn find-corresponding 
  "Find corresponding ops/tensors in a different graph.

  `targets` is a Python tree, that is, a nested structure of iterable
  (list, tupple, dictionary) whose leaves are instances of
  `tf.Tensor` or `tf.Operation`

  Args:
    targets: A Python tree containing `tf.Tensor` or `tf.Operation`
      belonging to the original graph.
    dst_graph: The graph in which the corresponding graph element must be found.
    dst_scope: A scope which is prepended to the name to look for.
    src_scope: A scope which is removed from the original of `top` name.

  Returns:
    A Python tree containin the corresponding tf.Tensor` or a `tf.Operation`.

  Raises:
    ValueError: if `src_name` does not start with `src_scope`.
    TypeError: if `top` is not a `tf.Tensor` or a `tf.Operation`
    KeyError: If the corresponding graph element cannot be found.
  "
  [targets dst_graph  & {:keys [dst_scope src_scope]} ]
    (py/call-attr-kw util "find_corresponding" [targets dst_graph] {:dst_scope dst_scope :src_scope src_scope }))
(defn find-corresponding-elem 
  "Find corresponding op/tensor in a different graph.

  Args:
    target: A `tf.Tensor` or a `tf.Operation` belonging to the original graph.
    dst_graph: The graph in which the corresponding graph element must be found.
    dst_scope: A scope which is prepended to the name to look for.
    src_scope: A scope which is removed from the original of `target` name.

  Returns:
    The corresponding tf.Tensor` or a `tf.Operation`.

  Raises:
    ValueError: if `src_name` does not start with `src_scope`.
    TypeError: if `target` is not a `tf.Tensor` or a `tf.Operation`
    KeyError: If the corresponding graph element cannot be found.
  "
  [target dst_graph  & {:keys [dst_scope src_scope]} ]
    (py/call-attr-kw util "find_corresponding_elem" [target dst_graph] {:dst_scope dst_scope :src_scope src_scope }))

(defn flatten-tree 
  "Flatten a tree into a list.

  Args:
    tree: iterable or not. If iterable, its elements (child) can also be
      iterable or not.
    leaves: list to which the tree leaves are appended (None by default).
  Returns:
    A list of all the leaves in the tree.
  "
  [ tree leaves ]
  (py/call-attr util "flatten_tree"  tree leaves ))

(defn get-consuming-ops 
  "Return all the consuming ops of the tensors in ts.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the consuming `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if ts cannot be converted to a list of `tf.Tensor`.
  "
  [ ts ]
  (py/call-attr util "get_consuming_ops"  ts ))

(defn get-generating-ops 
  "Return all the generating ops of the tensors in `ts`.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the generating `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor`.
  "
  [ ts ]
  (py/call-attr util "get_generating_ops"  ts ))

(defn get-predefined-collection-names 
  "Return all the predefined collection names."
  [  ]
  (py/call-attr util "get_predefined_collection_names"  ))

(defn get-tensors 
  "get all the tensors which are input or output of an op in the graph.

  Args:
    graph: a `tf.Graph`.
  Returns:
    A list of `tf.Tensor`.
  Raises:
    TypeError: if graph is not a `tf.Graph`.
  "
  [ graph ]
  (py/call-attr util "get_tensors"  graph ))
(defn get-unique-graph 
  "Return the unique graph used by the all the elements in tops.

  Args:
    tops: list of elements to check (usually a list of tf.Operation and/or
      tf.Tensor). Or a tf.Graph.
    check_types: check that the element in tops are of given type(s). If None,
      the types (tf.Operation, tf.Tensor) are used.
    none_if_empty: don't raise an error if tops is an empty list, just return
      None.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of tf.Operation.
    ValueError: if the graph is not unique.
  "
  [tops check_types  & {:keys [none_if_empty]} ]
    (py/call-attr-kw util "get_unique_graph" [tops check_types] {:none_if_empty none_if_empty }))

(defn is-iterable 
  "Return true if the object is iterable."
  [ obj ]
  (py/call-attr util "is_iterable"  obj ))

(defn iteritems 
  "Return an iterator over the (key, value) pairs of a dictionary."
  [ d ]
  (py/call-attr util "iteritems"  d ))
(defn make-list-of-op 
  "Convert ops to a list of `tf.Operation`.

  Args:
    ops: can be an iterable of `tf.Operation`, a `tf.Graph` or a single
      operation.
    check_graph: if `True` check if all the operations belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ts: if True, silently ignore `tf.Tensor`.
  Returns:
    A newly created list of `tf.Operation`.
  Raises:
    TypeError: if ops cannot be converted to a list of `tf.Operation` or,
     if `check_graph` is `True`, if all the ops do not belong to the
     same graph.
  "
  [ops  & {:keys [check_graph allow_graph ignore_ts]} ]
    (py/call-attr-kw util "make_list_of_op" [ops] {:check_graph check_graph :allow_graph allow_graph :ignore_ts ignore_ts }))
(defn make-list-of-t 
  "Convert ts to a list of `tf.Tensor`.

  Args:
    ts: can be an iterable of `tf.Tensor`, a `tf.Graph` or a single tensor.
    check_graph: if `True` check if all the tensors belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ops: if `True`, silently ignore `tf.Operation`.
  Returns:
    A newly created list of `tf.Tensor`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor` or,
     if `check_graph` is `True`, if all the ops do not belong to the same graph.
  "
  [ts  & {:keys [check_graph allow_graph ignore_ops]} ]
    (py/call-attr-kw util "make_list_of_t" [ts] {:check_graph check_graph :allow_graph allow_graph :ignore_ops ignore_ops }))
(defn make-placeholder-from-dtype-and-shape 
  "Create a tf.compat.v1.placeholder for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.
  The placeholder is named using the function placeholder_name (with no
  tensor argument).

  Args:
    dtype: the tensor type.
    shape: the tensor shape (optional).
    scope: absolute scope within which to create the placeholder. None means
      that the scope of t is preserved. \"\" means the root scope.
    prefix: placeholder name prefix.

  Returns:
    A newly created tf.placeholder.
  "
  [dtype shape scope  & {:keys [prefix]} ]
    (py/call-attr-kw util "make_placeholder_from_dtype_and_shape" [dtype shape scope] {:prefix prefix }))
(defn make-placeholder-from-tensor 
  "Create a `tf.compat.v1.placeholder` for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.

  Args:
    t: a `tf.Tensor` whose name will be used to create the placeholder (see
      function placeholder_name).
    scope: absolute scope within which to create the placeholder. None means
      that the scope of `t` is preserved. `\"\"` means the root scope.
    prefix: placeholder name prefix.

  Returns:
    A newly created `tf.compat.v1.placeholder`.
  Raises:
    TypeError: if `t` is not `None` or a `tf.Tensor`.
  "
  [t scope  & {:keys [prefix]} ]
    (py/call-attr-kw util "make_placeholder_from_tensor" [t scope] {:prefix prefix }))
(defn placeholder-name 
  "Create placeholder name for the graph editor.

  Args:
    t: optional tensor on which the placeholder operation's name will be based
      on
    scope: absolute scope with which to prefix the placeholder's name. None
      means that the scope of t is preserved. \"\" means the root scope.
    prefix: placeholder name prefix.
  Returns:
    A new placeholder name prefixed by \"geph\". Note that \"geph\" stands for
      Graph Editor PlaceHolder. This convention allows to quickly identify the
      placeholder generated by the Graph Editor.
  Raises:
    TypeError: if t is not None or a tf.Tensor.
  "
  [t scope  & {:keys [prefix]} ]
    (py/call-attr-kw util "placeholder_name" [t scope] {:prefix prefix }))

(defn scope-basename 
  ""
  [ scope ]
  (py/call-attr util "scope_basename"  scope ))

(defn scope-dirname 
  ""
  [ scope ]
  (py/call-attr util "scope_dirname"  scope ))

(defn scope-finalize 
  ""
  [ scope ]
  (py/call-attr util "scope_finalize"  scope ))
(defn transform-tree 
  "Transform all the nodes of a tree.

  Args:
    tree: iterable or not. If iterable, its elements (child) can also be
      iterable or not.
    fn: function to apply to each leaves.
    iterable_type: type use to construct the resulting tree for unknown
      iterable, typically `list` or `tuple`.
  Returns:
    A tree whose leaves has been transformed by `fn`.
    The hierarchy of the output tree mimics the one of the input tree.
  "
  [tree fn  & {:keys [iterable_type]} ]
    (py/call-attr-kw util "transform_tree" [tree fn] {:iterable_type iterable_type }))
