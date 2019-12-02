(ns tensorflow.contrib.graph-editor.transform
  "Class to transform an subgraph into another.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce transform (import-module "tensorflow.contrib.graph_editor.transform"))

(defn assign-renamed-collections-handler 
  "Add the transformed elem to the (renamed) collections of elem.

  A collection is renamed only if is not a known key, as described in
  `tf.compat.v1.GraphKeys`.

  Args:
    info: Transform._TmpInfo instance.
    elem: the original element (`tf.Tensor` or `tf.Operation`)
    elem_: the transformed element
  "
  [ info elem elem_ ]
  (py/call-attr transform "assign_renamed_collections_handler"  info elem elem_ ))
(defn copy 
  "Copy a subgraph.

  Args:
    sgv: the source subgraph-view. This argument is converted to a subgraph
      using the same rules than the function subgraph.make_view.
    dst_graph: the destination graph.
    dst_scope: the destination scope.
    src_scope: the source scope.
    reuse_dst_scope: if True the dst_scope is re-used if it already exists.
      Otherwise, the scope is given a unique name based on the one given
      by appending an underscore followed by a digit (default).
  Returns:
    A tuple `(sgv, info)` where:
      `sgv` is the transformed subgraph view;
      `info` is an instance of TransformerInfo containing
      information about the transform, including mapping between
      original and transformed tensors and operations.
  Raises:
    TypeError: if `dst_graph` is not a `tf.Graph`.
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [sgv dst_graph  & {:keys [dst_scope src_scope reuse_dst_scope]} ]
    (py/call-attr-kw transform "copy" [sgv dst_graph] {:dst_scope dst_scope :src_scope src_scope :reuse_dst_scope reuse_dst_scope }))

(defn copy-op-handler 
  "Copy a `tf.Operation`.

  Args:
    info: Transform._TmpInfo instance.
    op: the `tf.Operation` to be copied.
    new_inputs: The new inputs for this op.
    copy_shape: also copy the shape of the tensor
    nodedef_fn: If provided, a function that will be run on the NodeDef
      and should return a mutated NodeDef before a new Operation is created.
      This is useful as certain features cannot be set on the Operation and
      must be modified in NodeDef.

  Returns:
    A `(op, op_outputs)` tuple containing the transformed op and its outputs.
  "
  [info op new_inputs & {:keys [copy_shape nodedef_fn]
                       :or {nodedef_fn None}} ]
    (py/call-attr-kw transform "copy_op_handler" [info op new_inputs] {:copy_shape copy_shape :nodedef_fn nodedef_fn }))
(defn copy-with-input-replacements 
  "Copy a subgraph, replacing some of its inputs.

  Note a replacement only happens if the tensor to be replaced
  is an input of the given subgraph. The inputs of a subgraph can
  be queried using sgv.inputs.

  Args:
    sgv: the source subgraph-view. This argument is converted to a subgraph
      using the same rules as the function subgraph.make_view.
    replacement_ts: dictionary mapping from original tensors to the
      replaced one.
    dst_graph: the destination graph.
    dst_scope: the destination scope.
    src_scope: the source scope.
    reuse_dst_scope: if True the dst_scope is re-used if it already exists.
      Otherwise, the scope is given a unique name based on the one given
      by appending an underscore followed by a digit (default).
  Returns:
    A tuple `(sgv, info)` where:
      `sgv` is the transformed subgraph view;
      `info` is an instance of TransformerInfo containing
      information about the transform, including mapping between
      original and transformed tensors and operations.
  Raises:
    TypeError: if dst_graph is not a tf.Graph.
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules as the function subgraph.make_view.
  "
  [sgv replacement_ts dst_graph  & {:keys [dst_scope src_scope reuse_dst_scope]} ]
    (py/call-attr-kw transform "copy_with_input_replacements" [sgv replacement_ts dst_graph] {:dst_scope dst_scope :src_scope src_scope :reuse_dst_scope reuse_dst_scope }))
(defn deepcopy 
  "Deep copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info.
    "
  [x memo  & {:keys [_nil]} ]
    (py/call-attr-kw transform "deepcopy" [x memo] {:_nil _nil }))
(defn graph-replace 
  "Create a new graph which compute the targets from the replaced Tensors.

  Args:
    target_ts: a single tf.Tensor or an iterable of tf.Tensor.
    replacement_ts: dictionary mapping from original tensors to replaced tensors
    dst_scope: the destination scope.
    src_scope: the source scope.
    reuse_dst_scope: if True the dst_scope is re-used if it already exists.
      Otherwise, the scope is given a unique name based on the one given
      by appending an underscore followed by a digit (default).
  Returns:
    A single tf.Tensor or a list of target tf.Tensor, depending on
    the type of the input argument `target_ts`.
    The returned tensors are recomputed using the tensors from replacement_ts.
  Raises:
    ValueError: if the targets are not connected to replacement_ts.
  "
  [target_ts replacement_ts  & {:keys [dst_scope src_scope reuse_dst_scope]} ]
    (py/call-attr-kw transform "graph_replace" [target_ts replacement_ts] {:dst_scope dst_scope :src_scope src_scope :reuse_dst_scope reuse_dst_scope }))

(defn iteritems 
  "Return an iterator over the (key, value) pairs of a dictionary."
  [ d ]
  (py/call-attr transform "iteritems"  d ))

(defn keep-t-if-possible-handler 
  "Transform a tensor into itself (identity) if possible.

  This handler transform a tensor into itself if the source and destination
  graph are the same. Otherwise it will create a placeholder.
  This handler is typically used to transform a hidden input tensors.

  Args:
    info: Transform._TmpInfo instance.
    t: tensor whose input must be transformed into a place holder.
  Returns:
    The tensor generated by the newly created place holder.
  "
  [ info t ]
  (py/call-attr transform "keep_t_if_possible_handler"  info t ))

(defn replace-t-with-placeholder-handler 
  "Transform a tensor into a placeholder tensor.

  This handler is typically used to transform a subgraph input tensor into a
  placeholder.

  Args:
    info: Transform._TmpInfo instance.
    t: tensor whose input must be transformed into a place holder.
  Returns:
    The tensor generated by the newly created place holder.
  "
  [ info t ]
  (py/call-attr transform "replace_t_with_placeholder_handler"  info t ))
(defn transform-op-if-inside-handler 
  "Transform an optional op only if it is inside the subgraph.

  This handler is typically use to handle original op: it is fine to keep them
  if they are inside the subgraph, otherwise they are just ignored.

  Args:
    info: Transform._TmpInfo instance.
    op: the optional op to transform (or ignore).
    keep_if_possible: re-attach to the original op if possible, that is,
      if the source graph and the destination graph are the same.
  Returns:
    The transformed op or None.
  "
  [info op  & {:keys [keep_if_possible]} ]
    (py/call-attr-kw transform "transform_op_if_inside_handler" [info op] {:keep_if_possible keep_if_possible }))
