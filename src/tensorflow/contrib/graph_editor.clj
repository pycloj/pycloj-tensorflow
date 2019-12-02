(ns tensorflow.contrib.graph-editor
  "TensorFlow Graph Editor."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce graph-editor (import-module "tensorflow.contrib.graph_editor"))

(defn add-control-inputs 
  "Add the control inputs cops to op.

  Warning: this function is directly manipulating the internals of the tf.Graph.

  Args:
    op: a tf.Operation to which the control inputs are added.
    cops: an object convertible to a list of `tf.Operation`.
  Raises:
    TypeError: if op is not a tf.Operation
    ValueError: if any cop in cops is already a control input of op.
  "
  [ op cops ]
  (py/call-attr graph-editor "add_control_inputs"  op cops ))

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
  (py/call-attr graph-editor "assign_renamed_collections_handler"  info elem elem_ ))

(defn bypass 
  "Bypass the given subgraph by connecting its inputs to its outputs.

  Args:
    sgv: the subgraph view to be bypassed. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
      Note that sgv is modified in place.
  Returns:
    A tuple `(sgv, detached_inputs)` where:
      `sgv` is a new subgraph view of the bypassed subgraph;
      `detached_inputs` is a list of the created input placeholders.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [ sgv ]
  (py/call-attr graph-editor "bypass"  sgv ))

(defn can-be-regex 
  "Return True if obj can be turned into a regular expression."
  [ obj ]
  (py/call-attr graph-editor "can_be_regex"  obj ))

(defn check-cios 
  "Do various check on control_inputs and control_outputs.

  Args:
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A tuple `(control_inputs, control_outputs)` where:
      `control_inputs` is a boolean indicating whether to use control inputs.
      `control_outputs` is an instance of util.ControlOutputs or None
  Raises:
    ValueError: if control_inputs is an instance of util.ControlOutputs but
      control_outputs is not None
    TypeError: if control_outputs is not None and is not a util.ControlOutputs.
  "
  [ & {:keys [control_inputs control_outputs control_ios]
       :or {control_outputs None control_ios None}} ]
  
   (py/call-attr-kw graph-editor "check_cios" [] {:control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

(defn compute-boundary-ts 
  "Compute the tensors at the boundary of a set of ops.

  This function looks at all the tensors connected to the given ops (in/out)
  and classify them into three categories:
  1) input tensors: tensors whose generating operation is not in ops.
  2) output tensors: tensors whose consumer operations are not in ops
  3) inside tensors: tensors which are neither input nor output tensors.

  Note that a tensor can be both an inside tensor and an output tensor if it is
  consumed by operations both outside and inside of `ops`.

  Args:
    ops: an object convertible to a list of tf.Operation.
  Returns:
    A tuple `(outside_input_ts, outside_output_ts, inside_ts)` where:
      `outside_input_ts` is a Python list of input tensors;
      `outside_output_ts` is a python list of output tensors;
      `inside_ts` is a python list of inside tensors.
    Since a tensor can be both an inside tensor and an output tensor,
    `outside_output_ts` and `inside_ts` might intersect.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  "
  [ ops ]
  (py/call-attr graph-editor "compute_boundary_ts"  ops ))
(defn connect 
  "Connect the outputs of sgv0 to the inputs of sgv1.

  Args:
    sgv0: the first subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules as the function
      subgraph.make_view.
      Note that sgv0 is modified in place.
    sgv1: the second subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules as the function
      subgraph.make_view.
      Note that sgv1 is modified in place.
    disconnect_first: if True the current outputs of sgv0 are disconnected.
  Returns:
    A tuple `(sgv0, sgv1)` of the now connected subgraphs.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [sgv0 sgv1  & {:keys [disconnect_first]} ]
    (py/call-attr-kw graph-editor "connect" [sgv0 sgv1] {:disconnect_first disconnect_first }))
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
    (py/call-attr-kw graph-editor "copy" [sgv dst_graph] {:dst_scope dst_scope :src_scope src_scope :reuse_dst_scope reuse_dst_scope }))

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
    (py/call-attr-kw graph-editor "copy_op_handler" [info op new_inputs] {:copy_shape copy_shape :nodedef_fn nodedef_fn }))
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
    (py/call-attr-kw graph-editor "copy_with_input_replacements" [sgv replacement_ts dst_graph] {:dst_scope dst_scope :src_scope src_scope :reuse_dst_scope reuse_dst_scope }))

(defn detach 
  "Detach both the inputs and the outputs of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
      Note that sgv is modified in place.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A tuple `(sgv, detached_inputs, detached_outputs)` where:
    `sgv` is a new subgraph view of the detached subgraph;
    `detach_inputs` is a list of the created input placeholders;
    `detach_outputs` is a list of the created output placeholders.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [sgv & {:keys [control_inputs control_outputs control_ios]
                       :or {control_outputs None control_ios None}} ]
    (py/call-attr-kw graph-editor "detach" [sgv] {:control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

(defn detach-control-inputs 
  "Detach all the external control inputs of the subgraph sgv.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
  "
  [ sgv ]
  (py/call-attr graph-editor "detach_control_inputs"  sgv ))

(defn detach-control-outputs 
  "Detach all the external control outputs of the subgraph sgv.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
    control_outputs: a util.ControlOutputs instance.
  "
  [ sgv control_outputs ]
  (py/call-attr graph-editor "detach_control_outputs"  sgv control_outputs ))
(defn detach-inputs 
  "Detach the inputs of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
      Note that sgv is modified in place.
    control_inputs: if True control_inputs are also detached.
  Returns:
    A tuple `(sgv, input_placeholders)` where
      `sgv` is a new subgraph view of the detached subgraph;
      `input_placeholders` is a list of the created input placeholders.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [sgv  & {:keys [control_inputs]} ]
    (py/call-attr-kw graph-editor "detach_inputs" [sgv] {:control_inputs control_inputs }))

(defn detach-outputs 
  "Detach the output of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
      Note that sgv is modified in place.
    control_outputs: a util.ControlOutputs instance or None. If not None the
      control outputs are also detached.
  Returns:
    A tuple `(sgv, output_placeholders)` where
      `sgv` is a new subgraph view of the detached subgraph;
      `output_placeholders` is a list of the created output placeholders.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [ sgv control_outputs ]
  (py/call-attr graph-editor "detach_outputs"  sgv control_outputs ))

(defn filter-ops 
  "Get the ops passing the given filter.

  Args:
    ops: an object convertible to a list of tf.Operation.
    positive_filter: a function deciding where to keep an operation or not.
      If True, all the operations are returned.
  Returns:
    A list of selected tf.Operation.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  "
  [ ops positive_filter ]
  (py/call-attr graph-editor "filter_ops"  ops positive_filter ))

(defn filter-ops-from-regex 
  "Get all the operations that match the given regex.

  Args:
    ops: an object convertible to a list of `tf.Operation`.
    regex: a regular expression matching the operation's name.
      For example, `\"^foo(/.*)?$\"` will match all the operations in the \"foo\"
      scope.
  Returns:
    A list of `tf.Operation`.
  Raises:
    TypeError: if ops cannot be converted to a list of `tf.Operation`.
  "
  [ ops regex ]
  (py/call-attr graph-editor "filter_ops_from_regex"  ops regex ))

(defn filter-ts 
  "Get all the tensors which are input or output of an op in ops.

  Args:
    ops: an object convertible to a list of `tf.Operation`.
    positive_filter: a function deciding whether to keep a tensor or not.
      If `True`, all the tensors are returned.
  Returns:
    A list of `tf.Tensor`.
  Raises:
    TypeError: if ops cannot be converted to a list of `tf.Operation`.
  "
  [ ops positive_filter ]
  (py/call-attr graph-editor "filter_ts"  ops positive_filter ))

(defn filter-ts-from-regex 
  "Get all the tensors linked to ops that match the given regex.

  Args:
    ops: an object convertible to a list of tf.Operation.
    regex: a regular expression matching the tensors' name.
      For example, \"^foo(/.*)?:\d+$\" will match all the tensors in the \"foo\"
      scope.
  Returns:
    A list of tf.Tensor.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  "
  [ ops regex ]
  (py/call-attr graph-editor "filter_ts_from_regex"  ops regex ))

(defn get-backward-walk-ops 
  "Do a backward graph walk and return all the visited ops. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-06-06.
Instructions for updating:
Please use tensorflow.python.ops.op_selector.get_backward_walk_ops.

Args:
  seed_ops: an iterable of operations from which the backward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the generators of those tensors.
  inclusive: if True the given seed_ops are also part of the resulting set.
  within_ops: an iterable of `tf.Operation` within which the search is
    restricted. If `within_ops` is `None`, the search is performed within
    the whole graph.
  within_ops_fn: if provided, a function on ops that should return True iff
    the op is within the graph traversal. This can be used along within_ops,
    in which case an op is within if it is also in within_ops.
  stop_at_ts: an iterable of tensors at which the graph walk stops.
  control_inputs: if True, control inputs will be used while moving backward.
Returns:
  A Python set of all the `tf.Operation` behind `seed_ops`.
Raises:
  TypeError: if `seed_ops` or `within_ops` cannot be converted to a list of
    `tf.Operation`."
  [seed_ops & {:keys [inclusive within_ops within_ops_fn stop_at_ts control_inputs]
                       :or {within_ops None within_ops_fn None}} ]
    (py/call-attr-kw graph-editor "get_backward_walk_ops" [seed_ops] {:inclusive inclusive :within_ops within_ops :within_ops_fn within_ops_fn :stop_at_ts stop_at_ts :control_inputs control_inputs }))

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
  (py/call-attr graph-editor "get_consuming_ops"  ts ))

(defn get-forward-walk-ops 
  "Do a forward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of `tf.Operation` within which the search is
      restricted. If `within_ops` is `None`, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    stop_at_ts: an iterable of tensors at which the graph walk stops.
    control_outputs: a `util.ControlOutputs` instance or None.
      If not `None`, it will be used while walking the graph forward.
  Returns:
    A Python set of all the `tf.Operation` ahead of `seed_ops`.
  Raises:
    TypeError: if `seed_ops` or `within_ops` cannot be converted to a list of
      `tf.Operation`.
  "
  [seed_ops & {:keys [inclusive within_ops within_ops_fn stop_at_ts control_outputs]
                       :or {within_ops None within_ops_fn None control_outputs None}} ]
    (py/call-attr-kw graph-editor "get_forward_walk_ops" [seed_ops] {:inclusive inclusive :within_ops within_ops :within_ops_fn within_ops_fn :stop_at_ts stop_at_ts :control_outputs control_outputs }))

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
  (py/call-attr graph-editor "get_generating_ops"  ts ))

(defn get-name-scope-ops 
  "Get all the operations under the given scope path.

  Args:
    ops: an object convertible to a list of tf.Operation.
    scope: a scope path.
  Returns:
    A list of tf.Operation.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  "
  [ ops scope ]
  (py/call-attr graph-editor "get_name_scope_ops"  ops scope ))

(defn get-ops-ios 
  "Return all the `tf.Operation` which are connected to an op in ops.

  Args:
    ops: an object convertible to a list of `tf.Operation`.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of `util.ControlOutputs` or `None`. If not
      `None`, control outputs are enabled.
    control_ios:  An instance of `util.ControlOutputs` or `None`. If not `None`,
      both control inputs and control outputs are enabled. This is equivalent to
      set `control_inputs` to `True` and `control_outputs` to the
      `util.ControlOutputs` instance.
  Returns:
    All the `tf.Operation` surrounding the given ops.
  Raises:
    TypeError: if `ops` cannot be converted to a list of `tf.Operation`.
  "
  [ops & {:keys [control_inputs control_outputs control_ios]
                       :or {control_outputs None control_ios None}} ]
    (py/call-attr-kw graph-editor "get_ops_ios" [ops] {:control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

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
  (py/call-attr graph-editor "get_tensors"  graph ))

(defn get-walks-intersection-ops 
  "Return the intersection of a forward and a backward walk.

  Args:
    forward_seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    backward_seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    forward_inclusive: if True the given forward_seed_ops are also part of the
      resulting set.
    backward_inclusive: if True the given backward_seed_ops are also part of the
      resulting set.
    within_ops: an iterable of tf.Operation within which the search is
      restricted. If within_ops is None, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A Python set of all the tf.Operation in the intersection of a forward and a
      backward walk.
  Raises:
    TypeError: if `forward_seed_ops` or `backward_seed_ops` or `within_ops`
      cannot be converted to a list of `tf.Operation`.
  "
  [forward_seed_ops backward_seed_ops & {:keys [forward_inclusive backward_inclusive within_ops within_ops_fn control_inputs control_outputs control_ios]
                       :or {within_ops None within_ops_fn None control_outputs None control_ios None}} ]
    (py/call-attr-kw graph-editor "get_walks_intersection_ops" [forward_seed_ops backward_seed_ops] {:forward_inclusive forward_inclusive :backward_inclusive backward_inclusive :within_ops within_ops :within_ops_fn within_ops_fn :control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

(defn get-walks-union-ops 
  "Return the union of a forward and a backward walk.

  Args:
    forward_seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    backward_seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    forward_inclusive: if True the given forward_seed_ops are also part of the
      resulting set.
    backward_inclusive: if True the given backward_seed_ops are also part of the
      resulting set.
    within_ops: restrict the search within those operations. If within_ops is
      None, the search is done within the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A Python set of all the tf.Operation in the union of a forward and a
      backward walk.
  Raises:
    TypeError: if forward_seed_ops or backward_seed_ops or within_ops cannot be
      converted to a list of tf.Operation.
  "
  [forward_seed_ops backward_seed_ops & {:keys [forward_inclusive backward_inclusive within_ops within_ops_fn control_inputs control_outputs control_ios]
                       :or {within_ops None within_ops_fn None control_outputs None control_ios None}} ]
    (py/call-attr-kw graph-editor "get_walks_union_ops" [forward_seed_ops backward_seed_ops] {:forward_inclusive forward_inclusive :backward_inclusive backward_inclusive :within_ops within_ops :within_ops_fn within_ops_fn :control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

(defn get-within-boundary-ops 
  "Return all the `tf.Operation` within the given boundary.

  Args:
    ops: an object convertible to a list of `tf.Operation`. those ops define the
      set in which to perform the operation (if a `tf.Graph` is given, it
      will be converted to the list of all its operations).
    seed_ops: the operations from which to start expanding.
    boundary_ops: the ops forming the boundary.
    inclusive: if `True`, the result will also include the boundary ops.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of `util.ControlOutputs` or `None`. If not
      `None`, control outputs are enabled.
    control_ios:  An instance of `util.ControlOutputs` or `None`. If not
      `None`, both control inputs and control outputs are enabled. This is
      equivalent to set control_inputs to True and control_outputs to
      the `util.ControlOutputs` instance.
  Returns:
    All the `tf.Operation` surrounding the given ops.
  Raises:
    TypeError: if `ops` or `seed_ops` cannot be converted to a list of
      `tf.Operation`.
    ValueError: if the boundary is intersecting with the seeds.
  "
  [ops seed_ops & {:keys [boundary_ops inclusive control_inputs control_outputs control_ios]
                       :or {control_outputs None control_ios None}} ]
    (py/call-attr-kw graph-editor "get_within_boundary_ops" [ops seed_ops] {:boundary_ops boundary_ops :inclusive inclusive :control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))
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
    (py/call-attr-kw graph-editor "graph_replace" [target_ts replacement_ts] {:dst_scope dst_scope :src_scope src_scope :reuse_dst_scope reuse_dst_scope }))

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
  (py/call-attr graph-editor "keep_t_if_possible_handler"  info t ))
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
    (py/call-attr-kw graph-editor "make_list_of_op" [ops] {:check_graph check_graph :allow_graph allow_graph :ignore_ts ignore_ts }))
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
    (py/call-attr-kw graph-editor "make_list_of_t" [ts] {:check_graph check_graph :allow_graph allow_graph :ignore_ops ignore_ops }))
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
    (py/call-attr-kw graph-editor "make_placeholder_from_dtype_and_shape" [dtype shape scope] {:prefix prefix }))
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
    (py/call-attr-kw graph-editor "make_placeholder_from_tensor" [t scope] {:prefix prefix }))

(defn make-regex 
  "Return a compiled regular expression.

  Args:
    obj: a string or a regular expression.
  Returns:
    A compiled regular expression.
  Raises:
    ValueError: if obj could not be converted to a regular expression.
  "
  [ obj ]
  (py/call-attr graph-editor "make_regex"  obj ))

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
  (py/call-attr graph-editor "make_view"  ))

(defn make-view-from-scope 
  "Make a subgraph from a name scope.

  Args:
    scope: the name of the scope.
    graph: the `tf.Graph`.
  Returns:
    A subgraph view representing the given scope.
  "
  [ scope graph ]
  (py/call-attr graph-editor "make_view_from_scope"  scope graph ))
(defn ph 
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
    (py/call-attr-kw graph-editor "ph" [dtype shape scope] {:prefix prefix }))
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
    (py/call-attr-kw graph-editor "placeholder_name" [t scope] {:prefix prefix }))

(defn remove-control-inputs 
  "Remove the control inputs cops from co.

  Warning: this function is directly manipulating the internals of the
  `tf.Graph`.

  Args:
    op: a `tf.Operation` from which to remove the control inputs.
    cops: an object convertible to a list of `tf.Operation`.
  Raises:
    TypeError: if op is not a `tf.Operation`.
    ValueError: if any cop in cops is not a control input of op.
  "
  [ op cops ]
  (py/call-attr graph-editor "remove_control_inputs"  op cops ))

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
  (py/call-attr graph-editor "replace_t_with_placeholder_handler"  info t ))

(defn reroute-inputs 
  "Re-route all the inputs of two subgraphs.

  Args:
    sgv0: the first subgraph to have its inputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    sgv1: the second subgraph to have its inputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
  Returns:
    A tuple `(sgv0, sgv1)` of subgraph views with their inputs swapped.
      Note that the function argument sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [ sgv0 sgv1 ]
  (py/call-attr graph-editor "reroute_inputs"  sgv0 sgv1 ))

(defn reroute-ios 
  "Re-route the inputs and outputs of sgv0 to sgv1 (see _reroute_sgv)."
  [ sgv0 sgv1 ]
  (py/call-attr graph-editor "reroute_ios"  sgv0 sgv1 ))

(defn reroute-outputs 
  "Re-route all the outputs of two operations.

  Args:
    sgv0: the first subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    sgv1: the second subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
  Returns:
    A tuple `(sgv0, sgv1)` of subgraph views with their outputs swapped.
      Note that the function argument sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  "
  [ sgv0 sgv1 ]
  (py/call-attr graph-editor "reroute_outputs"  sgv0 sgv1 ))

(defn reroute-ts 
  "For each tensor's pair, replace the end of t1 by the end of t0.

      B0 B1     B0 B1
      |  |    => |/
      A0 A1     A0 A1

  The end of the tensors in ts1 are left dangling.

  Args:
    ts0: an object convertible to a list of `tf.Tensor`.
    ts1: an object convertible to a list of `tf.Tensor`.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot be modified. Any
      operation within cannot_modify will be left untouched by this function.
  Returns:
    The number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  "
  [ ts0 ts1 can_modify cannot_modify ]
  (py/call-attr graph-editor "reroute_ts"  ts0 ts1 can_modify cannot_modify ))

(defn select-ops 
  "Helper to select operations.

  Args:
    *args: list of 1) regular expressions (compiled or not) or 2) (array of)
      `tf.Operation`. `tf.Tensor` instances are silently ignored.
    **kwargs: 'graph': `tf.Graph` in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if `positive_filter(elem)` is
        `True`. This is optional.
      'restrict_ops_regex': a regular expression is ignored if it doesn't start
        with the substring \"(?#ops)\".
  Returns:
    A list of `tf.Operation`.
  Raises:
    TypeError: if the optional keyword argument graph is not a `tf.Graph`
      or if an argument in args is not an (array of) `tf.Operation`
      or an (array of) `tf.Tensor` (silently ignored) or a string
      or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  "
  [  ]
  (py/call-attr graph-editor "select_ops"  ))

(defn select-ops-and-ts 
  "Helper to select operations and tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or 2) (array of)
      `tf.Operation` 3) (array of) tf.Tensor. Regular expressions matching
      tensors must start with the comment `\"(?#ts)\"`, for instance:
      `\"(?#ts)^foo/.*\"`.
    **kwargs: 'graph': `tf.Graph` in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if `positive_filter(elem)` is
        `True`. This is optional.
  Returns:
    A tuple `(ops, ts)` where:
      `ops` is a list of `tf.Operation`, and
      `ts` is a list of `tf.Tensor`
  Raises:
    TypeError: if the optional keyword argument graph is not a `tf.Graph`
      or if an argument in args is not an (array of) `tf.Tensor`
      or an (array of) `tf.Operation` or a string or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  "
  [  ]
  (py/call-attr graph-editor "select_ops_and_ts"  ))

(defn select-ts 
  "Helper to select tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or 2) (array of)
      `tf.Tensor`. `tf.Operation` instances are silently ignored.
    **kwargs: 'graph': `tf.Graph` in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if `positive_filter(elem)` is
        `True`. This is optional.
      'restrict_ts_regex': a regular expression is ignored if it doesn't start
        with the substring \"(?#ts)\".
  Returns:
    A list of `tf.Tensor`.
  Raises:
    TypeError: if the optional keyword argument graph is not a `tf.Graph`
      or if an argument in args is not an (array of) `tf.Tensor`
      or an (array of) `tf.Operation` (silently ignored) or a string
      or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  "
  [  ]
  (py/call-attr graph-editor "select_ts"  ))

(defn sgv 
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
  (py/call-attr graph-editor "sgv"  ))

(defn sgv-scope 
  "Make a subgraph from a name scope.

  Args:
    scope: the name of the scope.
    graph: the `tf.Graph`.
  Returns:
    A subgraph view representing the given scope.
  "
  [ scope graph ]
  (py/call-attr graph-editor "sgv_scope"  scope graph ))

(defn swap-inputs 
  "Swap all the inputs of sgv0 and sgv1 (see reroute_inputs)."
  [ sgv0 sgv1 ]
  (py/call-attr graph-editor "swap_inputs"  sgv0 sgv1 ))

(defn swap-ios 
  "Swap the inputs and outputs of sgv1 to sgv0 (see _reroute_sgv)."
  [ sgv0 sgv1 ]
  (py/call-attr graph-editor "swap_ios"  sgv0 sgv1 ))

(defn swap-outputs 
  "Swap all the outputs of sgv0 and sgv1 (see reroute_outputs)."
  [ sgv0 sgv1 ]
  (py/call-attr graph-editor "swap_outputs"  sgv0 sgv1 ))

(defn swap-ts 
  "For each tensor's pair, swap the end of (t0,t1).

      B0 B1     B0 B1
      |  |    =>  X
      A0 A1     A0 A1

  Args:
    ts0: an object convertible to a list of `tf.Tensor`.
    ts1: an object convertible to a list of `tf.Tensor`.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot be modified.
      Any operation within cannot_modify will be left untouched by this
      function.
  Returns:
    The number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  "
  [ ts0 ts1 can_modify cannot_modify ]
  (py/call-attr graph-editor "swap_ts"  ts0 ts1 can_modify cannot_modify ))
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
    (py/call-attr-kw graph-editor "transform_op_if_inside_handler" [info op] {:keep_if_possible keep_if_possible }))
