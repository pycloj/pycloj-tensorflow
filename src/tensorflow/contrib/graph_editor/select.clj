(ns tensorflow.contrib.graph-editor.select
  "Various ways of selecting operations and tensors in a graph."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce select (import-module "tensorflow.contrib.graph_editor.select"))

(defn can-be-regex 
  "Return True if obj can be turned into a regular expression."
  [ obj ]
  (py/call-attr select "can_be_regex"  obj ))

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
  
   (py/call-attr-kw select "check_cios" [] {:control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

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
  (py/call-attr select "compute_boundary_ts"  ops ))

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
  (py/call-attr select "filter_ops"  ops positive_filter ))

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
  (py/call-attr select "filter_ops_from_regex"  ops regex ))

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
  (py/call-attr select "filter_ts"  ops positive_filter ))

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
  (py/call-attr select "filter_ts_from_regex"  ops regex ))

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
    (py/call-attr-kw select "get_backward_walk_ops" [seed_ops] {:inclusive inclusive :within_ops within_ops :within_ops_fn within_ops_fn :stop_at_ts stop_at_ts :control_inputs control_inputs }))

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
    (py/call-attr-kw select "get_forward_walk_ops" [seed_ops] {:inclusive inclusive :within_ops within_ops :within_ops_fn within_ops_fn :stop_at_ts stop_at_ts :control_outputs control_outputs }))

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
  (py/call-attr select "get_name_scope_ops"  ops scope ))

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
    (py/call-attr-kw select "get_ops_ios" [ops] {:control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

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
    (py/call-attr-kw select "get_walks_intersection_ops" [forward_seed_ops backward_seed_ops] {:forward_inclusive forward_inclusive :backward_inclusive backward_inclusive :within_ops within_ops :within_ops_fn within_ops_fn :control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

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
    (py/call-attr-kw select "get_walks_union_ops" [forward_seed_ops backward_seed_ops] {:forward_inclusive forward_inclusive :backward_inclusive backward_inclusive :within_ops within_ops :within_ops_fn within_ops_fn :control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

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
    (py/call-attr-kw select "get_within_boundary_ops" [ops seed_ops] {:boundary_ops boundary_ops :inclusive inclusive :control_inputs control_inputs :control_outputs control_outputs :control_ios control_ios }))

(defn iteritems 
  "Return an iterator over the (key, value) pairs of a dictionary."
  [ d ]
  (py/call-attr select "iteritems"  d ))

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
  (py/call-attr select "make_regex"  obj ))

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
  (py/call-attr select "select_ops"  ))

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
  (py/call-attr select "select_ops_and_ts"  ))

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
  (py/call-attr select "select_ts"  ))
