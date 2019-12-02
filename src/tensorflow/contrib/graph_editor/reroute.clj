(ns tensorflow.contrib.graph-editor.reroute
  "Various function for graph rerouting."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce reroute (import-module "tensorflow.contrib.graph_editor.reroute"))

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
  (py/call-attr reroute "add_control_inputs"  op cops ))

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
  (py/call-attr reroute "remove_control_inputs"  op cops ))

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
  (py/call-attr reroute "reroute_inputs"  sgv0 sgv1 ))

(defn reroute-ios 
  "Re-route the inputs and outputs of sgv0 to sgv1 (see _reroute_sgv)."
  [ sgv0 sgv1 ]
  (py/call-attr reroute "reroute_ios"  sgv0 sgv1 ))

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
  (py/call-attr reroute "reroute_outputs"  sgv0 sgv1 ))

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
  (py/call-attr reroute "reroute_ts"  ts0 ts1 can_modify cannot_modify ))

(defn swap-inputs 
  "Swap all the inputs of sgv0 and sgv1 (see reroute_inputs)."
  [ sgv0 sgv1 ]
  (py/call-attr reroute "swap_inputs"  sgv0 sgv1 ))

(defn swap-ios 
  "Swap the inputs and outputs of sgv1 to sgv0 (see _reroute_sgv)."
  [ sgv0 sgv1 ]
  (py/call-attr reroute "swap_ios"  sgv0 sgv1 ))

(defn swap-outputs 
  "Swap all the outputs of sgv0 and sgv1 (see reroute_outputs)."
  [ sgv0 sgv1 ]
  (py/call-attr reroute "swap_outputs"  sgv0 sgv1 ))

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
  (py/call-attr reroute "swap_ts"  ts0 ts1 can_modify cannot_modify ))
