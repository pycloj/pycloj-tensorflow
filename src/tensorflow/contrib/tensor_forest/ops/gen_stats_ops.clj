(ns tensorflow.contrib.tensor-forest.python.ops.gen-stats-ops
  "Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_stats_ops_py.cc
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gen-stats-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.gen_stats_ops"))

(defn CreateFertileStatsVariable 
  "Creates a stats model and returns a handle to it.

  Args:
    stats_handle: A `Tensor` of type `resource`.
      handle to the stats resource to be created.
    stats_config: A `Tensor` of type `string`. Serialized proto of the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ stats_handle stats_config params name ]
  (py/call-attr gen-stats-ops "CreateFertileStatsVariable"  stats_handle stats_config params name ))

(defn FertileStatsDeserialize 
  "Deserializes a serialized stats config and replaces current stats.

  Args:
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    stats_config: A `Tensor` of type `string`. Serialized proto of the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ stats_handle stats_config params name ]
  (py/call-attr gen-stats-ops "FertileStatsDeserialize"  stats_handle stats_config params name ))

(defn FertileStatsIsInitializedOp 
  "Checks whether a stats has been initialized.

  Args:
    stats_handle: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ stats_handle name ]
  (py/call-attr gen-stats-ops "FertileStatsIsInitializedOp"  stats_handle name ))

(defn FertileStatsResourceHandleOp 
  "TODO: add doc.

  Args:
    container: An optional `string`. Defaults to `\"\"`.
    shared_name: An optional `string`. Defaults to `\"\"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  "
  [ & {:keys [container shared_name name]
       :or {name None}} ]
  
   (py/call-attr-kw gen-stats-ops "FertileStatsResourceHandleOp" [] {:container container :shared_name shared_name :name name }))

(defn FertileStatsSerialize 
  "Serializes the stats to a proto.

  Args:
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Serialized proto of the stats.
  "
  [ stats_handle params name ]
  (py/call-attr gen-stats-ops "FertileStatsSerialize"  stats_handle params name ))

(defn FinalizeTree 
  "Puts the Leaf models inside the tree into their final form.

  If drop_final_class is true, the per-class probability prediction of the
  last class is not stored in the leaf models.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle stats_handle params name ]
  (py/call-attr gen-stats-ops "FinalizeTree"  tree_handle stats_handle params name ))

(defn GrowTreeV4 
  "Grows the tree for finished nodes and allocates waiting nodes.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    finished_nodes: A `Tensor` of type `int32`.
      A 1-d Tensor of finished node ids from ProcessInput.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle stats_handle finished_nodes params name ]
  (py/call-attr gen-stats-ops "GrowTreeV4"  tree_handle stats_handle finished_nodes params name ))

(defn ProcessInputV4 
  "Add labels to stats after traversing the tree for each example.

  Outputs node ids that are finished.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    input_data: A `Tensor` of type `float32`.
      The training batch's features as a 2-d tensor; `input_data[i][j]`
      gives the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_labels: A `Tensor` of type `float32`.
      The training batch's labels as a 1 or 2-d tensor.
      'input_labels[i][j]' gives the j-th label/target for the i-th input.
    input_weights: A `Tensor` of type `float32`.
      The training batch's weights as a 1-d tensor.
      'input_weights[i]' gives the weight for the i-th input.
    leaf_ids: A `Tensor` of type `int32`.
      `leaf_ids[i]` is the leaf id for input i.
    random_seed: An `int`.
    input_spec: A `string`.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    A 1-d tensor of node ids that have finished and are ready to
    grow.
  "
  [ tree_handle stats_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_labels input_weights leaf_ids random_seed input_spec params name ]
  (py/call-attr gen-stats-ops "ProcessInputV4"  tree_handle stats_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_labels input_weights leaf_ids random_seed input_spec params name ))

(defn create-fertile-stats-variable 
  "Creates a stats model and returns a handle to it.

  Args:
    stats_handle: A `Tensor` of type `resource`.
      handle to the stats resource to be created.
    stats_config: A `Tensor` of type `string`. Serialized proto of the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ stats_handle stats_config params name ]
  (py/call-attr gen-stats-ops "create_fertile_stats_variable"  stats_handle stats_config params name ))

(defn create-fertile-stats-variable-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function create_fertile_stats_variable
  "
  [ stats_handle stats_config params name ctx ]
  (py/call-attr gen-stats-ops "create_fertile_stats_variable_eager_fallback"  stats_handle stats_config params name ctx ))

(defn deprecated-endpoints 
  "Decorator for marking endpoints deprecated.

  This decorator does not print deprecation messages.
  TODO(annarev): eventually start printing deprecation warnings when
  @deprecation_endpoints decorator is added.

  Args:
    *args: Deprecated endpoint names.

  Returns:
    A function that takes symbol as an argument and adds
    _tf_deprecated_api_names to that symbol.
    _tf_deprecated_api_names would be set to a list of deprecated
    endpoint names for the symbol.
  "
  [  ]
  (py/call-attr gen-stats-ops "deprecated_endpoints"  ))

(defn fertile-stats-deserialize 
  "Deserializes a serialized stats config and replaces current stats.

  Args:
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    stats_config: A `Tensor` of type `string`. Serialized proto of the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ stats_handle stats_config params name ]
  (py/call-attr gen-stats-ops "fertile_stats_deserialize"  stats_handle stats_config params name ))

(defn fertile-stats-deserialize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fertile_stats_deserialize
  "
  [ stats_handle stats_config params name ctx ]
  (py/call-attr gen-stats-ops "fertile_stats_deserialize_eager_fallback"  stats_handle stats_config params name ctx ))

(defn fertile-stats-is-initialized-op 
  "Checks whether a stats has been initialized.

  Args:
    stats_handle: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ stats_handle name ]
  (py/call-attr gen-stats-ops "fertile_stats_is_initialized_op"  stats_handle name ))

(defn fertile-stats-is-initialized-op-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fertile_stats_is_initialized_op
  "
  [ stats_handle name ctx ]
  (py/call-attr gen-stats-ops "fertile_stats_is_initialized_op_eager_fallback"  stats_handle name ctx ))

(defn fertile-stats-resource-handle-op 
  "TODO: add doc.

  Args:
    container: An optional `string`. Defaults to `\"\"`.
    shared_name: An optional `string`. Defaults to `\"\"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  "
  [ & {:keys [container shared_name name]
       :or {name None}} ]
  
   (py/call-attr-kw gen-stats-ops "fertile_stats_resource_handle_op" [] {:container container :shared_name shared_name :name name }))

(defn fertile-stats-resource-handle-op-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fertile_stats_resource_handle_op
  "
  [ & {:keys [container shared_name name ctx]
       :or {name None ctx None}} ]
  
   (py/call-attr-kw gen-stats-ops "fertile_stats_resource_handle_op_eager_fallback" [] {:container container :shared_name shared_name :name name :ctx ctx }))

(defn fertile-stats-serialize 
  "Serializes the stats to a proto.

  Args:
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Serialized proto of the stats.
  "
  [ stats_handle params name ]
  (py/call-attr gen-stats-ops "fertile_stats_serialize"  stats_handle params name ))

(defn fertile-stats-serialize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fertile_stats_serialize
  "
  [ stats_handle params name ctx ]
  (py/call-attr gen-stats-ops "fertile_stats_serialize_eager_fallback"  stats_handle params name ctx ))

(defn finalize-tree 
  "Puts the Leaf models inside the tree into their final form.

  If drop_final_class is true, the per-class probability prediction of the
  last class is not stored in the leaf models.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle stats_handle params name ]
  (py/call-attr gen-stats-ops "finalize_tree"  tree_handle stats_handle params name ))

(defn finalize-tree-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function finalize_tree
  "
  [ tree_handle stats_handle params name ctx ]
  (py/call-attr gen-stats-ops "finalize_tree_eager_fallback"  tree_handle stats_handle params name ctx ))

(defn grow-tree-v4 
  "Grows the tree for finished nodes and allocates waiting nodes.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    finished_nodes: A `Tensor` of type `int32`.
      A 1-d Tensor of finished node ids from ProcessInput.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle stats_handle finished_nodes params name ]
  (py/call-attr gen-stats-ops "grow_tree_v4"  tree_handle stats_handle finished_nodes params name ))

(defn grow-tree-v4-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function grow_tree_v4
  "
  [ tree_handle stats_handle finished_nodes params name ctx ]
  (py/call-attr gen-stats-ops "grow_tree_v4_eager_fallback"  tree_handle stats_handle finished_nodes params name ctx ))

(defn process-input-v4 
  "Add labels to stats after traversing the tree for each example.

  Outputs node ids that are finished.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    stats_handle: A `Tensor` of type `resource`. The handle to the stats.
    input_data: A `Tensor` of type `float32`.
      The training batch's features as a 2-d tensor; `input_data[i][j]`
      gives the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_labels: A `Tensor` of type `float32`.
      The training batch's labels as a 1 or 2-d tensor.
      'input_labels[i][j]' gives the j-th label/target for the i-th input.
    input_weights: A `Tensor` of type `float32`.
      The training batch's weights as a 1-d tensor.
      'input_weights[i]' gives the weight for the i-th input.
    leaf_ids: A `Tensor` of type `int32`.
      `leaf_ids[i]` is the leaf id for input i.
    random_seed: An `int`.
    input_spec: A `string`.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    A 1-d tensor of node ids that have finished and are ready to
    grow.
  "
  [ tree_handle stats_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_labels input_weights leaf_ids random_seed input_spec params name ]
  (py/call-attr gen-stats-ops "process_input_v4"  tree_handle stats_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_labels input_weights leaf_ids random_seed input_spec params name ))

(defn process-input-v4-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function process_input_v4
  "
  [ tree_handle stats_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_labels input_weights leaf_ids random_seed input_spec params name ctx ]
  (py/call-attr gen-stats-ops "process_input_v4_eager_fallback"  tree_handle stats_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_labels input_weights leaf_ids random_seed input_spec params name ctx ))
