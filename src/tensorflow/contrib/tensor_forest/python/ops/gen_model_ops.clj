(ns tensorflow.contrib.tensor-forest.python.ops.gen-model-ops
  "Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_model_ops_py.cc
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gen-model-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.gen_model_ops"))

(defn CreateTreeVariable 
  "Creates a tree  model and returns a handle to it.

  Args:
    tree_handle: A `Tensor` of type `resource`.
      handle to the tree resource to be created.
    tree_config: A `Tensor` of type `string`. Serialized proto of the tree.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle tree_config params name ]
  (py/call-attr gen-model-ops "CreateTreeVariable"  tree_handle tree_config params name ))

(defn DecisionTreeResourceHandleOp 
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
  
   (py/call-attr-kw gen-model-ops "DecisionTreeResourceHandleOp" [] {:container container :shared_name shared_name :name name }))

(defn FeatureUsageCounts 
  "Outputs the number of times each feature was used in a split.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    `feature_counts[i]` is the number of times feature i was used
    in a split.
  "
  [ tree_handle params name ]
  (py/call-attr gen-model-ops "FeatureUsageCounts"  tree_handle params name ))

(defn TraverseTreeV4 
  "Outputs the leaf ids for the given input data.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    input_data: A `Tensor` of type `float32`.
      The training batch's features as a 2-d tensor; `input_data[i][j]`
      gives the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_spec: A `string`.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. `leaf_ids[i]` is the leaf id for input i.
  "
  [ tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ]
  (py/call-attr gen-model-ops "TraverseTreeV4"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ))

(defn TreeDeserialize 
  "Deserializes a serialized tree config and replaces current tree.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree .
    tree_config: A `Tensor` of type `string`. Serialized proto of the .
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle tree_config params name ]
  (py/call-attr gen-model-ops "TreeDeserialize"  tree_handle tree_config params name ))

(defn TreeIsInitializedOp 
  "Checks whether a tree has been initialized.

  Args:
    tree_handle: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ tree_handle name ]
  (py/call-attr gen-model-ops "TreeIsInitializedOp"  tree_handle name ))

(defn TreePredictionsV4 
  "Outputs the predictions for the given input data.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    input_data: A `Tensor` of type `float32`.
      The training batch's features as a 2-d tensor; `input_data[i][j]`
      gives the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_spec: A `string`.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (predictions, tree_paths).

    predictions: A `Tensor` of type `float32`. `predictions[i][j]` is the probability that input i is class j.
    tree_paths: A `Tensor` of type `string`. `tree_paths[i]` is a serialized TreePath proto for example i.
  "
  [ tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ]
  (py/call-attr gen-model-ops "TreePredictionsV4"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ))

(defn TreeSerialize 
  "Serializes the tree  to a proto.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Serialized proto of the tree.
  "
  [ tree_handle name ]
  (py/call-attr gen-model-ops "TreeSerialize"  tree_handle name ))

(defn TreeSize 
  "Outputs the size of the tree, including leaves.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. Size scalar.
  "
  [ tree_handle name ]
  (py/call-attr gen-model-ops "TreeSize"  tree_handle name ))

(defn UpdateModelV4 
  "Updates the given leaves for each example with the new labels.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    leaf_ids: A `Tensor` of type `int32`.
      `leaf_ids[i]` is the leaf id for input i.
    input_labels: A `Tensor` of type `float32`.
      The training batch's labels as a 1 or 2-d tensor.
      'input_labels[i][j]' gives the j-th label/target for the i-th input.
    input_weights: A `Tensor` of type `float32`.
      The training batch's weights as a 1-d tensor.
      'input_weights[i]' gives the weight for the i-th input.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle leaf_ids input_labels input_weights params name ]
  (py/call-attr gen-model-ops "UpdateModelV4"  tree_handle leaf_ids input_labels input_weights params name ))

(defn create-tree-variable 
  "Creates a tree  model and returns a handle to it.

  Args:
    tree_handle: A `Tensor` of type `resource`.
      handle to the tree resource to be created.
    tree_config: A `Tensor` of type `string`. Serialized proto of the tree.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle tree_config params name ]
  (py/call-attr gen-model-ops "create_tree_variable"  tree_handle tree_config params name ))

(defn create-tree-variable-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function create_tree_variable
  "
  [ tree_handle tree_config params name ctx ]
  (py/call-attr gen-model-ops "create_tree_variable_eager_fallback"  tree_handle tree_config params name ctx ))

(defn decision-tree-resource-handle-op 
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
  
   (py/call-attr-kw gen-model-ops "decision_tree_resource_handle_op" [] {:container container :shared_name shared_name :name name }))

(defn decision-tree-resource-handle-op-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function decision_tree_resource_handle_op
  "
  [ & {:keys [container shared_name name ctx]
       :or {name None ctx None}} ]
  
   (py/call-attr-kw gen-model-ops "decision_tree_resource_handle_op_eager_fallback" [] {:container container :shared_name shared_name :name name :ctx ctx }))

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
  (py/call-attr gen-model-ops "deprecated_endpoints"  ))

(defn feature-usage-counts 
  "Outputs the number of times each feature was used in a split.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    `feature_counts[i]` is the number of times feature i was used
    in a split.
  "
  [ tree_handle params name ]
  (py/call-attr gen-model-ops "feature_usage_counts"  tree_handle params name ))

(defn feature-usage-counts-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function feature_usage_counts
  "
  [ tree_handle params name ctx ]
  (py/call-attr gen-model-ops "feature_usage_counts_eager_fallback"  tree_handle params name ctx ))

(defn traverse-tree-v4 
  "Outputs the leaf ids for the given input data.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    input_data: A `Tensor` of type `float32`.
      The training batch's features as a 2-d tensor; `input_data[i][j]`
      gives the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_spec: A `string`.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. `leaf_ids[i]` is the leaf id for input i.
  "
  [ tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ]
  (py/call-attr gen-model-ops "traverse_tree_v4"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ))

(defn traverse-tree-v4-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function traverse_tree_v4
  "
  [ tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ctx ]
  (py/call-attr gen-model-ops "traverse_tree_v4_eager_fallback"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ctx ))

(defn tree-deserialize 
  "Deserializes a serialized tree config and replaces current tree.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree .
    tree_config: A `Tensor` of type `string`. Serialized proto of the .
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle tree_config params name ]
  (py/call-attr gen-model-ops "tree_deserialize"  tree_handle tree_config params name ))

(defn tree-deserialize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tree_deserialize
  "
  [ tree_handle tree_config params name ctx ]
  (py/call-attr gen-model-ops "tree_deserialize_eager_fallback"  tree_handle tree_config params name ctx ))

(defn tree-is-initialized-op 
  "Checks whether a tree has been initialized.

  Args:
    tree_handle: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ tree_handle name ]
  (py/call-attr gen-model-ops "tree_is_initialized_op"  tree_handle name ))

(defn tree-is-initialized-op-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tree_is_initialized_op
  "
  [ tree_handle name ctx ]
  (py/call-attr gen-model-ops "tree_is_initialized_op_eager_fallback"  tree_handle name ctx ))

(defn tree-predictions-v4 
  "Outputs the predictions for the given input data.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    input_data: A `Tensor` of type `float32`.
      The training batch's features as a 2-d tensor; `input_data[i][j]`
      gives the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_spec: A `string`.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (predictions, tree_paths).

    predictions: A `Tensor` of type `float32`. `predictions[i][j]` is the probability that input i is class j.
    tree_paths: A `Tensor` of type `string`. `tree_paths[i]` is a serialized TreePath proto for example i.
  "
  [ tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ]
  (py/call-attr gen-model-ops "tree_predictions_v4"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ))

(defn tree-predictions-v4-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tree_predictions_v4
  "
  [ tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ctx ]
  (py/call-attr gen-model-ops "tree_predictions_v4_eager_fallback"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ctx ))

(defn tree-serialize 
  "Serializes the tree  to a proto.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Serialized proto of the tree.
  "
  [ tree_handle name ]
  (py/call-attr gen-model-ops "tree_serialize"  tree_handle name ))

(defn tree-serialize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tree_serialize
  "
  [ tree_handle name ctx ]
  (py/call-attr gen-model-ops "tree_serialize_eager_fallback"  tree_handle name ctx ))

(defn tree-size 
  "Outputs the size of the tree, including leaves.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. Size scalar.
  "
  [ tree_handle name ]
  (py/call-attr gen-model-ops "tree_size"  tree_handle name ))

(defn tree-size-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tree_size
  "
  [ tree_handle name ctx ]
  (py/call-attr gen-model-ops "tree_size_eager_fallback"  tree_handle name ctx ))

(defn update-model-v4 
  "Updates the given leaves for each example with the new labels.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    leaf_ids: A `Tensor` of type `int32`.
      `leaf_ids[i]` is the leaf id for input i.
    input_labels: A `Tensor` of type `float32`.
      The training batch's labels as a 1 or 2-d tensor.
      'input_labels[i][j]' gives the j-th label/target for the i-th input.
    input_weights: A `Tensor` of type `float32`.
      The training batch's weights as a 1-d tensor.
      'input_weights[i]' gives the weight for the i-th input.
    params: A `string`. A serialized TensorForestParams proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ tree_handle leaf_ids input_labels input_weights params name ]
  (py/call-attr gen-model-ops "update_model_v4"  tree_handle leaf_ids input_labels input_weights params name ))

(defn update-model-v4-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function update_model_v4
  "
  [ tree_handle leaf_ids input_labels input_weights params name ctx ]
  (py/call-attr gen-model-ops "update_model_v4_eager_fallback"  tree_handle leaf_ids input_labels input_weights params name ctx ))
