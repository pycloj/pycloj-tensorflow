(ns tensorflow.contrib.tensor-forest.python.ops.model-ops
  "Model ops python wrappers."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce model-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.model_ops"))

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
  (py/call-attr model-ops "feature_usage_counts"  tree_handle params name ))

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
  (py/call-attr model-ops "traverse_tree_v4"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ))

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
  (py/call-attr model-ops "tree_predictions_v4"  tree_handle input_data sparse_input_indices sparse_input_values sparse_input_shape input_spec params name ))

(defn tree-size 
  "Outputs the size of the tree, including leaves.

  Args:
    tree_handle: A `Tensor` of type `resource`. The handle to the tree.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. Size scalar.
  "
  [ tree_handle name ]
  (py/call-attr model-ops "tree_size"  tree_handle name ))

(defn tree-variable 
  "Creates a tree model and returns a handle to it.

  Args:
    params: A TensorForestParams object.
    tree_config: A `Tensor` of type `string`. Serialized proto of the tree.
    stats_handle: Resource handle to the stats object.
    name: A name for the variable.
    container: An optional `string`. Defaults to `\"\"`.

  Returns:
    A `Tensor` of type mutable `string`. The handle to the tree.
  "
  [ params tree_config stats_handle name container ]
  (py/call-attr model-ops "tree_variable"  params tree_config stats_handle name container ))

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
  (py/call-attr model-ops "update_model_v4"  tree_handle leaf_ids input_labels input_weights params name ))
