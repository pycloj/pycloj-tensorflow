(ns tensorflow.contrib.tensor-forest.python.tensor-forest.RandomTreeGraphs
  "Builds TF graphs for random tree training and inference."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensor-forest (import-module "tensorflow.contrib.tensor_forest.python.tensor_forest"))

(defn RandomTreeGraphs 
  "Builds TF graphs for random tree training and inference."
  [ variables params tree_num ]
  (py/call-attr tensor-forest "RandomTreeGraphs"  variables params tree_num ))

(defn feature-usage-counts 
  ""
  [ self  ]
  (py/call-attr self "feature_usage_counts"  self  ))

(defn inference-graph 
  "Constructs a TF graph for evaluating a random tree.

    Args:
      input_data: A tensor or placeholder for input data.
      data_spec: A TensorForestDataSpec proto specifying the original input
        columns.
      sparse_features: A tf.SparseTensor for sparse input data.

    Returns:
      A tuple of (probabilities, tree_paths).
    "
  [ self input_data data_spec sparse_features ]
  (py/call-attr self "inference_graph"  self input_data data_spec sparse_features ))

(defn size 
  "Constructs a TF graph for evaluating the current number of nodes.

    Returns:
      The current number of nodes in the tree.
    "
  [ self  ]
  (py/call-attr self "size"  self  ))

(defn training-graph 
  "Constructs a TF graph for training a random tree.

    Args:
      input_data: A tensor or placeholder for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.
      random_seed: The random number generator seed to use for this tree.  0
        means use the current time as the seed.
      data_spec: A data_ops.TensorForestDataSpec object specifying the original
        feature/columns of the data.
      sparse_features: A tf.SparseTensor for sparse input data.
      input_weights: A float tensor or placeholder holding per-input weights, or
        None if all inputs are to be weighted equally.

    Returns:
      The last op in the random tree training graph.
    "
  [ self input_data input_labels random_seed data_spec sparse_features input_weights ]
  (py/call-attr self "training_graph"  self input_data input_labels random_seed data_spec sparse_features input_weights ))
