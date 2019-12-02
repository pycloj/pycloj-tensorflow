(ns tensorflow.contrib.tensor-forest.python.tensor-forest.RandomForestGraphs
  "Builds TF graphs for random forest training and inference."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensor-forest (import-module "tensorflow.contrib.tensor_forest.python.tensor_forest"))

(defn RandomForestGraphs 
  "Builds TF graphs for random forest training and inference."
  [params tree_configs tree_stats device_assigner variables & {:keys [tree_variables_class tree_graphs training]
                       :or {tree_graphs None}} ]
    (py/call-attr-kw tensor-forest "RandomForestGraphs" [params tree_configs tree_stats device_assigner variables] {:tree_variables_class tree_variables_class :tree_graphs tree_graphs :training training }))

(defn average-impurity 
  "Constructs a TF graph for evaluating the leaf impurity of a forest.

    Returns:
      The last op in the graph.
    "
  [ self  ]
  (py/call-attr self "average_impurity"  self  ))

(defn average-size 
  "Constructs a TF graph for evaluating the average size of a forest.

    Returns:
      The average number of nodes over the trees.
    "
  [ self  ]
  (py/call-attr self "average_size"  self  ))

(defn feature-importances 
  ""
  [ self  ]
  (py/call-attr self "feature_importances"  self  ))

(defn get-all-resource-handles 
  ""
  [ self  ]
  (py/call-attr self "get_all_resource_handles"  self  ))

(defn inference-graph 
  "Constructs a TF graph for evaluating a random forest.

    Args:
      input_data: A tensor or dict of string->Tensor for the input data. This
        input_data must generate the same spec as the
                  input_data used in training_graph:  the dict must have the
                    same keys, for example, and all tensors must have the same
                    size in their first dimension.
      **inference_args: Keyword arguments to pass through to each tree.

    Returns:
      A tuple of (probabilities, tree_paths, variance).

    Raises:
      NotImplementedError: If trying to use feature bagging with sparse
        features.
    "
  [ self input_data ]
  (py/call-attr self "inference_graph"  self input_data ))
(defn training-graph 
  "Constructs a TF graph for training a random forest.

    Args:
      input_data: A tensor or dict of string->Tensor for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.
      num_trainers: Number of parallel trainers to split trees among.
      trainer_id: Which trainer this instance is.
      **tree_kwargs: Keyword arguments passed to each tree's training_graph.

    Returns:
      The last op in the random forest training graph.

    Raises:
      NotImplementedError: If trying to use bagging with sparse features.
    "
  [self input_data input_labels  & {:keys [num_trainers trainer_id]} ]
    (py/call-attr-kw self "training_graph" [input_data input_labels] {:num_trainers num_trainers :trainer_id trainer_id }))
(defn training-loss 
  ""
  [self features labels  & {:keys [name]} ]
    (py/call-attr-kw self "training_loss" [features labels] {:name name }))

(defn validation-loss 
  ""
  [ self features labels ]
  (py/call-attr self "validation_loss"  self features labels ))
