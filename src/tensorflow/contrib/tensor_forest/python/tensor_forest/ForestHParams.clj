(ns tensorflow.contrib.tensor-forest.python.tensor-forest.ForestHParams
  "A base class for holding hyperparameters and calculating good defaults."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensor-forest (import-module "tensorflow.contrib.tensor_forest.python.tensor_forest"))

(defn ForestHParams 
  "A base class for holding hyperparameters and calculating good defaults."
  [ & {:keys [num_trees max_nodes bagging_fraction num_splits_to_consider feature_bagging_fraction max_fertile_nodes split_after_samples valid_leaf_threshold dominate_method dominate_fraction model_name split_finish_name split_pruning_name prune_every_samples early_finish_check_every_samples collate_examples checkpoint_stats use_running_stats_method initialize_average_splits inference_tree_paths param_file split_name]
       :or {param_file None}} ]
  
   (py/call-attr-kw tensor-forest "ForestHParams" [] {:num_trees num_trees :max_nodes max_nodes :bagging_fraction bagging_fraction :num_splits_to_consider num_splits_to_consider :feature_bagging_fraction feature_bagging_fraction :max_fertile_nodes max_fertile_nodes :split_after_samples split_after_samples :valid_leaf_threshold valid_leaf_threshold :dominate_method dominate_method :dominate_fraction dominate_fraction :model_name model_name :split_finish_name split_finish_name :split_pruning_name split_pruning_name :prune_every_samples prune_every_samples :early_finish_check_every_samples early_finish_check_every_samples :collate_examples collate_examples :checkpoint_stats checkpoint_stats :use_running_stats_method use_running_stats_method :initialize_average_splits initialize_average_splits :inference_tree_paths inference_tree_paths :param_file param_file :split_name split_name }))

(defn fill 
  "Intelligently sets any non-specific parameters."
  [ self  ]
  (py/call-attr self "fill"  self  ))

(defn values 
  ""
  [ self  ]
  (py/call-attr self "values"  self  ))
