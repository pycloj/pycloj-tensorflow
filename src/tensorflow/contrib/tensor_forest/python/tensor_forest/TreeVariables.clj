(ns tensorflow.contrib.tensor-forest.python.tensor-forest.TreeVariables
  "Stores tf.Variables for training a single random tree.

  Uses tf.compat.v1.get_variable to get tree-specific names so that this can be
  used
  with a tf.learn-style implementation (one that trains a model, saves it,
  then relies on restoring that model to evaluate).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensor-forest (import-module "tensorflow.contrib.tensor_forest.python.tensor_forest"))
(defn TreeVariables 
  "Stores tf.Variables for training a single random tree.

  Uses tf.compat.v1.get_variable to get tree-specific names so that this can be
  used
  with a tf.learn-style implementation (one that trains a model, saves it,
  then relies on restoring that model to evaluate).
  "
  [params tree_num training  & {:keys [tree_config tree_stat]} ]
    (py/call-attr-kw tensor-forest "TreeVariables" [params tree_num training] {:tree_config tree_config :tree_stat tree_stat }))

(defn get-tree-name 
  ""
  [ self name num ]
  (py/call-attr self "get_tree_name"  self name num ))
