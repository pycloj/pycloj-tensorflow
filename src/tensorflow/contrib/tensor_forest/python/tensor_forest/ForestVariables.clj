(ns tensorflow.contrib.tensor-forest.python.tensor-forest.ForestVariables
  "A container for a forests training data, consisting of multiple trees.

  Instantiates a TreeVariables object for each tree. We override the
  __getitem__ and __setitem__ function so that usage looks like this:

    forest_variables = ForestVariables(params)

    ... forest_variables.tree ...
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

(defn ForestVariables 
  "A container for a forests training data, consisting of multiple trees.

  Instantiates a TreeVariables object for each tree. We override the
  __getitem__ and __setitem__ function so that usage looks like this:

    forest_variables = ForestVariables(params)

    ... forest_variables.tree ...
  "
  [params device_assigner & {:keys [training tree_variables_class tree_configs tree_stats]
                       :or {tree_configs None tree_stats None}} ]
    (py/call-attr-kw tensor-forest "ForestVariables" [params device_assigner] {:training training :tree_variables_class tree_variables_class :tree_configs tree_configs :tree_stats tree_stats }))
