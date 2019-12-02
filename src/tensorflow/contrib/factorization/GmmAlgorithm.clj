(ns tensorflow.contrib.factorization.GmmAlgorithm
  "Tensorflow Gaussian mixture model clustering class."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce factorization (import-module "tensorflow.contrib.factorization"))
(defn GmmAlgorithm 
  "Tensorflow Gaussian mixture model clustering class."
  [data num_classes initial_means  & {:keys [params covariance_type random_seed]} ]
    (py/call-attr-kw factorization "GmmAlgorithm" [data num_classes initial_means] {:params params :covariance_type covariance_type :random_seed random_seed }))

(defn alphas 
  ""
  [ self  ]
  (py/call-attr self "alphas"  self  ))

(defn assignments 
  "Returns a list of Tensors with the matrix of assignments per shard."
  [ self  ]
  (py/call-attr self "assignments"  self  ))

(defn clusters 
  "Returns the clusters with dimensions num_classes X 1 X num_dimensions."
  [ self  ]
  (py/call-attr self "clusters"  self  ))

(defn covariances 
  "Returns the covariances matrices."
  [ self  ]
  (py/call-attr self "covariances"  self  ))

(defn init-ops 
  "Returns the initialization operation."
  [ self  ]
  (py/call-attr self "init_ops"  self  ))

(defn is-initialized 
  "Returns a boolean operation for initialized variables."
  [ self  ]
  (py/call-attr self "is_initialized"  self  ))

(defn log-likelihood-op 
  "Returns the log-likelihood operation."
  [ self  ]
  (py/call-attr self "log_likelihood_op"  self  ))

(defn scores 
  "Returns the per-sample likelihood fo the data.

    Returns:
      Log probabilities of each data point.
    "
  [ self  ]
  (py/call-attr self "scores"  self  ))

(defn training-ops 
  "Returns the training operation."
  [ self  ]
  (py/call-attr self "training_ops"  self  ))
