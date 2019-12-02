(ns tensorflow.contrib.factorization.KMeans
  "Creates the graph for k-means clustering."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce factorization (import-module "tensorflow.contrib.factorization"))
(defn KMeans 
  "Creates the graph for k-means clustering."
  [inputs num_clusters  & {:keys [initial_clusters distance_metric use_mini_batch mini_batch_steps_per_iteration random_seed kmeans_plus_plus_num_retries kmc2_chain_length]} ]
    (py/call-attr-kw factorization "KMeans" [inputs num_clusters] {:initial_clusters initial_clusters :distance_metric distance_metric :use_mini_batch use_mini_batch :mini_batch_steps_per_iteration mini_batch_steps_per_iteration :random_seed random_seed :kmeans_plus_plus_num_retries kmeans_plus_plus_num_retries :kmc2_chain_length kmc2_chain_length }))

(defn training-graph 
  "Generate a training graph for kmeans algorithm.

    This returns, among other things, an op that chooses initial centers
    (init_op), a boolean variable that is set to True when the initial centers
    are chosen (cluster_centers_initialized), and an op to perform either an
    entire Lloyd iteration or a mini-batch of a Lloyd iteration (training_op).
    The caller should use these components as follows. A single worker should
    execute init_op multiple times until cluster_centers_initialized becomes
    True. Then multiple workers may execute training_op any number of times.

    Returns:
      A tuple consisting of:
      all_scores: A matrix (or list of matrices) of dimensions (num_input,
        num_clusters) where the value is the distance of an input vector and a
        cluster center.
      cluster_idx: A vector (or list of vectors). Each element in the vector
        corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      scores: Similar to cluster_idx but specifies the distance to the
        assigned cluster instead.
      cluster_centers_initialized: scalar indicating whether clusters have been
        initialized.
      init_op: an op to initialize the clusters.
      training_op: an op that runs an iteration of training.
    "
  [ self  ]
  (py/call-attr self "training_graph"  self  ))
