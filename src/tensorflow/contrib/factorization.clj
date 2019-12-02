(ns tensorflow.contrib.factorization
  "Ops and modules related to factorization."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce factorization (import-module "tensorflow.contrib.factorization"))
(defn gmm 
  "Creates the graph for Gaussian mixture model (GMM) clustering.

  Args:
    inp: An input tensor or list of input tensors
    initial_clusters: Specifies the clusters used during
      initialization. Can be a tensor or numpy array, or a function
      that generates the clusters. Can also be \"random\" to specify
      that clusters should be chosen randomly from input data. Note: type
      is diverse to be consistent with skflow.
    num_clusters: number of clusters.
    random_seed: Python integer. Seed for PRNG used to initialize centers.
    covariance_type: one of \"diag\", \"full\".
    params: Controls which parameters are updated in the training
      process. Can contain any combination of \"w\" for weights, \"m\" for
      means, and \"c\" for covars.

  Returns:
    Note: tuple of lists returned to be consistent with skflow
    A tuple consisting of:
    assignments: A vector (or list of vectors). Each element in the vector
      corresponds to an input row in 'inp' and specifies the cluster id
      corresponding to the input.
    training_op: an op that runs an iteration of training.
    init_op: an op that runs the initialization.
  "
  [inp initial_clusters num_clusters random_seed  & {:keys [covariance_type params]} ]
    (py/call-attr-kw factorization "gmm" [inp initial_clusters num_clusters random_seed] {:covariance_type covariance_type :params params }))
