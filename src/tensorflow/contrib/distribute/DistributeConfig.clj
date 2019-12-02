(ns tensorflow.contrib.distribute.DistributeConfig
  "A config tuple for distribution strategies.

  Attributes:
    train_distribute: a `DistributionStrategy` object for training.
    eval_distribute: an optional `DistributionStrategy` object for
      evaluation.
    remote_cluster: a dict, `ClusterDef` or `ClusterSpec` object specifying
      the cluster configurations. If this is given, the `train_and_evaluate`
      method will be running as a standalone client which connects to the
      cluster for training.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.contrib.distribute"))

(defn DistributeConfig 
  "A config tuple for distribution strategies.

  Attributes:
    train_distribute: a `DistributionStrategy` object for training.
    eval_distribute: an optional `DistributionStrategy` object for
      evaluation.
    remote_cluster: a dict, `ClusterDef` or `ClusterSpec` object specifying
      the cluster configurations. If this is given, the `train_and_evaluate`
      method will be running as a standalone client which connects to the
      cluster for training.
  "
  [ train_distribute eval_distribute remote_cluster ]
  (py/call-attr distribute "DistributeConfig"  train_distribute eval_distribute remote_cluster ))

(defn eval-distribute 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "eval_distribute"))

(defn remote-cluster 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "remote_cluster"))

(defn train-distribute 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "train_distribute"))
