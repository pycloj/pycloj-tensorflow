(ns tensorflow.-api.v1.compat.v2.distribute.cluster-resolver.UnionResolver
  "Performs a union on underlying ClusterResolvers.

  This class performs a union given two or more existing ClusterResolvers. It
  merges the underlying ClusterResolvers, and returns one unified ClusterSpec
  when cluster_spec is called. The details of the merge function is
  documented in the cluster_spec function.

  For additional ClusterResolver properties such as task type, task index,
  rpc layer, environment, etc..., we will return the value from the first
  ClusterResolver in the union.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cluster-resolver (import-module "tensorflow._api.v1.compat.v2.distribute.cluster_resolver"))

(defn UnionResolver 
  "Performs a union on underlying ClusterResolvers.

  This class performs a union given two or more existing ClusterResolvers. It
  merges the underlying ClusterResolvers, and returns one unified ClusterSpec
  when cluster_spec is called. The details of the merge function is
  documented in the cluster_spec function.

  For additional ClusterResolver properties such as task type, task index,
  rpc layer, environment, etc..., we will return the value from the first
  ClusterResolver in the union.
  "
  [  ]
  (py/call-attr cluster-resolver "UnionResolver"  ))

(defn cluster-spec 
  "Returns a union of all the ClusterSpecs from the ClusterResolvers.

    Returns:
      A ClusterSpec containing host information merged from all the underlying
      ClusterResolvers.

    Raises:
      KeyError: If there are conflicting keys detected when merging two or
      more dictionaries, this exception is raised.

    Note: If there are multiple ClusterResolvers exposing ClusterSpecs with the
    same job name, we will merge the list/dict of workers.

    If *all* underlying ClusterSpecs expose the set of workers as lists, we will
    concatenate the lists of workers, starting with the list of workers from
    the first ClusterResolver passed into the constructor.

    If *any* of the ClusterSpecs expose the set of workers as a dict, we will
    treat all the sets of workers as dicts (even if they are returned as lists)
    and will only merge them into a dict if there is no conflicting keys. If
    there is a conflicting key, we will raise a `KeyError`.
    "
  [ self  ]
  (py/call-attr self "cluster_spec"  self  ))

(defn environment 
  ""
  [ self ]
    (py/call-attr self "environment"))

(defn master 
  "Returns the master address to use when creating a session.

    This usually returns the master from the first ClusterResolver passed in,
    but you can override this by specifying the task_type and task_id.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
    "
  [ self task_type task_id rpc_layer ]
  (py/call-attr self "master"  self task_type task_id rpc_layer ))

(defn num-accelerators 
  ""
  [ self task_type task_id config_proto ]
  (py/call-attr self "num_accelerators"  self task_type task_id config_proto ))

(defn rpc-layer 
  ""
  [ self ]
    (py/call-attr self "rpc_layer"))

(defn task-id 
  ""
  [ self ]
    (py/call-attr self "task_id"))

(defn task-type 
  ""
  [ self ]
    (py/call-attr self "task_type"))
