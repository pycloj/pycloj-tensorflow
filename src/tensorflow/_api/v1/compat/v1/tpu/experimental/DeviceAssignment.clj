(ns tensorflow.-api.v1.compat.v1.tpu.experimental.DeviceAssignment
  "Mapping from logical cores in a computation to the physical TPU topology.

  Prefer to use the `DeviceAssignment.build()` helper to construct a
  `DeviceAssignment`; it is easier if less flexible than constructing a
  `DeviceAssignment` directly.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.tpu.experimental"))

(defn DeviceAssignment 
  "Mapping from logical cores in a computation to the physical TPU topology.

  Prefer to use the `DeviceAssignment.build()` helper to construct a
  `DeviceAssignment`; it is easier if less flexible than constructing a
  `DeviceAssignment` directly.
  "
  [ topology core_assignment ]
  (py/call-attr experimental "DeviceAssignment"  topology core_assignment ))
(defn build 
  ""
  [self topology computation_shape computation_stride  & {:keys [num_replicas]} ]
    (py/call-attr-kw self "build" [topology computation_shape computation_stride] {:num_replicas num_replicas }))

(defn coordinates 
  "Returns the physical topology coordinates of a logical core."
  [ self replica logical_core ]
  (py/call-attr self "coordinates"  self replica logical_core ))

(defn core-assignment 
  "The logical to physical core mapping.

    Returns:
      An integer numpy array of rank 3, with shape
      `[num_replicas, num_cores_per_replica, topology_rank]`. Maps
      (replica, logical core) pairs to physical topology coordinates.
    "
  [ self ]
    (py/call-attr self "core_assignment"))

(defn host-device 
  "Returns the CPU device attached to a logical core."
  [self  & {:keys [replica logical_core job]
                       :or {job None}} ]
    (py/call-attr-kw self "host_device" [] {:replica replica :logical_core logical_core :job job }))

(defn lookup-replicas 
  "Lookup replica ids by task number and logical core.

    Args:
      task_id: TensorFlow task number.
      logical_core: An integer, identifying a logical core.
    Returns:
      A sorted list of the replicas that are attached to that task and
      logical_core.
    Raises:
      ValueError: If no replica exists in the task which contains the logical
      core.
    "
  [ self task_id logical_core ]
  (py/call-attr self "lookup_replicas"  self task_id logical_core ))

(defn num-cores-per-replica 
  "The number of cores per replica."
  [ self ]
    (py/call-attr self "num_cores_per_replica"))

(defn num-replicas 
  "The number of replicas of the computation."
  [ self ]
    (py/call-attr self "num_replicas"))

(defn topology 
  "A `Topology` that describes the TPU topology."
  [ self ]
    (py/call-attr self "topology"))

(defn tpu-device 
  "Returns the name of the TPU device assigned to a logical core."
  [self  & {:keys [replica logical_core job]
                       :or {job None}} ]
    (py/call-attr-kw self "tpu_device" [] {:replica replica :logical_core logical_core :job job }))
(defn tpu-ordinal 
  "Returns the ordinal of the TPU device assigned to a logical core."
  [self   & {:keys [replica logical_core]} ]
    (py/call-attr-kw self "tpu_ordinal" [] {:replica replica :logical_core logical_core }))
