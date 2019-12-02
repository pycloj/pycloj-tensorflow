(ns tensorflow.contrib.tpu.Topology
  "Describes a set of TPU devices.

  Represents both the shape of the physical mesh, and the mapping between
  TensorFlow TPU devices to physical mesh coordinates.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow.contrib.tpu"))

(defn Topology 
  "Describes a set of TPU devices.

  Represents both the shape of the physical mesh, and the mapping between
  TensorFlow TPU devices to physical mesh coordinates.
  "
  [ serialized mesh_shape device_coordinates ]
  (py/call-attr tpu "Topology"  serialized mesh_shape device_coordinates ))

(defn cpu-device-name-at-coordinates 
  "Returns the CPU device attached to a logical core."
  [ self device_coordinates job ]
  (py/call-attr self "cpu_device_name_at_coordinates"  self device_coordinates job ))

(defn device-coordinates 
  "Describes the mapping from TPU devices to topology coordinates.

    Returns:
      A rank 3 int32 array with shape `[tasks, devices, axis]`.
      `tasks` is the number of tasks in the TPU cluster, `devices` is the number
      of TPU devices per task, and `axis` is the number of axes in the TPU
      cluster topology. Each entry gives the `axis`-th coordinate in the
      topology of a task/device pair. TPU topologies are 3-dimensional, with
      dimensions `(x, y, core number)`.
    "
  [ self ]
    (py/call-attr self "device_coordinates"))

(defn mesh-rank 
  "Returns the number of dimensions in the mesh."
  [ self ]
    (py/call-attr self "mesh_rank"))

(defn mesh-shape 
  "A rank 1 int32 array describing the shape of the TPU topology."
  [ self ]
    (py/call-attr self "mesh_shape"))

(defn num-tasks 
  "Returns the number of TensorFlow tasks in the TPU slice."
  [ self ]
    (py/call-attr self "num_tasks"))

(defn num-tpus-per-task 
  "Returns the number of TPU devices per task in the TPU slice."
  [ self ]
    (py/call-attr self "num_tpus_per_task"))

(defn serialized 
  "Returns the serialized form of the topology."
  [ self  ]
  (py/call-attr self "serialized"  self  ))

(defn task-ordinal-at-coordinates 
  "Returns the TensorFlow task number attached to `device_coordinates`.

    Args:
      device_coordinates: An integer sequence describing a device's physical
        coordinates in the TPU fabric.

    Returns:
      Returns the TensorFlow task number that contains the TPU device with those
      physical coordinates.
    "
  [ self device_coordinates ]
  (py/call-attr self "task_ordinal_at_coordinates"  self device_coordinates ))

(defn tpu-device-name-at-coordinates 
  "Returns the name of the TPU device assigned to a logical core."
  [ self device_coordinates job ]
  (py/call-attr self "tpu_device_name_at_coordinates"  self device_coordinates job ))

(defn tpu-device-ordinal-at-coordinates 
  "Returns the TensorFlow device number at `device_coordinates`.

    Args:
      device_coordinates: An integer sequence describing a device's physical
        coordinates in the TPU fabric.

    Returns:
      Returns the TensorFlow device number within the task corresponding to
      attached to the device with those physical coordinates.
    "
  [ self device_coordinates ]
  (py/call-attr self "tpu_device_ordinal_at_coordinates"  self device_coordinates ))
