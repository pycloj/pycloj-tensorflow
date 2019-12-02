(ns tensorflow.-api.v1.compat.v2.tpu.experimental
  "Public API for tf.tpu.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.tpu.experimental"))

(defn initialize-tpu-system 
  "Initialize the TPU devices.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
  Returns:
    The tf.tpu.Topology object for the topology of the TPU cluster.

  Raises:
    RuntimeError: If no TPU devices found for eager execution.
  "
  [ cluster_resolver ]
  (py/call-attr experimental "initialize_tpu_system"  cluster_resolver ))
