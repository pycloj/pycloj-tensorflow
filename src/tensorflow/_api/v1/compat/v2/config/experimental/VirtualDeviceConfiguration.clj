(ns tensorflow.-api.v1.compat.v2.config.experimental.VirtualDeviceConfiguration
  "Configuration class for virtual devices for a PhysicalDevice.

  Fields:
    memory_limit: (optional) Maximum memory (in MB) to allocate on the virtual
      device. Currently only supported for GPUs.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.config.experimental"))

(defn VirtualDeviceConfiguration 
  "Configuration class for virtual devices for a PhysicalDevice.

  Fields:
    memory_limit: (optional) Maximum memory (in MB) to allocate on the virtual
      device. Currently only supported for GPUs.
  "
  [ memory_limit ]
  (py/call-attr experimental "VirtualDeviceConfiguration"  memory_limit ))

(defn memory-limit 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "memory_limit"))
