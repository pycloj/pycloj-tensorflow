(ns tensorflow-core.contrib.slim.VariableDeviceChooser
  "Device chooser for variables.

  When using a parameter server it will assign them in a round-robin fashion.
  When not using a parameter server it allows GPU or CPU placement.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce slim (import-module "tensorflow_core.contrib.slim"))

(defn VariableDeviceChooser 
  "Device chooser for variables.

  When using a parameter server it will assign them in a round-robin fashion.
  When not using a parameter server it allows GPU or CPU placement.
  "
  [ & {:keys [num_tasks job_name device_type device_index replica]
       :or {replica None}} ]
  
   (py/call-attr-kw slim "VariableDeviceChooser" [] {:num_tasks num_tasks :job_name job_name :device_type device_type :device_index device_index :replica replica }))
