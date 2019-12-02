(ns tensorflow.-api.v1.compat.v2.config.experimental
  "Public API for tf.config.experimental namespace.
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

(defn get-device-policy 
  "Gets the current device policy.

  The device policy controls how operations requiring inputs on a specific
  device (e.g., on GPU:0) handle inputs on a different device (e.g. GPU:1).

  This function only gets the device policy for the current thread. Any
  subsequently started thread will again use the default policy.

  Returns:
    Current thread device policy
  "
  [  ]
  (py/call-attr experimental "get_device_policy"  ))

(defn get-memory-growth 
  "Get if memory growth is enabled for a PhysicalDevice.

  A PhysicalDevice with memory growth set will not allocate all memory on the
  device upfront.

  For example:

  ```python
  physical_devices = config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, \"Not enough GPU hardware devices available\"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True
  ```

  Args:
    device: PhysicalDevice to query

  Returns:
    Current memory growth setting.
  "
  [ device ]
  (py/call-attr experimental "get_memory_growth"  device ))

(defn get-synchronous-execution 
  "Gets whether operations are executed synchronously or asynchronously.

  TensorFlow can execute operations synchronously or asynchronously. If
  asynchronous execution is enabled, operations may return \"non-ready\" handles.

  Returns:
    Current thread execution mode
  "
  [  ]
  (py/call-attr experimental "get_synchronous_execution"  ))

(defn get-virtual-device-configuration 
  "Get the virtual device configuration for a PhysicalDevice.

  Returns the list of VirtualDeviceConfiguration objects previously configured
  by a call to `tf.config.experimental.set_virtual_device_configuration()`.

  For example:

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('CPU')
  assert len(physical_devices) == 1, \"No CPUs found\"
  configs = tf.config.experimental.get_virtual_device_configuration(
      physical_devices[0])
  assert configs is None
  tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration(),
       tf.config.experimental.VirtualDeviceConfiguration()])
  configs = tf.config.experimental.get_virtual_device_configuration(
      physical_devices[0])
  assert len(configs) == 2
  ```

  Args:
    device: PhysicalDevice to query

  Returns:
    List of `tf.config.experimental.VirtualDeviceConfiguration` objects or
    `None` if no virtual device configuration has been set for this physical
    device.
  "
  [ device ]
  (py/call-attr experimental "get_virtual_device_configuration"  device ))

(defn get-visible-devices 
  "Get the list of visible physical devices.

  Returns a list of PhysicalDevice objects that are current marked as visible to
  the runtime. Any visible devices will have LogicalDevices assigned to them
  once the runtime is initialized.

  The following example verifies all visible GPUs have been disabled:

  ```python
  physical_devices = config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, \"Not enough GPU hardware devices available\"
  # Disable all GPUS
  tf.config.experimental.set_visible_devices([], 'GPU')
  visible_devices = tf.config.experimental.get_visible_devices()
  for device in visible_devices:
    assert device.device_type != 'GPU'
  ```

  Args:
    device_type: (optional) Device types to limit query to.

  Returns:
    List of PhysicalDevice objects
  "
  [ device_type ]
  (py/call-attr experimental "get_visible_devices"  device_type ))

(defn list-logical-devices 
  "Return a list of logical devices created by runtime.

  Logical devices may correspond to physical devices or remote devices in the
  cluster. Operations and tensors may be placed on these devices by using the
  `name` of the LogicalDevice.

  For example:

  ```python
  logical_devices = tf.config.experimental.list_logical_devices('GPU')
  # Allocate on GPU:0
  with tf.device(logical_devices[0].name):
    one = tf.constant(1)
  # Allocate on GPU:1
  with tf.device(logical_devices[1].name):
    two = tf.constant(2)
  ```

  Args:
    device_type: (optional) Device type to filter by such as \"CPU\" or \"GPU\"

  Returns:
    List of LogicalDevice objects
  "
  [ device_type ]
  (py/call-attr experimental "list_logical_devices"  device_type ))

(defn list-physical-devices 
  "Return a list of physical devices visible to the runtime.

  Physical devices are hardware devices locally present on the current machine.
  By default all discovered CPU and GPU devices are considered visible. The
  `list_physical_devices` allows querying the hardware prior to runtime
  initialization.

  The following example ensures the machine can see at least 1 GPU.

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, \"No GPUs found.\"
  ```

  Args:
    device_type: (optional) Device type to filter by such as \"CPU\" or \"GPU\"

  Returns:
    List of PhysicalDevice objects
  "
  [ device_type ]
  (py/call-attr experimental "list_physical_devices"  device_type ))

(defn set-device-policy 
  "Sets the current thread device policy.

  The device policy controls how operations requiring inputs on a specific
  device (e.g., on GPU:0) handle inputs on a different device (e.g. GPU:1).

  When using the default, an appropriate policy will be picked automatically.
  The default policy may change over time.

  This function only sets the device policy for the current thread. Any
  subsequently started thread will again use the default policy.

  Args:
    device_policy: A device policy.
      Valid values:
      - None: Switch to a system default.
      - 'warn': Copies the tensors which are not on the right device and logs
          a warning.
      - 'explicit': Raises an error if the placement is not as required.
      - 'silent': Silently copies the tensors. Note that this may hide
          performance problems as there is no notification provided when
          operations are blocked on the tensor being copied between devices.
      - 'silent_for_int32': silently copies `int32` tensors, raising errors on
          the other ones.

  Raises:
      ValueError: If an invalid `device_policy` is passed.
  "
  [ device_policy ]
  (py/call-attr experimental "set_device_policy"  device_policy ))

(defn set-memory-growth 
  "Set if memory growth should be enabled for a PhysicalDevice.

  A PhysicalDevice with memory growth set will not allocate all memory on the
  device upfront. Memory growth cannot be configured on a PhysicalDevice with
  virtual devices configured.

  For example:

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, \"Not enough GPU hardware devices available\"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  ```

  Args:
    device: PhysicalDevice to configure
    enable: Whether to enable or disable memory growth
  "
  [ device enable ]
  (py/call-attr experimental "set_memory_growth"  device enable ))

(defn set-synchronous-execution 
  "Specifies whether operations are executed synchronously or asynchronously.

  TensorFlow can execute operations synchronously or asynchronously. If
  asynchronous execution is enabled, operations may return \"non-ready\" handles.

  When `enable` is set to None, an appropriate value will be picked
  automatically. The value picked may change between TensorFlow releases.

  Args:
    enable: Whether operations should be dispatched synchronously.
      Valid values:
      - None: sets the system default.
      - True: executes each operation synchronously.
      - False: executes each operation asynchronously.
  "
  [ enable ]
  (py/call-attr experimental "set_synchronous_execution"  enable ))

(defn set-virtual-device-configuration 
  "Set the virtual device configuration for a PhysicalDevice.

  A PhysicalDevice marked as visible will by default have a single LogicalDevice
  allocated to it once the runtime is configured. Specifying a list of
  tf.config.experimental.VirtualDeviceConfiguration objects allows multiple
  devices to be configured that utilize the same PhysicalDevice.

  The following example splits the CPU into 2 virtual devices:

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('CPU')
  assert len(physical_devices) == 1, \"No CPUs found\"
  # Specify 2 virtual CPUs. Note currently memory limit is not supported.
  tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(),
     tf.config.experimental.VirtualDeviceConfiguration()])
  logical_devices = tf.config.experimental.list_logical_devices('CPU')
  assert len(logical_devices) == 2

  try:
    tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration(),
       tf.config.experimental.VirtualDeviceConfiguration(),
       tf.config.experimental.VirtualDeviceConfiguration(),
       tf.config.experimental.VirtualDeviceConfiguration()])
  except:
    print('Cannot modify the virtual devices once they have been initialized.')
  ```

  The following example splits the GPU into 2 virtual devices with 100 MB each:

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, \"No GPUs found\"
  tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])

  try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
  except:
    print('Cannot set memory growth when virtual devices configured')

  logical_devices = tf.config.experimental.list_logical_devices('GPU')
  assert len(logical_devices) == len(physical_devices) + 1

  try:
    tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10),
       tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10)])
  except:
    print('Cannot modify the virtual devices once they have been initialized.')
  ```

  Args:
    device: (optional) Need to update
    virtual_devices: (optional) Need to update
  "
  [ device virtual_devices ]
  (py/call-attr experimental "set_virtual_device_configuration"  device virtual_devices ))

(defn set-visible-devices 
  "Set the list of visible devices.

  Sets the list of PhysicalDevices to be marked as visible to the runtime. Any
  devices that are not marked as visible means TensorFlow will not allocate
  memory on it and will not be able to place any operations on it as no
  LogicalDevice will be created on it. By default all discovered devices are
  marked as visible.

  The following example demonstrates disabling the first GPU on the machine.

  ```python
  physical_devices = config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, \"Not enough GPU hardware devices available\"
  # Disable first GPU
  tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')
  logical_devices = config.experimental.list_logical_devices('GPU')
  # Logical device was not created for first GPU
  assert len(logical_devices) == len(physical_devices) - 1
  ```

  Args:
    devices: (optional) List of PhysicalDevice objects to make visible
    device_type: (optional) Device types to limit visibility configuration to.
      Other device types will be left unaltered.
  "
  [ devices device_type ]
  (py/call-attr experimental "set_visible_devices"  devices device_type ))
