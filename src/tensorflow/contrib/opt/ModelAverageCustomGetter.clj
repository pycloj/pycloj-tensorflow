(ns tensorflow.contrib.opt.ModelAverageCustomGetter
  "Custom_getter class is used to do.

  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device. Besides, use 'tf.compat.v1.get_variable' instead of
    'tf.Variable' to
    use this custom getter.

  For example,
  ma_custom_getter = ModelAverageCustomGetter(worker_device)
  with tf.device(
    tf.compat.v1.train.replica_device_setter(
      worker_device=worker_device,
      ps_device=\"/job:ps/cpu:0\",
      cluster=cluster)),
    tf.compat.v1.variable_scope('',custom_getter=ma_custom_getter):
    hid_w = tf.compat.v1.get_variable(
      initializer=tf.random.truncated_normal(
          [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
          stddev=1.0 / IMAGE_PIXELS),
      name=\"hid_w\")
    hid_b =
    tf.compat.v1.get_variable(initializer=tf.zeros([FLAGS.hidden_units]),
                            name=\"hid_b\")
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce opt (import-module "tensorflow.contrib.opt"))

(defn ModelAverageCustomGetter 
  "Custom_getter class is used to do.

  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device. Besides, use 'tf.compat.v1.get_variable' instead of
    'tf.Variable' to
    use this custom getter.

  For example,
  ma_custom_getter = ModelAverageCustomGetter(worker_device)
  with tf.device(
    tf.compat.v1.train.replica_device_setter(
      worker_device=worker_device,
      ps_device=\"/job:ps/cpu:0\",
      cluster=cluster)),
    tf.compat.v1.variable_scope('',custom_getter=ma_custom_getter):
    hid_w = tf.compat.v1.get_variable(
      initializer=tf.random.truncated_normal(
          [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
          stddev=1.0 / IMAGE_PIXELS),
      name=\"hid_w\")
    hid_b =
    tf.compat.v1.get_variable(initializer=tf.zeros([FLAGS.hidden_units]),
                            name=\"hid_b\")
  "
  [ worker_device ]
  (py/call-attr opt "ModelAverageCustomGetter"  worker_device ))
