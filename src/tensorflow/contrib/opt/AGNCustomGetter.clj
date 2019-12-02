(ns tensorflow.contrib.opt.AGNCustomGetter
  "Custom_getter class is used to do:

  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables(global center variables)
  3. Generate grad variables(gradients) which record the gradients sum
    and place them at worker device
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device.
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

(defn AGNCustomGetter 
  "Custom_getter class is used to do:

  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables(global center variables)
  3. Generate grad variables(gradients) which record the gradients sum
    and place them at worker device
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device.
  "
  [ worker_device ]
  (py/call-attr opt "AGNCustomGetter"  worker_device ))
