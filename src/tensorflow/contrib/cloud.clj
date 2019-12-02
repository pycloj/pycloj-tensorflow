(ns tensorflow.contrib.cloud
  "Module for cloud ops."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cloud (import-module "tensorflow.contrib.cloud"))

(defn configure-colab-session 
  "ConfigureColabSession configures the GCS file system in Colab.

  Args:
    session: A `tf.compat.v1.Session` session.
  "
  [ session ]
  (py/call-attr cloud "configure_colab_session"  session ))

(defn configure-gcs 
  "Configures the GCS file system for a given a session.

  Warning: GCS `credentials` may be transmitted over the network unencrypted.
  Please ensure that the network is trusted before using this function. For
  users running code entirely within Google Cloud, your data is protected by
  encryption in between data centers. For more information, please take a look
  at https://cloud.google.com/security/encryption-in-transit/.

  Args:
    session: A `tf.compat.v1.Session` session that should be used to configure
      the GCS file system.
    credentials: [Optional.] A JSON string
    block_cache: [Optional.] A BlockCacheParams to configure the block cache .
    device: [Optional.] The device to place the configure ops.
  "
  [ session credentials block_cache device ]
  (py/call-attr cloud "configure_gcs"  session credentials block_cache device ))
