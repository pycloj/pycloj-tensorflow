(ns tensorflow.contrib.cloud.ConfigureGcsHook
  "ConfigureGcsHook configures GCS when used with Estimator/TPUEstimator.

  Warning: GCS `credentials` may be transmitted over the network unencrypted.
  Please ensure that the network is trusted before using this function. For
  users running code entirely within Google Cloud, your data is protected by
  encryption in between data centers. For more information, please take a look
  at https://cloud.google.com/security/encryption-in-transit/.

  Example:

  ```
  sess = tf.compat.v1.Session()
  refresh_token = raw_input(\"Refresh token: \")
  client_secret = raw_input(\"Client secret: \")
  client_id = \"<REDACTED>\"
  creds = {
      \"client_id\": client_id,
      \"refresh_token\": refresh_token,
      \"client_secret\": client_secret,
      \"type\": \"authorized_user\",
  }
  tf.contrib.cloud.configure_gcs(sess, credentials=creds)
  ```

  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cloud (import-module "tensorflow.contrib.cloud"))

(defn ConfigureGcsHook 
  "ConfigureGcsHook configures GCS when used with Estimator/TPUEstimator.

  Warning: GCS `credentials` may be transmitted over the network unencrypted.
  Please ensure that the network is trusted before using this function. For
  users running code entirely within Google Cloud, your data is protected by
  encryption in between data centers. For more information, please take a look
  at https://cloud.google.com/security/encryption-in-transit/.

  Example:

  ```
  sess = tf.compat.v1.Session()
  refresh_token = raw_input(\"Refresh token: \")
  client_secret = raw_input(\"Client secret: \")
  client_id = \"<REDACTED>\"
  creds = {
      \"client_id\": client_id,
      \"refresh_token\": refresh_token,
      \"client_secret\": client_secret,
      \"type\": \"authorized_user\",
  }
  tf.contrib.cloud.configure_gcs(sess, credentials=creds)
  ```

  "
  [ credentials block_cache ]
  (py/call-attr cloud "ConfigureGcsHook"  credentials block_cache ))

(defn after-create-session 
  ""
  [ self session coord ]
  (py/call-attr self "after_create_session"  self session coord ))

(defn after-run 
  "Called after each call to run().

    The `run_values` argument contains results of requested ops/tensors by
    `before_run()`.

    The `run_context` argument is the same one send to `before_run` call.
    `run_context.request_stop()` can be called to stop the iteration.

    If `session.run()` raises any exceptions then `after_run()` is not called.

    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    "
  [ self run_context run_values ]
  (py/call-attr self "after_run"  self run_context run_values ))

(defn before-run 
  "Called before each call to run().

    You can return from this call a `SessionRunArgs` object indicating ops or
    tensors to add to the upcoming `run()` call.  These ops/tensors will be run
    together with the ops/tensors originally passed to the original run() call.
    The run args you return can also contain feeds to be added to the run()
    call.

    The `run_context` argument is a `SessionRunContext` that provides
    information about the upcoming `run()` call: the originally requested
    op/tensors, the TensorFlow Session.

    At this point graph is finalized and you can not add ops.

    Args:
      run_context: A `SessionRunContext` object.

    Returns:
      None or a `SessionRunArgs` object.
    "
  [ self run_context ]
  (py/call-attr self "before_run"  self run_context ))

(defn begin 
  ""
  [ self  ]
  (py/call-attr self "begin"  self  ))

(defn end 
  "Called at the end of session.

    The `session` argument can be used in case the hook wants to run final ops,
    such as saving a last checkpoint.

    If `session.run()` raises exception other than OutOfRangeError or
    StopIteration then `end()` is not called.
    Note the difference between `end()` and `after_run()` behavior when
    `session.run()` raises OutOfRangeError or StopIteration. In that case
    `end()` is called but `after_run()` is not called.

    Args:
      session: A TensorFlow Session that will be soon closed.
    "
  [ self session ]
  (py/call-attr self "end"  self session ))
