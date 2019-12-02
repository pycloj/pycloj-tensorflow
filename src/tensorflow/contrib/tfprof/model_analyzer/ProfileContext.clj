(ns tensorflow.contrib.tfprof.model-analyzer.ProfileContext
  "A Context that captures RunMetadata and performs profiling.

  ```python
    # Trace steps 100~200, profile at [150, 200] and dump profile at 200.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=range(100, 200, 3),
                                          dump_steps=[200]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling('op', opts, [150, 200])
      train_loop().

    # Tracing only.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir') as pctx:
      # Run train/eval loop for at least few hundred steps. Profiles will be
      # dumped to train_dir. Use web UI or command line to do profiling.
      train_loop().

    # When session object is available, do explicit trace, profile and dump.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=[],
                                          dump_steps=[]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.trace_next_step()
      _ = session.run(train_op)
      pctx.profiler.profile_operations(options=opts)
  ```

  Args:
    profile_dir: Directory to store profiles.
    trace_steps: A list of session run steps to trace. If None, use
        pre-defined steps.
    dump_steps: A list of steps to dump the profile to `profile_dir`. If None,
        use pre-defined steps.
    enabled: If false, everything is disabled with minimal overhead. It allows
        user to only enable profiling when needed.
    debug: If true, also dumps the raw trace RunMetadata text file to
        profile_dir. And print debugging message. Useful for bug report.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce model-analyzer (import-module "tensorflow.contrib.tfprof.model_analyzer"))
(defn ProfileContext 
  "A Context that captures RunMetadata and performs profiling.

  ```python
    # Trace steps 100~200, profile at [150, 200] and dump profile at 200.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=range(100, 200, 3),
                                          dump_steps=[200]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling('op', opts, [150, 200])
      train_loop().

    # Tracing only.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir') as pctx:
      # Run train/eval loop for at least few hundred steps. Profiles will be
      # dumped to train_dir. Use web UI or command line to do profiling.
      train_loop().

    # When session object is available, do explicit trace, profile and dump.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=[],
                                          dump_steps=[]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.trace_next_step()
      _ = session.run(train_op)
      pctx.profiler.profile_operations(options=opts)
  ```

  Args:
    profile_dir: Directory to store profiles.
    trace_steps: A list of session run steps to trace. If None, use
        pre-defined steps.
    dump_steps: A list of steps to dump the profile to `profile_dir`. If None,
        use pre-defined steps.
    enabled: If false, everything is disabled with minimal overhead. It allows
        user to only enable profiling when needed.
    debug: If true, also dumps the raw trace RunMetadata text file to
        profile_dir. And print debugging message. Useful for bug report.
  "
  [profile_dir trace_steps dump_steps  & {:keys [enabled debug]} ]
    (py/call-attr-kw model-analyzer "ProfileContext" [profile_dir trace_steps dump_steps] {:enabled enabled :debug debug }))

(defn add-auto-profiling 
  "Traces and profiles at some session run steps.

    Args:
      cmd: The profiling commands. (i.e. scope, op, python, graph)
      options: The profiling options.
      profile_steps: A list/set of integers. The profiling command and options
          will be run automatically at these integer steps. Each step is
          a session.run.
    "
  [ self cmd options profile_steps ]
  (py/call-attr self "add_auto_profiling"  self cmd options profile_steps ))

(defn dump-next-step 
  "Enable tracing and dump profiles at next step."
  [ self  ]
  (py/call-attr self "dump_next_step"  self  ))

(defn get-profiles 
  "Returns profiling results for each step at which `cmd` was run.

    Args:
      cmd: string, profiling command used in an `add_auto_profiling` call.

    Returns:
      dict[int: (MultiGraphNodeProto | GraphNodeProto)]. Keys are steps at which
      the profiling command was run. Values are the outputs of profiling.
      For \"code\" and \"op\" commands this will be a `MultiGraphNodeProto`, for
      \"scope\" and \"graph\" commands this will be a `GraphNodeProto.

    Raises:
      ValueError: if `cmd` was never run (either because no session.run call was
      made or because there was no `add_auto_profiling` call with the specified
      `cmd`.
    "
  [ self cmd ]
  (py/call-attr self "get_profiles"  self cmd ))

(defn profiler 
  "Returns the current profiler object."
  [ self ]
    (py/call-attr self "profiler"))

(defn trace-next-step 
  "Enables tracing and adds traces to profiler at next step."
  [ self  ]
  (py/call-attr self "trace_next_step"  self  ))
