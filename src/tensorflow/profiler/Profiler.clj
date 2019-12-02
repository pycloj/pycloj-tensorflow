(ns tensorflow.profiler.Profiler
  "TensorFlow multi-step profiler.

  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/README.md

  ```python
  Typical use case:
    # Currently we are only allowed to create 1 profiler per process.
    profiler = Profiler(sess.graph)

    for i in xrange(total_steps):
      if i % 10000 == 0:
        run_meta = tf.compat.v1.RunMetadata()
        _ = sess.run(...,
                     options=tf.compat.v1.RunOptions(
                         trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_meta)
        profiler.add_step(i, run_meta)

        # Profile the parameters of your model.
        profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
            .trainable_variables_parameter()))

        # Or profile the timing of your model operations.
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = (option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(i)
                .with_timeline_output(filename).build())
        profiler.profile_graph(options=opts)
      else:
        _ = sess.run(...)
    # Auto detect problems and generate advice.
    profiler.advise()
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
(defonce profiler (import-module "tensorflow.profiler"))

(defn Profiler 
  "TensorFlow multi-step profiler.

  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/README.md

  ```python
  Typical use case:
    # Currently we are only allowed to create 1 profiler per process.
    profiler = Profiler(sess.graph)

    for i in xrange(total_steps):
      if i % 10000 == 0:
        run_meta = tf.compat.v1.RunMetadata()
        _ = sess.run(...,
                     options=tf.compat.v1.RunOptions(
                         trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_meta)
        profiler.add_step(i, run_meta)

        # Profile the parameters of your model.
        profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
            .trainable_variables_parameter()))

        # Or profile the timing of your model operations.
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = (option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(i)
                .with_timeline_output(filename).build())
        profiler.profile_graph(options=opts)
      else:
        _ = sess.run(...)
    # Auto detect problems and generate advice.
    profiler.advise()
  ```
  "
  [ graph op_log ]
  (py/call-attr profiler "Profiler"  graph op_log ))

(defn add-step 
  "Add statistics of a step.

    Args:
      step: int, An id used to group one or more different `run_meta` together.
          When profiling with the profile_xxx APIs, user can use the `step`
          id in the `options` to profile these `run_meta` together.
      run_meta: RunMetadata proto that contains statistics of a session run.
    "
  [ self step run_meta ]
  (py/call-attr self "add_step"  self step run_meta ))

(defn advise 
  "Automatically detect problems and generate reports.

    Args:
      options: A dict of options. See ALL_ADVICE example above.
    Returns:
      A Advise proto that conains the reports from all checkers.
    "
  [ self options ]
  (py/call-attr self "advise"  self options ))

(defn profile-graph 
  "Profile the statistics of graph nodes, organized by dataflow graph.

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a GraphNodeProto that records the results.
    "
  [ self options ]
  (py/call-attr self "profile_graph"  self options ))

(defn profile-name-scope 
  "Profile the statistics of graph nodes, organized by name scope.

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a GraphNodeProto that records the results.
    "
  [ self options ]
  (py/call-attr self "profile_name_scope"  self options ))

(defn profile-operations 
  "Profile the statistics of the Operation types (e.g. MatMul, Conv2D).

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a MultiGraphNodeProto that records the results.
    "
  [ self options ]
  (py/call-attr self "profile_operations"  self options ))

(defn profile-python 
  "Profile the statistics of the Python codes.

      By default, it shows the call stack from root. To avoid
      redundant output, you may use options to filter as below
        options['show_name_regexes'] = ['.*my_code.py.*']

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a MultiGraphNodeProto that records the results.
    "
  [ self options ]
  (py/call-attr self "profile_python"  self options ))

(defn serialize-to-string 
  "Serialize the ProfileProto to a binary string.

      Users can write it to file for offline analysis by tfprof commandline
      or graphical interface.

    Returns:
      ProfileProto binary string.
    "
  [ self  ]
  (py/call-attr self "serialize_to_string"  self  ))
