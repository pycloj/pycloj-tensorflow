(ns tensorflow.-api.v1.autograph.experimental
  "Public API for tf.autograph.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.autograph.experimental"))

(defn do-not-convert 
  "Decorator that suppresses the conversion of a function.

  Args:
    func: function to decorate.

  Returns:
    If `func` is not None, returns a `Callable` which is equivalent to
    `func`, but is not converted by AutoGraph.
    If `func` is None, returns a decorator that, when invoked with a
    single `func` argument, returns a `Callable` equivalent to the
    above case.
  "
  [ func ]
  (py/call-attr experimental "do_not_convert"  func ))

(defn set-loop-options 
  "Specifies additional arguments to be passed to the enclosing while_loop.

  The parameters apply to and only to the immediately enclosing loop. It only
  has effect if the loop is staged as a TF while_loop; otherwise the parameters
  have no effect.

  Usage example:

    @tf.function(autograph=True)
    def dynamic_rnn(..., parallel_iterations=32):
      num_steps = ...
      for t in tf.range(num_steps):
        tf.autograph.experimental.set_loop_options(
            parallel_iterations=parallel_iterations)
        ...

  Args:
    parallel_iterations: See tf.while_loop.
    back_prop: See tf.while_loop.
    swap_memory: See tf.while_loop.
    maximum_iterations: See tf.while_loop.
  "
  [ & {:keys [parallel_iterations back_prop swap_memory maximum_iterations]} ]
   (py/call-attr-kw experimental "set_loop_options" [] {:parallel_iterations parallel_iterations :back_prop back_prop :swap_memory swap_memory :maximum_iterations maximum_iterations }))
