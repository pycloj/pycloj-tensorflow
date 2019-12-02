(ns tensorflow.contrib.opt.ScipyOptimizerInterface
  "Wrapper allowing `scipy.optimize.minimize` to operate a `tf.compat.v1.Session`.

  Example:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))

  optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})

  with tf.compat.v1.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [0., 0.].
  ```

  Example with simple bound constraints:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))

  optimizer = ScipyOptimizerInterface(
      loss, var_to_bounds={vector: ([1, 2], np.infty)})

  with tf.compat.v1.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [1., 2.].
  ```

  Example with more complicated constraints:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  # Ensure the vector's y component is = 1.
  equalities = [vector[1] - 1.]
  # Ensure the vector's x component is >= 1.
  inequalities = [vector[0] - 1.]

  # Our default SciPy optimization algorithm, L-BFGS-B, does not support
  # general constraints. Thus we use SLSQP instead.
  optimizer = ScipyOptimizerInterface(
      loss, equalities=equalities, inequalities=inequalities, method='SLSQP')

  with tf.compat.v1.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [1., 1.].
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
(defonce opt (import-module "tensorflow.contrib.opt"))

(defn ScipyOptimizerInterface 
  "Wrapper allowing `scipy.optimize.minimize` to operate a `tf.compat.v1.Session`.

  Example:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))

  optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})

  with tf.compat.v1.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [0., 0.].
  ```

  Example with simple bound constraints:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))

  optimizer = ScipyOptimizerInterface(
      loss, var_to_bounds={vector: ([1, 2], np.infty)})

  with tf.compat.v1.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [1., 2.].
  ```

  Example with more complicated constraints:

  ```python
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  # Ensure the vector's y component is = 1.
  equalities = [vector[1] - 1.]
  # Ensure the vector's x component is >= 1.
  inequalities = [vector[0] - 1.]

  # Our default SciPy optimization algorithm, L-BFGS-B, does not support
  # general constraints. Thus we use SLSQP instead.
  optimizer = ScipyOptimizerInterface(
      loss, equalities=equalities, inequalities=inequalities, method='SLSQP')

  with tf.compat.v1.Session() as session:
    optimizer.minimize(session)

  # The value of vector should now be [1., 1.].
  ```
  "
  [ loss var_list equalities inequalities var_to_bounds ]
  (py/call-attr opt "ScipyOptimizerInterface"  loss var_list equalities inequalities var_to_bounds ))

(defn minimize 
  "Minimize a scalar `Tensor`.

    Variables subject to optimization are updated in-place at the end of
    optimization.

    Note that this method does *not* just return a minimization `Op`, unlike
    `Optimizer.minimize()`; instead it actually performs minimization by
    executing commands to control a `Session`.

    Args:
      session: A `Session` instance.
      feed_dict: A feed dict to be passed to calls to `session.run`.
      fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
        as positional arguments.
      step_callback: A function to be called at each optimization step;
        arguments are the current values of all optimization variables
        flattened into a single vector.
      loss_callback: A function to be called every time the loss and gradients
        are computed, with evaluated fetches supplied as positional arguments.
      **run_kwargs: kwargs to pass to `session.run`.
    "
  [ self session feed_dict fetches step_callback loss_callback ]
  (py/call-attr self "minimize"  self session feed_dict fetches step_callback loss_callback ))
