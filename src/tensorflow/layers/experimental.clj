(ns tensorflow.-api.v1.layers.experimental
  "Public API for tf.layers.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.layers.experimental"))

(defn keras-style-scope 
  "Use Keras-style variable management.

  All tf.layers and tf RNN cells created in this scope use Keras-style
  variable management.  Creating such layers with a scope= argument is
  disallowed, and reuse=True is disallowed.

  The purpose of this scope is to allow users of existing layers to
  slowly transition to a Keras layers API without breaking existing
  functionality.

  One example of this is when using TensorFlow's RNN classes with Keras
  Models or Networks.  Because Keras models do not properly set variable
  scopes, users of RNNs may either accidentally share scopes between two
  different models, or get errors about variables that already exist.

  Example:

  ```python
  class RNNModel(tf.keras.Model):

    def __init__(self, name):
      super(RNNModel, self).__init__(name=name)
      self.rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
        [tf.compat.v1.nn.rnn_cell.LSTMCell(64) for _ in range(2)])

    def call(self, input, state):
      return self.rnn(input, state)

  model_1 = RNNModel(\"model_1\")
  model_2 = RNNModel(\"model_2\")

  # OK
  output_1, next_state_1 = model_1(input, state)
  # Raises an error about trying to create an already existing variable.
  output_2, next_state_2 = model_2(input, state)
  ```

  The solution is to wrap the model construction and execution in a keras-style
  scope:

  ```python
  with keras_style_scope():
    model_1 = RNNModel(\"model_1\")
    model_2 = RNNModel(\"model_2\")

    # model_1 and model_2 are guaranteed to create their own variables.
    output_1, next_state_1 = model_1(input, state)
    output_2, next_state_2 = model_2(input, state)

    assert len(model_1.weights) > 0
    assert len(model_2.weights) > 0
    assert(model_1.weights != model_2.weights)
  ```

  Yields:
    A keras layer style scope.
  "
  [  ]
  (py/call-attr experimental "keras_style_scope"  ))

(defn set-keras-style 
  "Use Keras-style variable management.

  All tf.layers and tf RNN cells created after keras style ha been enabled
  use Keras-style variable management.  Creating such layers with a
  scope= argument is disallowed, and reuse=True is disallowed.

  The purpose of this function is to allow users of existing layers to
  slowly transition to Keras layers API without breaking existing
  functionality.

  For more details, see the documentation for `keras_style_scope`.

  Note, once keras style has been set, it is set globally for the entire
  program and cannot be unset.

  Example:

  ```python
  set_keras_style()

  model_1 = RNNModel(name=\"model_1\")
  model_2 = RNNModel(name=\"model_2\")

  # model_1 and model_2 are guaranteed to create their own variables.
  output_1, next_state_1 = model_1(input, state)
  output_2, next_state_2 = model_2(input, state)

  assert len(model_1.weights) > 0
  assert len(model_2.weights) > 0
  assert(model_1.weights != model_2.weights)
  ```
  "
  [  ]
  (py/call-attr experimental "set_keras_style"  ))
