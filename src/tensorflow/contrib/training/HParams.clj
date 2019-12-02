(ns tensorflow.contrib.training.HParams
  "Class to hold a set of hyperparameters as name-value pairs.

  A `HParams` object holds hyperparameters used to build and train a model,
  such as the number of hidden units in a neural net layer or the learning rate
  to use when training.

  You first create a `HParams` object by specifying the names and values of the
  hyperparameters.

  To make them easily accessible the parameter names are added as direct
  attributes of the class.  A typical usage is as follows:

  ```python
  # Create a HParams object specifying names and values of the model
  # hyperparameters:
  hparams = HParams(learning_rate=0.1, num_hidden_units=100)

  # The hyperparameter are available as attributes of the HParams object:
  hparams.learning_rate ==> 0.1
  hparams.num_hidden_units ==> 100
  ```

  Hyperparameters have type, which is inferred from the type of their value
  passed at construction type.   The currently supported types are: integer,
  float, boolean, string, and list of integer, float, boolean, or string.

  You can override hyperparameter values by calling the
  [`parse()`](#HParams.parse) method, passing a string of comma separated
  `name=value` pairs.  This is intended to make it possible to override
  any hyperparameter values from a single command-line flag to which
  the user passes 'hyper-param=value' pairs.  It avoids having to define
  one flag for each hyperparameter.

  The syntax expected for each value depends on the type of the parameter.
  See `parse()` for a description of the syntax.

  Example:

  ```python
  # Define a command line flag to pass name=value pairs.
  # For example using argparse:
  import argparse
  parser = argparse.ArgumentParser(description='Train my model.')
  parser.add_argument('--hparams', type=str,
                      help='Comma separated list of \"name=value\" pairs.')
  args = parser.parse_args()
  ...
  def my_program():
    # Create a HParams object specifying the names and values of the
    # model hyperparameters:
    hparams = tf.contrib.training.HParams(
        learning_rate=0.1,
        num_hidden_units=100,
        activations=['relu', 'tanh'])

    # Override hyperparameters values by parsing the command line
    hparams.parse(args.hparams)

    # If the user passed `--hparams=learning_rate=0.3` on the command line
    # then 'hparams' has the following attributes:
    hparams.learning_rate ==> 0.3
    hparams.num_hidden_units ==> 100
    hparams.activations ==> ['relu', 'tanh']

    # If the hyperparameters are in json format use parse_json:
    hparams.parse_json('{\"learning_rate\": 0.3, \"activations\": \"relu\"}')
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
(defonce training (import-module "tensorflow.contrib.training"))

(defn HParams 
  "Class to hold a set of hyperparameters as name-value pairs.

  A `HParams` object holds hyperparameters used to build and train a model,
  such as the number of hidden units in a neural net layer or the learning rate
  to use when training.

  You first create a `HParams` object by specifying the names and values of the
  hyperparameters.

  To make them easily accessible the parameter names are added as direct
  attributes of the class.  A typical usage is as follows:

  ```python
  # Create a HParams object specifying names and values of the model
  # hyperparameters:
  hparams = HParams(learning_rate=0.1, num_hidden_units=100)

  # The hyperparameter are available as attributes of the HParams object:
  hparams.learning_rate ==> 0.1
  hparams.num_hidden_units ==> 100
  ```

  Hyperparameters have type, which is inferred from the type of their value
  passed at construction type.   The currently supported types are: integer,
  float, boolean, string, and list of integer, float, boolean, or string.

  You can override hyperparameter values by calling the
  [`parse()`](#HParams.parse) method, passing a string of comma separated
  `name=value` pairs.  This is intended to make it possible to override
  any hyperparameter values from a single command-line flag to which
  the user passes 'hyper-param=value' pairs.  It avoids having to define
  one flag for each hyperparameter.

  The syntax expected for each value depends on the type of the parameter.
  See `parse()` for a description of the syntax.

  Example:

  ```python
  # Define a command line flag to pass name=value pairs.
  # For example using argparse:
  import argparse
  parser = argparse.ArgumentParser(description='Train my model.')
  parser.add_argument('--hparams', type=str,
                      help='Comma separated list of \"name=value\" pairs.')
  args = parser.parse_args()
  ...
  def my_program():
    # Create a HParams object specifying the names and values of the
    # model hyperparameters:
    hparams = tf.contrib.training.HParams(
        learning_rate=0.1,
        num_hidden_units=100,
        activations=['relu', 'tanh'])

    # Override hyperparameters values by parsing the command line
    hparams.parse(args.hparams)

    # If the user passed `--hparams=learning_rate=0.3` on the command line
    # then 'hparams' has the following attributes:
    hparams.learning_rate ==> 0.3
    hparams.num_hidden_units ==> 100
    hparams.activations ==> ['relu', 'tanh']

    # If the hyperparameters are in json format use parse_json:
    hparams.parse_json('{\"learning_rate\": 0.3, \"activations\": \"relu\"}')
  ```
  "
  [ hparam_def model_structure ]
  (py/call-attr training "HParams"  hparam_def model_structure ))

(defn add-hparam 
  "Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    "
  [ self name value ]
  (py/call-attr self "add_hparam"  self name value ))

(defn del-hparam 
  "Removes the hyperparameter with key 'name'.

    Does nothing if it isn't present.

    Args:
      name: Name of the hyperparameter.
    "
  [ self name ]
  (py/call-attr self "del_hparam"  self name ))

(defn from-proto 
  ""
  [ self hparam_def import_scope ]
  (py/call-attr self "from_proto"  self hparam_def import_scope ))

(defn get 
  "Returns the value of `key` if it exists, else `default`."
  [ self key default ]
  (py/call-attr self "get"  self key default ))

(defn get-model-structure 
  ""
  [ self  ]
  (py/call-attr self "get_model_structure"  self  ))

(defn override-from-dict 
  "Override existing hyperparameter values, parsing new values from a dictionary.

    Args:
      values_dict: Dictionary of name:value pairs.

    Returns:
      The `HParams` instance.

    Raises:
      KeyError: If a hyperparameter in `values_dict` doesn't exist.
      ValueError: If `values_dict` cannot be parsed.
    "
  [ self values_dict ]
  (py/call-attr self "override_from_dict"  self values_dict ))

(defn parse 
  "Override existing hyperparameter values, parsing new values from a string.

    See parse_values for more detail on the allowed format for values.

    Args:
      values: String.  Comma separated list of `name=value` pairs where 'value'
        must follow the syntax described above.

    Returns:
      The `HParams` instance.

    Raises:
      ValueError: If `values` cannot be parsed or a hyperparameter in `values`
      doesn't exist.
    "
  [ self values ]
  (py/call-attr self "parse"  self values ))

(defn parse-json 
  "Override existing hyperparameter values, parsing new values from a json object.

    Args:
      values_json: String containing a json object of name:value pairs.

    Returns:
      The `HParams` instance.

    Raises:
      KeyError: If a hyperparameter in `values_json` doesn't exist.
      ValueError: If `values_json` cannot be parsed.
    "
  [ self values_json ]
  (py/call-attr self "parse_json"  self values_json ))

(defn set-from-map 
  "DEPRECATED. Use override_from_dict. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `override_from_dict`."
  [ self values_map ]
  (py/call-attr self "set_from_map"  self values_map ))

(defn set-hparam 
  "Set the value of an existing hyperparameter.

    This function verifies that the type of the value matches the type of the
    existing hyperparameter.

    Args:
      name: Name of the hyperparameter.
      value: New value of the hyperparameter.

    Raises:
      KeyError: If the hyperparameter doesn't exist.
      ValueError: If there is a type mismatch.
    "
  [ self name value ]
  (py/call-attr self "set_hparam"  self name value ))

(defn set-model-structure 
  ""
  [ self model_structure ]
  (py/call-attr self "set_model_structure"  self model_structure ))
(defn to-json 
  "Serializes the hyperparameters into JSON.

    Args:
      indent: If a non-negative integer, JSON array elements and object members
        will be pretty-printed with that indent level. An indent level of 0, or
        negative, will only insert newlines. `None` (the default) selects the
        most compact representation.
      separators: Optional `(item_separator, key_separator)` tuple. Default is
        `(', ', ': ')`.
      sort_keys: If `True`, the output dictionaries will be sorted by key.

    Returns:
      A JSON string.
    "
  [self indent separators  & {:keys [sort_keys]} ]
    (py/call-attr-kw self "to_json" [indent separators] {:sort_keys sort_keys }))

(defn to-proto 
  "Converts a `HParams` object to a `HParamDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `HParamDef` protocol buffer.
    "
  [ self export_scope ]
  (py/call-attr self "to_proto"  self export_scope ))

(defn values 
  "Return the hyperparameter values as a Python dictionary.

    Returns:
      A dictionary with hyperparameter names as keys.  The values are the
      hyperparameter values.
    "
  [ self  ]
  (py/call-attr self "values"  self  ))
