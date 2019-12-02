(ns tensorflow.-api.v1.compat.v2.lite.Interpreter
  "Interpreter interface for TensorFlow Lite Models.

  This makes the TensorFlow Lite interpreter accessible in Python.
  It is possible to use this interpreter in a multithreaded Python environment,
  but you must be sure to call functions of a particular instance from only
  one thread at a time. So if you want to have 4 threads running different
  inferences simultaneously, create  an interpreter for each one as thread-local
  data. Similarly, if you are calling invoke() in one thread on a single
  interpreter but you want to use tensor() on another thread once it is done,
  you must use a synchronization primitive between the threads to ensure invoke
  has returned before calling tensor().
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lite (import-module "tensorflow._api.v1.compat.v2.lite"))

(defn Interpreter 
  "Interpreter interface for TensorFlow Lite Models.

  This makes the TensorFlow Lite interpreter accessible in Python.
  It is possible to use this interpreter in a multithreaded Python environment,
  but you must be sure to call functions of a particular instance from only
  one thread at a time. So if you want to have 4 threads running different
  inferences simultaneously, create  an interpreter for each one as thread-local
  data. Similarly, if you are calling invoke() in one thread on a single
  interpreter but you want to use tensor() on another thread once it is done,
  you must use a synchronization primitive between the threads to ensure invoke
  has returned before calling tensor().
  "
  [ model_path model_content experimental_delegates ]
  (py/call-attr lite "Interpreter"  model_path model_content experimental_delegates ))

(defn allocate-tensors 
  ""
  [ self  ]
  (py/call-attr self "allocate_tensors"  self  ))

(defn get-input-details 
  "Gets model input details.

    Returns:
      A list of input details.
    "
  [ self  ]
  (py/call-attr self "get_input_details"  self  ))

(defn get-output-details 
  "Gets model output details.

    Returns:
      A list of output details.
    "
  [ self  ]
  (py/call-attr self "get_output_details"  self  ))

(defn get-tensor 
  "Gets the value of the input tensor (get a copy).

    If you wish to avoid the copy, use `tensor()`. This function cannot be used
    to read intermediate results.

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
                    the 'index' field in get_output_details.

    Returns:
      a numpy array.
    "
  [ self tensor_index ]
  (py/call-attr self "get_tensor"  self tensor_index ))

(defn get-tensor-details 
  "Gets tensor details for every tensor with valid tensor details.

    Tensors where required information about the tensor is not found are not
    added to the list. This includes temporary tensors without a name.

    Returns:
      A list of dictionaries containing tensor information.
    "
  [ self  ]
  (py/call-attr self "get_tensor_details"  self  ))

(defn invoke 
  "Invoke the interpreter.

    Be sure to set the input sizes, allocate tensors and fill values before
    calling this. Also, note that this function releases the GIL so heavy
    computation can be done in the background while the Python interpreter
    continues. No other function on this object should be called while the
    invoke() call has not finished.

    Raises:
      ValueError: When the underlying interpreter fails raise ValueError.
    "
  [ self  ]
  (py/call-attr self "invoke"  self  ))

(defn reset-all-variables 
  ""
  [ self  ]
  (py/call-attr self "reset_all_variables"  self  ))

(defn resize-tensor-input 
  "Resizes an input tensor.

    Args:
      input_index: Tensor index of input to set. This value can be gotten from
                   the 'index' field in get_input_details.
      tensor_size: The tensor_shape to resize the input to.

    Raises:
      ValueError: If the interpreter could not resize the input tensor.
    "
  [ self input_index tensor_size ]
  (py/call-attr self "resize_tensor_input"  self input_index tensor_size ))

(defn set-tensor 
  "Sets the value of the input tensor. Note this copies data in `value`.

    If you want to avoid copying, you can use the `tensor()` function to get a
    numpy buffer pointing to the input buffer in the tflite interpreter.

    Args:
      tensor_index: Tensor index of tensor to set. This value can be gotten from
                    the 'index' field in get_input_details.
      value: Value of tensor to set.

    Raises:
      ValueError: If the interpreter could not set the tensor.
    "
  [ self tensor_index value ]
  (py/call-attr self "set_tensor"  self tensor_index value ))

(defn tensor 
  "Returns function that gives a numpy view of the current tensor buffer.

    This allows reading and writing to this tensors w/o copies. This more
    closely mirrors the C++ Interpreter class interface's tensor() member, hence
    the name. Be careful to not hold these output references through calls
    to `allocate_tensors()` and `invoke()`. This function cannot be used to read
    intermediate results.

    Usage:

    ```
    interpreter.allocate_tensors()
    input = interpreter.tensor(interpreter.get_input_details()[0][\"index\"])
    output = interpreter.tensor(interpreter.get_output_details()[0][\"index\"])
    for i in range(10):
      input().fill(3.)
      interpreter.invoke()
      print(\"inference %s\" % output())
    ```

    Notice how this function avoids making a numpy array directly. This is
    because it is important to not hold actual numpy views to the data longer
    than necessary. If you do, then the interpreter can no longer be invoked,
    because it is possible the interpreter would resize and invalidate the
    referenced tensors. The NumPy API doesn't allow any mutability of the
    the underlying buffers.

    WRONG:

    ```
    input = interpreter.tensor(interpreter.get_input_details()[0][\"index\"])()
    output = interpreter.tensor(interpreter.get_output_details()[0][\"index\"])()
    interpreter.allocate_tensors()  # This will throw RuntimeError
    for i in range(10):
      input.fill(3.)
      interpreter.invoke()  # this will throw RuntimeError since input,output
    ```

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
                    the 'index' field in get_output_details.

    Returns:
      A function that can return a new numpy array pointing to the internal
      TFLite tensor state at any point. It is safe to hold the function forever,
      but it is not safe to hold the numpy array forever.
    "
  [ self tensor_index ]
  (py/call-attr self "tensor"  self tensor_index ))
