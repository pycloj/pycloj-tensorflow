(ns tensorflow.test.TestCase
  "Base class for tests that need to test TensorFlow."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce test (import-module "tensorflow.test"))

(defn TestCase 
  "Base class for tests that need to test TensorFlow."
  [ & {:keys [methodName]} ]
   (py/call-attr-kw test "TestCase" [] {:methodName methodName }))

(defn addCleanup 
  "Add a function, with arguments, to be called when the test is
        completed. Functions added are called on a LIFO basis and are
        called after tearDown on test failure or success.

        Cleanup items are called even if setUp fails (unlike tearDown)."
  [ self  ]
  (py/call-attr self "addCleanup"  self  ))

(defn addTypeEqualityFunc 
  "Add a type specific assertEqual style function to compare a type.

        This method is for use by TestCase subclasses that need to register
        their own type equality functions to provide nicer error messages.

        Args:
            typeobj: The data type to call this function on when both values
                    are of the same type in assertEqual().
            function: The callable taking two arguments and an optional
                    msg= argument that raises self.failureException with a
                    useful error message when the two arguments are not equal.
        "
  [ self typeobj function ]
  (py/call-attr self "addTypeEqualityFunc"  self typeobj function ))

(defn assertAllClose 
  "Asserts that two structures of numpy arrays or Tensors, have near values.

    `a` and `b` can be arbitrarily nested structures. A layer of a nested
    structure can be a `dict`, `namedtuple`, `tuple` or `list`.

    Args:
      a: The expected numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      b: The actual numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      rtol: relative tolerance.
      atol: absolute tolerance.
      msg: Optional message to report on failure.

    Raises:
      ValueError: if only one of `a[p]` and `b[p]` is a dict or
          `a[p]` and `b[p]` have different length, where `[p]` denotes a path
          to the nested structure, e.g. given `a = [(1, 1), {'d': (6, 7)}]` and
          `[p] = [1]['d']`, then `a[p] = (6, 7)`.
    "
  [self a b & {:keys [rtol atol msg]
                       :or {msg None}} ]
    (py/call-attr-kw self "assertAllClose" [a b] {:rtol rtol :atol atol :msg msg }))

(defn assertAllCloseAccordingToType 
  "Like assertAllClose, but also suitable for comparing fp16 arrays.

    In particular, the tolerance is reduced to 1e-3 if at least
    one of the arguments is of type float16.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      rtol: relative tolerance.
      atol: absolute tolerance.
      float_rtol: relative tolerance for float32.
      float_atol: absolute tolerance for float32.
      half_rtol: relative tolerance for float16.
      half_atol: absolute tolerance for float16.
      bfloat16_rtol: relative tolerance for bfloat16.
      bfloat16_atol: absolute tolerance for bfloat16.
      msg: Optional message to report on failure.
    "
  [self a b & {:keys [rtol atol float_rtol float_atol half_rtol half_atol bfloat16_rtol bfloat16_atol msg]
                       :or {msg None}} ]
    (py/call-attr-kw self "assertAllCloseAccordingToType" [a b] {:rtol rtol :atol atol :float_rtol float_rtol :float_atol float_atol :half_rtol half_rtol :half_atol half_atol :bfloat16_rtol bfloat16_rtol :bfloat16_atol bfloat16_atol :msg msg }))

(defn assertAllEqual 
  "Asserts that two numpy arrays or Tensors have the same values.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    "
  [ self a b msg ]
  (py/call-attr self "assertAllEqual"  self a b msg ))

(defn assertAllGreater 
  "Assert element values are all greater than a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    "
  [ self a comparison_target ]
  (py/call-attr self "assertAllGreater"  self a comparison_target ))

(defn assertAllGreaterEqual 
  "Assert element values are all greater than or equal to a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    "
  [ self a comparison_target ]
  (py/call-attr self "assertAllGreaterEqual"  self a comparison_target ))
(defn assertAllInRange 
  "Assert that elements in a Tensor are all in a given range.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      lower_bound: lower bound of the range
      upper_bound: upper bound of the range
      open_lower_bound: (`bool`) whether the lower bound is open (i.e., > rather
        than the default >=)
      open_upper_bound: (`bool`) whether the upper bound is open (i.e., < rather
        than the default <=)

    Raises:
      AssertionError:
        if the value tensor does not have an ordered numeric type (float* or
          int*), or
        if there are nan values, or
        if any of the elements do not fall in the specified range.
    "
  [self target lower_bound upper_bound  & {:keys [open_lower_bound open_upper_bound]} ]
    (py/call-attr-kw self "assertAllInRange" [target lower_bound upper_bound] {:open_lower_bound open_lower_bound :open_upper_bound open_upper_bound }))

(defn assertAllInSet 
  "Assert that elements of a Tensor are all in a given closed set.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      expected_set: (`list`, `tuple` or `set`) The closed set that the elements
        of the value of `target` are expected to fall into.

    Raises:
      AssertionError:
        if any of the elements do not fall into `expected_set`.
    "
  [ self target expected_set ]
  (py/call-attr self "assertAllInSet"  self target expected_set ))

(defn assertAllLess 
  "Assert element values are all less than a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    "
  [ self a comparison_target ]
  (py/call-attr self "assertAllLess"  self a comparison_target ))

(defn assertAllLessEqual 
  "Assert element values are all less than or equal to a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    "
  [ self a comparison_target ]
  (py/call-attr self "assertAllLessEqual"  self a comparison_target ))

(defn assertAlmostEqual 
  "Fail if the two objects are unequal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is more than the given
           delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           If the two objects compare equal then they will automatically
           compare almost equal.
        "
  [ self first second places msg delta ]
  (py/call-attr self "assertAlmostEqual"  self first second places msg delta ))

(defn assertAlmostEquals 
  ""
  [ self  ]
  (py/call-attr self "assertAlmostEquals"  self  ))

(defn assertArrayNear 
  "Asserts that two float arrays are near each other.

    Checks that for all elements of farray1 and farray2
    |f1 - f2| < err.  Asserts a test failure if not.

    Args:
      farray1: a list of float values.
      farray2: a list of float values.
      err: a float value.
      msg: Optional message to report on failure.
    "
  [ self farray1 farray2 err msg ]
  (py/call-attr self "assertArrayNear"  self farray1 farray2 err msg ))

(defn assertBetween 
  "Asserts that value is between minv and maxv (inclusive)."
  [ self value minv maxv msg ]
  (py/call-attr self "assertBetween"  self value minv maxv msg ))

(defn assertCommandFails 
  "Asserts a shell command fails and the error matches a regex in a list.

    Args:
      command: List or string representing the command to run.
      regexes: the list of regular expression strings.
      env: Dictionary of environment variable settings. If None, no environment
          variables will be set for the child process. This is to make tests
          more hermetic. NOTE: this behavior is different than the standard
          subprocess module.
      close_fds: Whether or not to close all open fd's in the child after
          forking.
      msg: Optional message to report on failure.
    "
  [self command regexes env & {:keys [close_fds msg]
                       :or {msg None}} ]
    (py/call-attr-kw self "assertCommandFails" [command regexes env] {:close_fds close_fds :msg msg }))

(defn assertCommandSucceeds 
  "Asserts that a shell command succeeds (i.e. exits with code 0).

    Args:
      command: List or string representing the command to run.
      regexes: List of regular expression byte strings that match success.
      env: Dictionary of environment variable settings. If None, no environment
          variables will be set for the child process. This is to make tests
          more hermetic. NOTE: this behavior is different than the standard
          subprocess module.
      close_fds: Whether or not to close all open fd's in the child after
          forking.
      msg: Optional message to report on failure.
    "
  [self command & {:keys [regexes env close_fds msg]
                       :or {env None msg None}} ]
    (py/call-attr-kw self "assertCommandSucceeds" [command] {:regexes regexes :env env :close_fds close_fds :msg msg }))

(defn assertContainsExactSubsequence 
  "Asserts that \"container\" contains \"subsequence\" as an exact subsequence.

    Asserts that \"container\" contains all the elements of \"subsequence\", in
    order, and without other elements interspersed. For example, [1, 2, 3] is an
    exact subsequence of [0, 0, 1, 2, 3, 0] but not of [0, 0, 1, 2, 0, 3, 0].

    Args:
      container: the list we're testing for subsequence inclusion.
      subsequence: the list we hope will be an exact subsequence of container.
      msg: Optional message to report on failure.
    "
  [ self container subsequence msg ]
  (py/call-attr self "assertContainsExactSubsequence"  self container subsequence msg ))

(defn assertContainsInOrder 
  "Asserts that the strings provided are found in the target in order.

    This may be useful for checking HTML output.

    Args:
      strings: A list of strings, such as [ 'fox', 'dog' ]
      target: A target string in which to look for the strings, such as
          'The quick brown fox jumped over the lazy dog'.
      msg: Optional message to report on failure.
    "
  [ self strings target msg ]
  (py/call-attr self "assertContainsInOrder"  self strings target msg ))

(defn assertContainsSubsequence 
  "Asserts that \"container\" contains \"subsequence\" as a subsequence.

    Asserts that \"container\" contains all the elements of \"subsequence\", in
    order, but possibly with other elements interspersed. For example, [1, 2, 3]
    is a subsequence of [0, 0, 1, 2, 0, 3, 0] but not of [0, 0, 1, 3, 0, 2, 0].

    Args:
      container: the list we're testing for subsequence inclusion.
      subsequence: the list we hope will be a subsequence of container.
      msg: Optional message to report on failure.
    "
  [ self container subsequence msg ]
  (py/call-attr self "assertContainsSubsequence"  self container subsequence msg ))

(defn assertContainsSubset 
  "Checks whether actual iterable is a superset of expected iterable."
  [ self expected_subset actual_set msg ]
  (py/call-attr self "assertContainsSubset"  self expected_subset actual_set msg ))

(defn assertCountEqual 
  "An unordered sequence comparison asserting that the same elements,
        regardless of order.  If the same element occurs more than once,
        it verifies that the elements occur the same number of times.

            self.assertEqual(Counter(list(first)),
                             Counter(list(second)))

         Example:
            - [0, 1, 1] and [1, 0, 1] compare equal.
            - [0, 0, 1] and [0, 1] compare unequal.

        "
  [ self first second msg ]
  (py/call-attr self "assertCountEqual"  self first second msg ))

(defn assertDTypeEqual 
  "Assert ndarray data type is equal to expected.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      expected_dtype: Expected data type.
    "
  [ self target expected_dtype ]
  (py/call-attr self "assertDTypeEqual"  self target expected_dtype ))

(defn assertDeviceEqual 
  "Asserts that the two given devices are the same.

    Args:
      device1: A string device name or TensorFlow `DeviceSpec` object.
      device2: A string device name or TensorFlow `DeviceSpec` object.
      msg: Optional message to report on failure.
    "
  [ self device1 device2 msg ]
  (py/call-attr self "assertDeviceEqual"  self device1 device2 msg ))

(defn assertDictContainsSubset 
  "Checks whether dictionary is a superset of subset."
  [ self subset dictionary msg ]
  (py/call-attr self "assertDictContainsSubset"  self subset dictionary msg ))

(defn assertDictEqual 
  "Raises AssertionError if a and b are not equal dictionaries.

    Args:
      a: A dict, the expected value.
      b: A dict, the actual value.
      msg: An optional str, the associated message.

    Raises:
      AssertionError: if the dictionaries are not equal.
    "
  [ self a b msg ]
  (py/call-attr self "assertDictEqual"  self a b msg ))

(defn assertEmpty 
  "Asserts that an object has zero length.

    Args:
      container: Anything that implements the collections.abc.Sized interface.
      msg: Optional message to report on failure.
    "
  [ self container msg ]
  (py/call-attr self "assertEmpty"  self container msg ))

(defn assertEndsWith 
  "Asserts that actual.endswith(expected_end) is True.

    Args:
      actual: str
      expected_end: str
      msg: Optional message to report on failure.
    "
  [ self actual expected_end msg ]
  (py/call-attr self "assertEndsWith"  self actual expected_end msg ))

(defn assertEqual 
  "Fail if the two objects are unequal as determined by the '=='
           operator.
        "
  [ self first second msg ]
  (py/call-attr self "assertEqual"  self first second msg ))

(defn assertEquals 
  ""
  [ self  ]
  (py/call-attr self "assertEquals"  self  ))

(defn assertFalse 
  "Check that the expression is false."
  [ self expr msg ]
  (py/call-attr self "assertFalse"  self expr msg ))

(defn assertGreater 
  "Just like self.assertTrue(a > b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertGreater"  self a b msg ))

(defn assertGreaterEqual 
  "Just like self.assertTrue(a >= b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertGreaterEqual"  self a b msg ))

(defn assertIn 
  "Just like self.assertTrue(a in b), but with a nicer default message."
  [ self member container msg ]
  (py/call-attr self "assertIn"  self member container msg ))

(defn assertIs 
  "Just like self.assertTrue(a is b), but with a nicer default message."
  [ self expr1 expr2 msg ]
  (py/call-attr self "assertIs"  self expr1 expr2 msg ))

(defn assertIsInstance 
  "Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."
  [ self obj cls msg ]
  (py/call-attr self "assertIsInstance"  self obj cls msg ))

(defn assertIsNone 
  "Same as self.assertTrue(obj is None), with a nicer default message."
  [ self obj msg ]
  (py/call-attr self "assertIsNone"  self obj msg ))

(defn assertIsNot 
  "Just like self.assertTrue(a is not b), but with a nicer default message."
  [ self expr1 expr2 msg ]
  (py/call-attr self "assertIsNot"  self expr1 expr2 msg ))

(defn assertIsNotNone 
  "Included for symmetry with assertIsNone."
  [ self obj msg ]
  (py/call-attr self "assertIsNotNone"  self obj msg ))

(defn assertItemsEqual 
  "An unordered sequence comparison asserting that the same elements,
        regardless of order.  If the same element occurs more than once,
        it verifies that the elements occur the same number of times.

            self.assertEqual(Counter(list(first)),
                             Counter(list(second)))

         Example:
            - [0, 1, 1] and [1, 0, 1] compare equal.
            - [0, 0, 1] and [0, 1] compare unequal.

        "
  [ self first second msg ]
  (py/call-attr self "assertItemsEqual"  self first second msg ))

(defn assertJsonEqual 
  "Asserts that the JSON objects defined in two strings are equal.

    A summary of the differences will be included in the failure message
    using assertSameStructure.

    Args:
      first: A string containing JSON to decode and compare to second.
      second: A string containing JSON to decode and compare to first.
      msg: Additional text to include in the failure message.
    "
  [ self first second msg ]
  (py/call-attr self "assertJsonEqual"  self first second msg ))

(defn assertLen 
  "Asserts that an object has the expected length.

    Args:
      container: Anything that implements the collections.abc.Sized interface.
      expected_len: The expected length of the container.
      msg: Optional message to report on failure.
    "
  [ self container expected_len msg ]
  (py/call-attr self "assertLen"  self container expected_len msg ))

(defn assertLess 
  "Just like self.assertTrue(a < b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertLess"  self a b msg ))

(defn assertLessEqual 
  "Just like self.assertTrue(a <= b), but with a nicer default message."
  [ self a b msg ]
  (py/call-attr self "assertLessEqual"  self a b msg ))

(defn assertListEqual 
  "A list-specific equality assertion.

        Args:
            list1: The first list to compare.
            list2: The second list to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        "
  [ self list1 list2 msg ]
  (py/call-attr self "assertListEqual"  self list1 list2 msg ))

(defn assertLogs 
  "Fail unless a log message of level *level* or higher is emitted
        on *logger_name* or its children.  If omitted, *level* defaults to
        INFO and *logger* defaults to the root logger.

        This method must be used as a context manager, and will yield
        a recording object with two attributes: `output` and `records`.
        At the end of the context manager, the `output` attribute will
        be a list of the matching formatted log messages and the
        `records` attribute will be a list of the corresponding LogRecord
        objects.

        Example::

            with self.assertLogs('foo', level='INFO') as cm:
                logging.getLogger('foo').info('first message')
                logging.getLogger('foo.bar').error('second message')
            self.assertEqual(cm.output, ['INFO:foo:first message',
                                         'ERROR:foo.bar:second message'])
        "
  [ self logger level ]
  (py/call-attr self "assertLogs"  self logger level ))

(defn assertMultiLineEqual 
  "Asserts that two multi-line strings are equal."
  [ self first second msg ]
  (py/call-attr self "assertMultiLineEqual"  self first second msg ))

(defn assertNDArrayNear 
  "Asserts that two numpy arrays have near values.

    Args:
      ndarray1: a numpy ndarray.
      ndarray2: a numpy ndarray.
      err: a float. The maximum absolute difference allowed.
      msg: Optional message to report on failure.
    "
  [ self ndarray1 ndarray2 err msg ]
  (py/call-attr self "assertNDArrayNear"  self ndarray1 ndarray2 err msg ))

(defn assertNear 
  "Asserts that two floats are near each other.

    Checks that |f1 - f2| < err and asserts a test failure
    if not.

    Args:
      f1: A float value.
      f2: A float value.
      err: A float value.
      msg: An optional string message to append to the failure message.
    "
  [ self f1 f2 err msg ]
  (py/call-attr self "assertNear"  self f1 f2 err msg ))

(defn assertNoCommonElements 
  "Checks whether actual iterable and expected iterable are disjoint."
  [ self expected_seq actual_seq msg ]
  (py/call-attr self "assertNoCommonElements"  self expected_seq actual_seq msg ))

(defn assertNotAllClose 
  "Assert that two numpy arrays, or Tensors, do not have near values.

    Args:
      a: the first value to compare.
      b: the second value to compare.
      **kwargs: additional keyword arguments to be passed to the underlying
        `assertAllClose` call.

    Raises:
      AssertionError: If `a` and `b` are unexpectedly close at all elements.
    "
  [ self a b ]
  (py/call-attr self "assertNotAllClose"  self a b ))

(defn assertNotAllEqual 
  "Asserts that two numpy arrays or Tensors do not have the same values.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    "
  [ self a b msg ]
  (py/call-attr self "assertNotAllEqual"  self a b msg ))

(defn assertNotAlmostEqual 
  "Fail if the two objects are equal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is less than the given delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           Objects that are equal automatically fail.
        "
  [ self first second places msg delta ]
  (py/call-attr self "assertNotAlmostEqual"  self first second places msg delta ))

(defn assertNotAlmostEquals 
  ""
  [ self  ]
  (py/call-attr self "assertNotAlmostEquals"  self  ))

(defn assertNotEmpty 
  "Asserts that an object has non-zero length.

    Args:
      container: Anything that implements the collections.abc.Sized interface.
      msg: Optional message to report on failure.
    "
  [ self container msg ]
  (py/call-attr self "assertNotEmpty"  self container msg ))

(defn assertNotEndsWith 
  "Asserts that actual.endswith(unexpected_end) is False.

    Args:
      actual: str
      unexpected_end: str
      msg: Optional message to report on failure.
    "
  [ self actual unexpected_end msg ]
  (py/call-attr self "assertNotEndsWith"  self actual unexpected_end msg ))

(defn assertNotEqual 
  "Fail if the two objects are equal as determined by the '!='
           operator.
        "
  [ self first second msg ]
  (py/call-attr self "assertNotEqual"  self first second msg ))

(defn assertNotEquals 
  ""
  [ self  ]
  (py/call-attr self "assertNotEquals"  self  ))

(defn assertNotIn 
  "Just like self.assertTrue(a not in b), but with a nicer default message."
  [ self member container msg ]
  (py/call-attr self "assertNotIn"  self member container msg ))

(defn assertNotIsInstance 
  "Included for symmetry with assertIsInstance."
  [ self obj cls msg ]
  (py/call-attr self "assertNotIsInstance"  self obj cls msg ))

(defn assertNotRegex 
  "Fail the test if the text matches the regular expression."
  [ self text unexpected_regex msg ]
  (py/call-attr self "assertNotRegex"  self text unexpected_regex msg ))

(defn assertNotRegexpMatches 
  ""
  [ self  ]
  (py/call-attr self "assertNotRegexpMatches"  self  ))

(defn assertNotStartsWith 
  "Asserts that actual.startswith(unexpected_start) is False.

    Args:
      actual: str
      unexpected_start: str
      msg: Optional message to report on failure.
    "
  [ self actual unexpected_start msg ]
  (py/call-attr self "assertNotStartsWith"  self actual unexpected_start msg ))

(defn assertProtoEquals 
  "Asserts that message is same as parsed expected_message_ascii.

    Creates another prototype of message, reads the ascii message into it and
    then compares them using self._AssertProtoEqual().

    Args:
      expected_message_maybe_ascii: proto message in original or ascii form.
      message: the message to validate.
      msg: Optional message to report on failure.
    "
  [ self expected_message_maybe_ascii message msg ]
  (py/call-attr self "assertProtoEquals"  self expected_message_maybe_ascii message msg ))

(defn assertProtoEqualsVersion 
  ""
  [self expected actual & {:keys [producer min_consumer msg]
                       :or {msg None}} ]
    (py/call-attr-kw self "assertProtoEqualsVersion" [expected actual] {:producer producer :min_consumer min_consumer :msg msg }))

(defn assertRaises 
  "Fail unless an exception of class expected_exception is raised
           by the callable when invoked with specified positional and
           keyword arguments. If a different type of exception is
           raised, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertRaises(SomeException):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertRaises
           is used as a context object.

           The context manager keeps a reference to the exception as
           the 'exception' attribute. This allows you to inspect the
           exception after the assertion::

               with self.assertRaises(SomeException) as cm:
                   do_something()
               the_exception = cm.exception
               self.assertEqual(the_exception.error_code, 3)
        "
  [ self expected_exception ]
  (py/call-attr self "assertRaises"  self expected_exception ))

(defn assertRaisesOpError 
  ""
  [ self expected_err_re_or_predicate ]
  (py/call-attr self "assertRaisesOpError"  self expected_err_re_or_predicate ))

(defn assertRaisesRegex 
  "Asserts that the message in a raised exception matches a regex.

        Args:
            expected_exception: Exception class expected to be raised.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertRaisesRegex is used as a context manager.
        "
  [ self expected_exception expected_regex ]
  (py/call-attr self "assertRaisesRegex"  self expected_exception expected_regex ))

(defn assertRaisesRegexp 
  "Asserts that the message in a raised exception matches a regex.

        Args:
            expected_exception: Exception class expected to be raised.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertRaisesRegex is used as a context manager.
        "
  [ self expected_exception expected_regex ]
  (py/call-attr self "assertRaisesRegexp"  self expected_exception expected_regex ))

(defn assertRaisesWithLiteralMatch 
  "Asserts that the message in a raised exception equals the given string.

    Unlike assertRaisesRegex, this method takes a literal string, not
    a regular expression.

    with self.assertRaisesWithLiteralMatch(ExType, 'message'):
      DoSomething()

    Args:
      expected_exception: Exception class expected to be raised.
      expected_exception_message: String message expected in the raised
          exception.  For a raise exception e, expected_exception_message must
          equal str(e).
      callable_obj: Function to be called, or None to return a context.
      *args: Extra args.
      **kwargs: Extra kwargs.

    Returns:
      A context manager if callable_obj is None. Otherwise, None.

    Raises:
      self.failureException if callable_obj does not raise a matching exception.
    "
  [ self expected_exception expected_exception_message callable_obj ]
  (py/call-attr self "assertRaisesWithLiteralMatch"  self expected_exception expected_exception_message callable_obj ))

(defn assertRaisesWithPredicateMatch 
  "Returns a context manager to enclose code expected to raise an exception.

    If the exception is an OpError, the op stack is also included in the message
    predicate search.

    Args:
      exception_type: The expected type of exception that should be raised.
      expected_err_re_or_predicate: If this is callable, it should be a function
        of one argument that inspects the passed-in exception and returns True
        (success) or False (please fail the test). Otherwise, the error message
        is expected to match this regular expression partially.

    Returns:
      A context manager to surround code that is expected to raise an
      exception.
    "
  [ self exception_type expected_err_re_or_predicate ]
  (py/call-attr self "assertRaisesWithPredicateMatch"  self exception_type expected_err_re_or_predicate ))

(defn assertRegex 
  "Fail the test unless the text matches the regular expression."
  [ self text expected_regex msg ]
  (py/call-attr self "assertRegex"  self text expected_regex msg ))

(defn assertRegexMatch 
  "Asserts that at least one regex in regexes matches str.

    If possible you should use `assertRegex`, which is a simpler
    version of this method. `assertRegex` takes a single regular
    expression (a string or re compiled object) instead of a list.

    Notes:
    1. This function uses substring matching, i.e. the matching
       succeeds if *any* substring of the error message matches *any*
       regex in the list.  This is more convenient for the user than
       full-string matching.

    2. If regexes is the empty list, the matching will always fail.

    3. Use regexes=[''] for a regex that will always pass.

    4. '.' matches any single character *except* the newline.  To
       match any character, use '(.|\n)'.

    5. '^' matches the beginning of each line, not just the beginning
       of the string.  Similarly, '$' matches the end of each line.

    6. An exception will be thrown if regexes contains an invalid
       regex.

    Args:
      actual_str:  The string we try to match with the items in regexes.
      regexes:  The regular expressions we want to match against str.
          See \"Notes\" above for detailed notes on how this is interpreted.
      message:  The message to be printed if the test fails.
    "
  [ self actual_str regexes message ]
  (py/call-attr self "assertRegexMatch"  self actual_str regexes message ))

(defn assertRegexpMatches 
  ""
  [ self  ]
  (py/call-attr self "assertRegexpMatches"  self  ))

(defn assertSameElements 
  "Asserts that two sequences have the same elements (in any order).

    This method, unlike assertCountEqual, doesn't care about any
    duplicates in the expected and actual sequences.

      >> assertSameElements([1, 1, 1, 0, 0, 0], [0, 1])
      # Doesn't raise an AssertionError

    If possible, you should use assertCountEqual instead of
    assertSameElements.

    Args:
      expected_seq: A sequence containing elements we are expecting.
      actual_seq: The sequence that we are testing.
      msg: The message to be printed if the test fails.
    "
  [ self expected_seq actual_seq msg ]
  (py/call-attr self "assertSameElements"  self expected_seq actual_seq msg ))

(defn assertSameStructure 
  "Asserts that two values contain the same structural content.

    The two arguments should be data trees consisting of trees of dicts and
    lists. They will be deeply compared by walking into the contents of dicts
    and lists; other items will be compared using the == operator.
    If the two structures differ in content, the failure message will indicate
    the location within the structures where the first difference is found.
    This may be helpful when comparing large structures.

    Mixed Sequence and Set types are supported. Mixed Mapping types are
    supported, but the order of the keys will not be considered in the
    comparison.

    Args:
      a: The first structure to compare.
      b: The second structure to compare.
      aname: Variable name to use for the first structure in assertion messages.
      bname: Variable name to use for the second structure.
      msg: Additional text to include in the failure message.
    "
  [self a b & {:keys [aname bname msg]
                       :or {msg None}} ]
    (py/call-attr-kw self "assertSameStructure" [a b] {:aname aname :bname bname :msg msg }))

(defn assertSequenceAlmostEqual 
  "An approximate equality assertion for ordered sequences.

    Fail if the two sequences are unequal as determined by their value
    differences rounded to the given number of decimal places (default 7) and
    comparing to zero, or by comparing that the difference between each value
    in the two sequences is more than the given delta.

    Note that decimal places (from zero) are usually not the same as significant
    digits (measured from the most significant digit).

    If the two sequences compare equal then they will automatically compare
    almost equal.

    Args:
      expected_seq: A sequence containing elements we are expecting.
      actual_seq: The sequence that we are testing.
      places: The number of decimal places to compare.
      msg: The message to be printed if the test fails.
      delta: The OK difference between compared values.
    "
  [ self expected_seq actual_seq places msg delta ]
  (py/call-attr self "assertSequenceAlmostEqual"  self expected_seq actual_seq places msg delta ))

(defn assertSequenceEqual 
  "An equality assertion for ordered sequences (like lists and tuples).

        For the purposes of this function, a valid ordered sequence type is one
        which can be indexed, has a length, and has an equality operator.

        Args:
            seq1: The first sequence to compare.
            seq2: The second sequence to compare.
            seq_type: The expected datatype of the sequences, or None if no
                    datatype should be enforced.
            msg: Optional message to use on failure instead of a list of
                    differences.
        "
  [ self seq1 seq2 msg seq_type ]
  (py/call-attr self "assertSequenceEqual"  self seq1 seq2 msg seq_type ))

(defn assertSequenceStartsWith 
  "An equality assertion for the beginning of ordered sequences.

    If prefix is an empty sequence, it will raise an error unless whole is also
    an empty sequence.

    If prefix is not a sequence, it will raise an error if the first element of
    whole does not match.

    Args:
      prefix: A sequence expected at the beginning of the whole parameter.
      whole: The sequence in which to look for prefix.
      msg: Optional message to report on failure.
    "
  [ self prefix whole msg ]
  (py/call-attr self "assertSequenceStartsWith"  self prefix whole msg ))

(defn assertSetEqual 
  "A set-specific equality assertion.

        Args:
            set1: The first set to compare.
            set2: The second set to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        assertSetEqual uses ducktyping to support different types of sets, and
        is optimized for sets specifically (parameters must support a
        difference method).
        "
  [ self set1 set2 msg ]
  (py/call-attr self "assertSetEqual"  self set1 set2 msg ))

(defn assertShapeEqual 
  "Asserts that a Numpy ndarray and a TensorFlow tensor have the same shape.

    Args:
      np_array: A Numpy ndarray or Numpy scalar.
      tf_tensor: A Tensor.
      msg: Optional message to report on failure.

    Raises:
      TypeError: If the arguments have the wrong type.
    "
  [ self np_array tf_tensor msg ]
  (py/call-attr self "assertShapeEqual"  self np_array tf_tensor msg ))

(defn assertStartsWith 
  "Assert that actual.startswith(expected_start) is True.

    Args:
      actual: str
      expected_start: str
      msg: Optional message to report on failure.
    "
  [ self actual expected_start msg ]
  (py/call-attr self "assertStartsWith"  self actual expected_start msg ))

(defn assertTotallyOrdered 
  "Asserts that total ordering has been implemented correctly.

    For example, say you have a class A that compares only on its attribute x.
    Comparators other than __lt__ are omitted for brevity.

    class A(object):
      def __init__(self, x, y):
        self.x = x
        self.y = y

      def __hash__(self):
        return hash(self.x)

      def __lt__(self, other):
        try:
          return self.x < other.x
        except AttributeError:
          return NotImplemented

    assertTotallyOrdered will check that instances can be ordered correctly.
    For example,

    self.assertTotallyOrdered(
      [None],  # None should come before everything else.
      [1],     # Integers sort earlier.
      [A(1, 'a')],
      [A(2, 'b')],  # 2 is after 1.
      [A(3, 'c'), A(3, 'd')],  # The second argument is irrelevant.
      [A(4, 'z')],
      ['foo'])  # Strings sort last.

    Args:
     *groups: A list of groups of elements.  Each group of elements is a list
         of objects that are equal.  The elements in each group must be less
         than the elements in the group after it.  For example, these groups are
         totally ordered: [None], [1], [2, 2], [3].
      **kwargs: optional msg keyword argument can be passed.
    "
  [ self  ]
  (py/call-attr self "assertTotallyOrdered"  self  ))

(defn assertTrue 
  "Check that the expression is true."
  [ self expr msg ]
  (py/call-attr self "assertTrue"  self expr msg ))

(defn assertTupleEqual 
  "A tuple-specific equality assertion.

        Args:
            tuple1: The first tuple to compare.
            tuple2: The second tuple to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.
        "
  [ self tuple1 tuple2 msg ]
  (py/call-attr self "assertTupleEqual"  self tuple1 tuple2 msg ))

(defn assertUrlEqual 
  "Asserts that urls are equal, ignoring ordering of query params."
  [ self a b msg ]
  (py/call-attr self "assertUrlEqual"  self a b msg ))

(defn assertWarns 
  "Fail unless a warning of class warnClass is triggered
           by the callable when invoked with specified positional and
           keyword arguments.  If a different type of warning is
           triggered, it will not be handled: depending on the other
           warning filtering rules in effect, it might be silenced, printed
           out, or raised as an exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertWarns(SomeWarning):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertWarns
           is used as a context object.

           The context manager keeps a reference to the first matching
           warning as the 'warning' attribute; similarly, the 'filename'
           and 'lineno' attributes give you information about the line
           of Python code from which the warning was triggered.
           This allows you to inspect the warning after the assertion::

               with self.assertWarns(SomeWarning) as cm:
                   do_something()
               the_warning = cm.warning
               self.assertEqual(the_warning.some_attribute, 147)
        "
  [ self expected_warning ]
  (py/call-attr self "assertWarns"  self expected_warning ))

(defn assertWarnsRegex 
  "Asserts that the message in a triggered warning matches a regexp.
        Basic functioning is similar to assertWarns() with the addition
        that only warnings whose messages also match the regular expression
        are considered successful matches.

        Args:
            expected_warning: Warning class expected to be triggered.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertWarnsRegex is used as a context manager.
        "
  [ self expected_warning expected_regex ]
  (py/call-attr self "assertWarnsRegex"  self expected_warning expected_regex ))

(defn assert- 
  ""
  [ self  ]
  (py/call-attr self "assert_"  self  ))
(defn cached-session 
  "Returns a TensorFlow Session for use in executing tests.

    This method behaves differently than self.session(): for performance reasons
    `cached_session` will by default reuse the same session within the same
    test. The session returned by this function will only be closed at the end
    of the test (in the TearDown function).

    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.

    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.cached_session(use_gpu=True) as sess:
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError(\"negative input not supported\"):
            MyOperator(invalid_input).eval()
    ```

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    "
  [self graph config  & {:keys [use_gpu force_gpu]} ]
    (py/call-attr-kw self "cached_session" [graph config] {:use_gpu use_gpu :force_gpu force_gpu }))

(defn captureWritesToStream 
  "A context manager that captures the writes to a given stream.

    This context manager captures all writes to a given stream inside of a
    `CapturedWrites` object. When this context manager is created, it yields
    the `CapturedWrites` object. The captured contents can be accessed  by
    calling `.contents()` on the `CapturedWrites`.

    For this function to work, the stream must have a file descriptor that
    can be modified using `os.dup` and `os.dup2`, and the stream must support
    a `.flush()` method. The default python sys.stdout and sys.stderr are
    examples of this. Note that this does not work in Colab or Jupyter
    notebooks, because those use alternate stdout streams.

    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        input = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.captureWritesToStream(sys.stdout) as captured:
          result = MyOperator(input).eval()
        self.assertStartsWith(captured.contents(), \"This was printed.\")
    ```

    Args:
      stream: The stream whose writes should be captured. This stream must have
        a file descriptor, support writing via using that file descriptor, and
        must have a `.flush()` method.

    Yields:
      A `CapturedWrites` object that contains all writes to the specified stream
      made during this context.
    "
  [ self stream ]
  (py/call-attr self "captureWritesToStream"  self stream ))

(defn checkedThread 
  "Returns a Thread wrapper that asserts 'target' completes successfully.

    This method should be used to create all threads in test cases, as
    otherwise there is a risk that a thread will silently fail, and/or
    assertions made in the thread will not be respected.

    Args:
      target: A callable object to be executed in the thread.
      args: The argument tuple for the target invocation. Defaults to ().
      kwargs: A dictionary of keyword arguments for the target invocation.
        Defaults to {}.

    Returns:
      A wrapper for threading.Thread that supports start() and join() methods.
    "
  [ self target args kwargs ]
  (py/call-attr self "checkedThread"  self target args kwargs ))

(defn countTestCases 
  ""
  [ self  ]
  (py/call-attr self "countTestCases"  self  ))

(defn create-tempdir 
  "Create a temporary directory specific to the test.

    NOTE: The directory and its contents will be recursively cleared before
    creation. This ensures that there is no pre-existing state.

    This creates a named directory on disk that is isolated to this test, and
    will be properly cleaned up by the test. This avoids several pitfalls of
    creating temporary directories for test purposes, as well as makes it easier
    to setup directories and verify their contents.

    See also: `create_tempfile()` for creating temporary files.

    Args:
      name: Optional name of the directory. If not given, a unique
        name will be generated and used.
      cleanup: Optional cleanup policy on when/if to remove the directory (and
        all its contents) at the end of the test. If None, then uses
        `self.tempfile_cleanup`.

    Returns:
      A _TempDir representing the created directory.
    "
  [ self name cleanup ]
  (py/call-attr self "create_tempdir"  self name cleanup ))

(defn create-tempfile 
  "Create a temporary file specific to the test.

    This creates a named file on disk that is isolated to this test, and will
    be properly cleaned up by the test. This avoids several pitfalls of
    creating temporary files for test purposes, as well as makes it easier
    to setup files, their data, read them back, and inspect them when
    a test fails.

    NOTE: This will zero-out the file. This ensures there is no pre-existing
    state.

    See also: `create_tempdir()` for creating temporary directories.

    Args:
      file_path: Optional file path for the temp file. If not given, a unique
        file name will be generated and used. Slashes are allowed in the name;
        any missing intermediate directories will be created. NOTE: This path is
        the path that will be cleaned up, including any directories in the path,
        e.g., 'foo/bar/baz.txt' will `rm -r foo`.
      content: Optional string or
        bytes to initially write to the file. If not
        specified, then an empty file is created.
      mode: Mode string to use when writing content. Only used if `content` is
        non-empty.
      encoding: Encoding to use when writing string content. Only used if
        `content` is text.
      errors: How to handle text to bytes encoding errors. Only used if
        `content` is text.
      cleanup: Optional cleanup policy on when/if to remove the directory (and
        all its contents) at the end of the test. If None, then uses
        `self.tempfile_cleanup`.

    Returns:
      A _TempFile representing the created file.
    "
  [self file_path content & {:keys [mode encoding errors cleanup]
                       :or {cleanup None}} ]
    (py/call-attr-kw self "create_tempfile" [file_path content] {:mode mode :encoding encoding :errors errors :cleanup cleanup }))

(defn debug 
  "Run the test without collecting errors in a TestResult"
  [ self  ]
  (py/call-attr self "debug"  self  ))

(defn defaultTestResult 
  ""
  [ self  ]
  (py/call-attr self "defaultTestResult"  self  ))

(defn doCleanups 
  "Execute all cleanup functions. Normally called for you after
        tearDown."
  [ self  ]
  (py/call-attr self "doCleanups"  self  ))

(defn evaluate 
  "Evaluates tensors and returns numpy values.

    Args:
      tensors: A Tensor or a nested list/tuple of Tensors.

    Returns:
      tensors numpy values.
    "
  [ self tensors ]
  (py/call-attr self "evaluate"  self tensors ))

(defn fail 
  "Fail immediately with the given message, optionally prefixed."
  [ self msg prefix ]
  (py/call-attr self "fail"  self msg prefix ))

(defn failIf 
  ""
  [ self  ]
  (py/call-attr self "failIf"  self  ))

(defn failIfAlmostEqual 
  ""
  [ self  ]
  (py/call-attr self "failIfAlmostEqual"  self  ))

(defn failIfEqual 
  ""
  [ self  ]
  (py/call-attr self "failIfEqual"  self  ))

(defn failUnless 
  ""
  [ self  ]
  (py/call-attr self "failUnless"  self  ))

(defn failUnlessAlmostEqual 
  ""
  [ self  ]
  (py/call-attr self "failUnlessAlmostEqual"  self  ))

(defn failUnlessEqual 
  ""
  [ self  ]
  (py/call-attr self "failUnlessEqual"  self  ))

(defn failUnlessRaises 
  ""
  [ self  ]
  (py/call-attr self "failUnlessRaises"  self  ))

(defn get-temp-dir 
  "Returns a unique temporary directory for the test to use.

    If you call this method multiple times during in a test, it will return the
    same folder. However, across different runs the directories will be
    different. This will ensure that across different runs tests will not be
    able to pollute each others environment.
    If you need multiple unique directories within a single test, you should
    use tempfile.mkdtemp as follows:
      tempfile.mkdtemp(dir=self.get_temp_dir()):

    Returns:
      string, the path to the unique temporary directory created for this test.
    "
  [ self  ]
  (py/call-attr self "get_temp_dir"  self  ))

(defn id 
  ""
  [ self  ]
  (py/call-attr self "id"  self  ))

(defn run 
  ""
  [ self result ]
  (py/call-attr self "run"  self result ))
(defn session 
  "Returns a TensorFlow Session for use in executing tests.

    Note that this will set this session and the graph as global defaults.

    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.

    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.session(use_gpu=True):
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError(\"negative input not supported\"):
            MyOperator(invalid_input).eval()
    ```

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    "
  [self graph config  & {:keys [use_gpu force_gpu]} ]
    (py/call-attr-kw self "session" [graph config] {:use_gpu use_gpu :force_gpu force_gpu }))

(defn setUp 
  ""
  [ self  ]
  (py/call-attr self "setUp"  self  ))

(defn shortDescription 
  "Formats both the test method name and the first line of its docstring.

    If no docstring is given, only returns the method name.

    This method overrides unittest.TestCase.shortDescription(), which
    only returns the first line of the docstring, obscuring the name
    of the test upon failure.

    Returns:
      desc: A short description of a test method.
    "
  [ self  ]
  (py/call-attr self "shortDescription"  self  ))

(defn skipTest 
  "Skip this test."
  [ self reason ]
  (py/call-attr self "skipTest"  self reason ))
(defn subTest 
  "Return a context manager that will return the enclosed block
        of code in a subtest identified by the optional message and
        keyword parameters.  A failure in the subtest marks the test
        case as failed but resumes execution at the end of the enclosed
        block, allowing further test code to be executed.
        "
  [self   & {:keys [msg]} ]
    (py/call-attr-kw self "subTest" [] {:msg msg }))

(defn tearDown 
  ""
  [ self  ]
  (py/call-attr self "tearDown"  self  ))
(defn test-session 
  "Use cached_session instead. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `self.session()` or `self.cached_session()` instead."
  [self graph config  & {:keys [use_gpu force_gpu]} ]
    (py/call-attr-kw self "test_session" [graph config] {:use_gpu use_gpu :force_gpu force_gpu }))
