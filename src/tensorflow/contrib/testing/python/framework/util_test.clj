(ns tensorflow.contrib.testing.python.framework.util-test
  "Test utilities."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce util-test (import-module "tensorflow.contrib.testing.python.framework.util_test"))

(defn assert-summary 
  "Asserts summary contains the specified tags and values.

  Args:
    expected_tags: All tags in summary.
    expected_simple_values: Simply values for some tags.
    summary_proto: Summary to validate.

  Raises:
    ValueError: if expectations are not met.
  "
  [ expected_tags expected_simple_values summary_proto ]
  (py/call-attr util-test "assert_summary"  expected_tags expected_simple_values summary_proto ))

(defn latest-event-file 
  "Find latest event file in `base_dir`.

  Args:
    base_dir: Base directory in which TF event flies are stored.
  Returns:
    File path, or `None` if none exists.
  "
  [ base_dir ]
  (py/call-attr util-test "latest_event_file"  base_dir ))

(defn latest-events 
  "Parse events from latest event file in base_dir.

  Args:
    base_dir: Base directory in which TF event flies are stored.
  Returns:
    Iterable of event protos.
  Raises:
    ValueError: if no event files exist under base_dir.
  "
  [ base_dir ]
  (py/call-attr util-test "latest_events"  base_dir ))

(defn latest-summaries 
  "Parse summary events from latest event file in base_dir.

  Args:
    base_dir: Base directory in which TF event flies are stored.
  Returns:
    List of event protos.
  Raises:
    ValueError: if no event files exist under base_dir.
  "
  [ base_dir ]
  (py/call-attr util-test "latest_summaries"  base_dir ))

(defn simple-values-from-events 
  "Parse summaries from events with simple_value.

  Args:
    events: List of tensorflow.Event protos.
    tags: List of string event tags corresponding to simple_value summaries.
  Returns:
    dict of tag:value.
  Raises:
   ValueError: if a summary with a specified tag does not contain simple_value.
  "
  [ events tags ]
  (py/call-attr util-test "simple_values_from_events"  events tags ))

(defn to-summary-proto 
  "Create summary based on latest stats.

  Args:
    summary_str: Serialized summary.
  Returns:
    summary_pb2.Summary.
  Raises:
    ValueError: if tensor is not a valid summary tensor.
  "
  [ summary_str ]
  (py/call-attr util-test "to_summary_proto"  summary_str ))
