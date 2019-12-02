(ns tensorflow.python.platform.flags.Flag
  "Information about a command-line flag.

  'Flag' objects define the following fields:
    .name - the name for this flag;
    .default - the default value for this flag;
    .default_unparsed - the unparsed default value for this flag.
    .default_as_str - default value as repr'd string, e.g., \"'true'\" (or None);
    .value - the most recent parsed value of this flag; set by parse();
    .help - a help string or None if no help is available;
    .short_name - the single letter alias for this flag (or None);
    .boolean - if 'true', this flag does not accept arguments;
    .present - true if this flag was parsed from command line flags;
    .parser - an ArgumentParser object;
    .serializer - an ArgumentSerializer object;
    .allow_override - the flag may be redefined without raising an error, and
                      newly defined flag overrides the old one.
    .allow_override_cpp - use the flag from C++ if available; the flag
                          definition is replaced by the C++ flag after init;
    .allow_hide_cpp - use the Python flag despite having a C++ flag with
                      the same name (ignore the C++ flag);
    .using_default_value - the flag value has not been set by user;
    .allow_overwrite - the flag may be parsed more than once without raising
                       an error, the last set value will be used;
    .allow_using_method_names - whether this flag can be defined even if it has
                                a name that conflicts with a FlagValues method.

  The only public method of a 'Flag' object is parse(), but it is
  typically only called by a 'FlagValues' object.  The parse() method is
  a thin wrapper around the 'ArgumentParser' parse() method.  The parsed
  value is saved in .value, and the .present attribute is updated.  If
  this flag was already present, an Error is raised.

  parse() is also called during __init__ to parse the default value and
  initialize the .value attribute.  This enables other python modules to
  safely use flags even if the __main__ module neglects to parse the
  command line arguments.  The .present attribute is cleared after
  __init__ parsing.  If the default value is set to None, then the
  __init__ parsing step is skipped and the .value attribute is
  initialized to None.

  Note: The default value is also presented to the user in the help
  string, so it is important that it be a legal value for this flag.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))
(defn Flag 
  "Information about a command-line flag.

  'Flag' objects define the following fields:
    .name - the name for this flag;
    .default - the default value for this flag;
    .default_unparsed - the unparsed default value for this flag.
    .default_as_str - default value as repr'd string, e.g., \"'true'\" (or None);
    .value - the most recent parsed value of this flag; set by parse();
    .help - a help string or None if no help is available;
    .short_name - the single letter alias for this flag (or None);
    .boolean - if 'true', this flag does not accept arguments;
    .present - true if this flag was parsed from command line flags;
    .parser - an ArgumentParser object;
    .serializer - an ArgumentSerializer object;
    .allow_override - the flag may be redefined without raising an error, and
                      newly defined flag overrides the old one.
    .allow_override_cpp - use the flag from C++ if available; the flag
                          definition is replaced by the C++ flag after init;
    .allow_hide_cpp - use the Python flag despite having a C++ flag with
                      the same name (ignore the C++ flag);
    .using_default_value - the flag value has not been set by user;
    .allow_overwrite - the flag may be parsed more than once without raising
                       an error, the last set value will be used;
    .allow_using_method_names - whether this flag can be defined even if it has
                                a name that conflicts with a FlagValues method.

  The only public method of a 'Flag' object is parse(), but it is
  typically only called by a 'FlagValues' object.  The parse() method is
  a thin wrapper around the 'ArgumentParser' parse() method.  The parsed
  value is saved in .value, and the .present attribute is updated.  If
  this flag was already present, an Error is raised.

  parse() is also called during __init__ to parse the default value and
  initialize the .value attribute.  This enables other python modules to
  safely use flags even if the __main__ module neglects to parse the
  command line arguments.  The .present attribute is cleared after
  __init__ parsing.  If the default value is set to None, then the
  __init__ parsing step is skipped and the .value attribute is
  initialized to None.

  Note: The default value is also presented to the user in the help
  string, so it is important that it be a legal value for this flag.
  "
  [parser serializer name default help_string short_name  & {:keys [boolean allow_override allow_override_cpp allow_hide_cpp allow_overwrite allow_using_method_names]} ]
    (py/call-attr-kw flags "Flag" [parser serializer name default help_string short_name] {:boolean boolean :allow_override allow_override :allow_override_cpp allow_override_cpp :allow_hide_cpp allow_hide_cpp :allow_overwrite allow_overwrite :allow_using_method_names allow_using_method_names }))

(defn flag-type 
  "Returns a str that describes the type of the flag.

    NOTE: we use strings, and not the types.*Type constants because
    our flags can have more exotic types, e.g., 'comma separated list
    of strings', 'whitespace separated list of strings', etc.
    "
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "Parses string and sets flag value.

    Args:
      argument: str or the correct flag value type, argument to be parsed.
    "
  [ self argument ]
  (py/call-attr self "parse"  self argument ))

(defn serialize 
  "Serializes the flag."
  [ self  ]
  (py/call-attr self "serialize"  self  ))

(defn unparse 
  ""
  [ self  ]
  (py/call-attr self "unparse"  self  ))

(defn value 
  ""
  [ self ]
    (py/call-attr self "value"))
