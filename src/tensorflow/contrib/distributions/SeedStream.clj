(ns tensorflow.contrib.distributions.SeedStream
  "Local PRNG for amplifying seed entropy into seeds for base operations.

  Writing sampling code which correctly sets the pseudo-random number
  generator (PRNG) seed is surprisingly difficult.  This class serves as
  a helper for the TensorFlow Probability coding pattern designed to
  avoid common mistakes.

  # Motivating Example

  A common first-cut implementation of a sampler for the beta
  distribution is to compute the ratio of a gamma with itself plus
  another gamma.  This code snippet tries to do that, but contains a
  surprisingly common error:

  ```python
  def broken_beta(shape, alpha, beta, seed):
    x = tf.random.gamma(shape, alpha, seed=seed)
    y = tf.random.gamma(shape, beta, seed=seed)
    return x / (x + y)
  ```

  The mistake is that the two gamma draws are seeded with the same
  seed.  This causes them to always produce the same results, which,
  in turn, leads this code snippet to always return `0.5`.  Because it
  can happen across abstraction boundaries, this kind of error is
  surprisingly easy to make when handling immutable seeds.

  # Goals

  TensorFlow Probability adopts a code style designed to eliminate the
  above class of error, without exacerbating others.  The goals of
  this code style are:

  - Support reproducibility of results (by encouraging seeding of all
    pseudo-random operations).

  - Avoid shared-write global state (by not relying on a global PRNG).

  - Prevent accidental seed reuse by TF Probability implementers.  This
    goal is served with the local pseudo-random seed generator provided
    in this module.

  - Mitigate potential accidental seed reuse by TF Probability clients
    (with a salting scheme).

  - Prevent accidental resonances with downstream PRNGs (by hashing the
    output).

  ## Non-goals

  - Implementing a high-performance PRNG for generating large amounts of
    entropy.  That's the job of the underlying TensorFlow PRNG we are
    seeding.

  - Avoiding random seed collisions, aka \"birthday attacks\".

  # Code pattern

  ```python
  def random_beta(shape, alpha, beta, seed):        # (a)
    seed = SeedStream(seed, salt=\"random_beta\")     # (b)
    x = tf.random.gamma(shape, alpha, seed=seed())  # (c)
    y = tf.random.gamma(shape, beta, seed=seed())   # (c)
    return x / (x + y)
  ```

  The elements of this pattern are:

  - Accept an explicit seed (line a) as an argument in all public
    functions, and write the function to be deterministic (up to any
    numerical issues) for fixed seed.

    - Rationale: This provides the client with the ability to reproduce
      results.  Accepting an immutable seed rather than a mutable PRNG
      object reduces code coupling, permitting different sections to be
      reproducible independently.

  - Use that seed only to initialize a local `SeedStream` instance (line b).

    - Rationale: Avoids accidental seed reuse.

  - Supply the name of the function being implemented as a salt to the
    `SeedStream` instance (line b).  This serves to keep the salts
    unique; unique salts ensure that clients of TF Probability will see
    different functions always produce independent results even if
    called with the same seeds.

  - Seed each callee operation with the output of a unique call to the
    `SeedStream` instance (lines c).  This ensures reproducibility of
    results while preventing seed reuse across callee invocations.

  # Why salt?

  Salting the `SeedStream` instances (with unique salts) is defensive
  programming against a client accidentally committing a mistake
  similar to our motivating example.  Consider the following situation
  that might arise without salting:

  ```python
  def tfp_foo(seed):
    seed = SeedStream(seed, salt=\"\")
    foo_stuff = tf.random.normal(seed=seed())
    ...

  def tfp_bar(seed):
    seed = SeedStream(seed, salt=\"\")
    bar_stuff = tf.random.normal(seed=seed())
    ...

  def client_baz(seed):
    foo = tfp_foo(seed=seed)
    bar = tfp_bar(seed=seed)
    ...
  ```

  The client should have used different seeds as inputs to `foo` and
  `bar`.  However, because they didn't, *and because `foo` and `bar`
  both sample a Gaussian internally as their first action*, the
  internal `foo_stuff` and `bar_stuff` will be the same, and the
  returned `foo` and `bar` will not be independent, leading to subtly
  incorrect answers from the client's simulation.  This kind of bug is
  particularly insidious for the client, because it depends on a
  Distributions implementation detail, namely the order in which `foo`
  and `bar` invoke the samplers they depend on.  In particular, a
  Bayesflow team member can introduce such a bug in previously
  (accidentally) correct client code by performing an internal
  refactoring that causes this operation order alignment.

  A salting discipline eliminates this problem by making sure that the
  seeds seen by `foo`'s callees will differ from those seen by `bar`'s
  callees, even if `foo` and `bar` are invoked with the same input
  seed.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributions (import-module "tensorflow.contrib.distributions"))

(defn SeedStream 
  "Local PRNG for amplifying seed entropy into seeds for base operations.

  Writing sampling code which correctly sets the pseudo-random number
  generator (PRNG) seed is surprisingly difficult.  This class serves as
  a helper for the TensorFlow Probability coding pattern designed to
  avoid common mistakes.

  # Motivating Example

  A common first-cut implementation of a sampler for the beta
  distribution is to compute the ratio of a gamma with itself plus
  another gamma.  This code snippet tries to do that, but contains a
  surprisingly common error:

  ```python
  def broken_beta(shape, alpha, beta, seed):
    x = tf.random.gamma(shape, alpha, seed=seed)
    y = tf.random.gamma(shape, beta, seed=seed)
    return x / (x + y)
  ```

  The mistake is that the two gamma draws are seeded with the same
  seed.  This causes them to always produce the same results, which,
  in turn, leads this code snippet to always return `0.5`.  Because it
  can happen across abstraction boundaries, this kind of error is
  surprisingly easy to make when handling immutable seeds.

  # Goals

  TensorFlow Probability adopts a code style designed to eliminate the
  above class of error, without exacerbating others.  The goals of
  this code style are:

  - Support reproducibility of results (by encouraging seeding of all
    pseudo-random operations).

  - Avoid shared-write global state (by not relying on a global PRNG).

  - Prevent accidental seed reuse by TF Probability implementers.  This
    goal is served with the local pseudo-random seed generator provided
    in this module.

  - Mitigate potential accidental seed reuse by TF Probability clients
    (with a salting scheme).

  - Prevent accidental resonances with downstream PRNGs (by hashing the
    output).

  ## Non-goals

  - Implementing a high-performance PRNG for generating large amounts of
    entropy.  That's the job of the underlying TensorFlow PRNG we are
    seeding.

  - Avoiding random seed collisions, aka \"birthday attacks\".

  # Code pattern

  ```python
  def random_beta(shape, alpha, beta, seed):        # (a)
    seed = SeedStream(seed, salt=\"random_beta\")     # (b)
    x = tf.random.gamma(shape, alpha, seed=seed())  # (c)
    y = tf.random.gamma(shape, beta, seed=seed())   # (c)
    return x / (x + y)
  ```

  The elements of this pattern are:

  - Accept an explicit seed (line a) as an argument in all public
    functions, and write the function to be deterministic (up to any
    numerical issues) for fixed seed.

    - Rationale: This provides the client with the ability to reproduce
      results.  Accepting an immutable seed rather than a mutable PRNG
      object reduces code coupling, permitting different sections to be
      reproducible independently.

  - Use that seed only to initialize a local `SeedStream` instance (line b).

    - Rationale: Avoids accidental seed reuse.

  - Supply the name of the function being implemented as a salt to the
    `SeedStream` instance (line b).  This serves to keep the salts
    unique; unique salts ensure that clients of TF Probability will see
    different functions always produce independent results even if
    called with the same seeds.

  - Seed each callee operation with the output of a unique call to the
    `SeedStream` instance (lines c).  This ensures reproducibility of
    results while preventing seed reuse across callee invocations.

  # Why salt?

  Salting the `SeedStream` instances (with unique salts) is defensive
  programming against a client accidentally committing a mistake
  similar to our motivating example.  Consider the following situation
  that might arise without salting:

  ```python
  def tfp_foo(seed):
    seed = SeedStream(seed, salt=\"\")
    foo_stuff = tf.random.normal(seed=seed())
    ...

  def tfp_bar(seed):
    seed = SeedStream(seed, salt=\"\")
    bar_stuff = tf.random.normal(seed=seed())
    ...

  def client_baz(seed):
    foo = tfp_foo(seed=seed)
    bar = tfp_bar(seed=seed)
    ...
  ```

  The client should have used different seeds as inputs to `foo` and
  `bar`.  However, because they didn't, *and because `foo` and `bar`
  both sample a Gaussian internally as their first action*, the
  internal `foo_stuff` and `bar_stuff` will be the same, and the
  returned `foo` and `bar` will not be independent, leading to subtly
  incorrect answers from the client's simulation.  This kind of bug is
  particularly insidious for the client, because it depends on a
  Distributions implementation detail, namely the order in which `foo`
  and `bar` invoke the samplers they depend on.  In particular, a
  Bayesflow team member can introduce such a bug in previously
  (accidentally) correct client code by performing an internal
  refactoring that causes this operation order alignment.

  A salting discipline eliminates this problem by making sure that the
  seeds seen by `foo`'s callees will differ from those seen by `bar`'s
  callees, even if `foo` and `bar` are invoked with the same input
  seed.
  "
  [ seed salt ]
  (py/call-attr distributions "SeedStream"  seed salt ))

(defn original-seed 
  ""
  [ self ]
    (py/call-attr self "original_seed"))

(defn salt 
  ""
  [ self ]
    (py/call-attr self "salt"))
