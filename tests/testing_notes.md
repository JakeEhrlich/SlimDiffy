testing notes:
* need to test each method in pytree
* need to fuzz test every combination operations for both jit and grad
* need to test every operator at least once
* need to test from_sequence with predicates of various kinds, probably with fuzz testing from_sequence/to_sequence
* should more heavily fuzz test the from_value/to_value
* need to fuzz test mapkeys heavily because its very load bearing
* need better unit testing for freeze as its somewhat load bearing
* need better non-trivial unit testing for things like training and some physics examples
* should fuzz test optimizations better
* add unit tests for each Tracer method
* shouldn't be hard to get full coverage
* need better fuzz testing for batch_contract
* should have unit tests for unindex_pytree, index_pytree, etc...
* I think what underpins this all is generating things?

1) I need to generate pytrees in order to property test pytree
2) I need to be able to generate programs in order to property test all sorts of things
3) I need claude to just crank away at unit tests somehow so that we get coverage

pytree properties:
* to_value and from_value are inverses
* static fields do not affect to_value/from_value
* static fields never get mapped over
* to_value(leaf(x)) = x
* to_value(from_dict(x)) = x
* to_value(from_sequence(x)) = x (fuzz over statics)
* mapkeys over the identity is an identity
* map f followed by map g is the same as mapping f then g
* mapkeys of a dict picks up exactly the fields we expect
* mapkeys of a sequence picks up exactly the fields we expect
* mapkeys of nested dicts/sequences pick up exactly what we expect
* to_sequence and from_sequence are inverses
* freeze needs a special "unfreezes to" predicate for testing to define its complex behavoir

I wonder if there's a "python fuzzing" library that would
work well for this?

autodiff properties:
* for any generated python program on numpy arrays, jitting it does not change its behavoir
* need unit tests for everything to get full coverage just from unit tests
* need unit tests that validate that batch_contract behaves like we expect
* for any valid generated program, we can find its grad
* for any valid generated program, we can find its jacobian
* if we differentiate a small program and then perform monte carlo integration results should be close
* for each operator, if we differentiate it the result should be close to the method of finite differences
* for any generated program, DCE does not affect its behavoir
* for any generated program, CSE does not affect its behaovir
* for any generated program, constant folding does not affect its behavoir
* for any generated program, algebraic simplification does not affect its behavoir
* test all pairs of pipelines other than Gradiant
* test that grad followed by an optimization is always the same as just the gradiant
* for any program, adding static arguments does not affect the programs behavoir
* jit is idempotent

I think having Claude generate a courpus of functions
and then paramterizing tests with that list is a solid idea.
It also means I can just add example functions to the list as I find bugs!
That will make keeping things well tested easy and I think Claude should be able
to do a good job if given the right instructions.
