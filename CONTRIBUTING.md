# Contributing

Run the CPU numerical tests and formatting checks before proposing a change.
Keep time direction, prediction target, sampler endpoint and evaluation scope
explicit. Add independent mathematical oracles for scientific changes; do not
validate an equation by comparing it only with a second copy of itself.

Use separate run directories for experiments. Never replace reference records,
change a teacher/codec without recording its identity, or select checkpoints on
the test set. Include fitting and evaluation seeds, NFE, representation, sample
counts, source hashes and relevant unsuccessful baselines. Keep datasets and
large training checkpoints outside the public distribution. Original borrowed
legacy sources remain excluded until their provenance permits publication.

Public HTML must work offline, preserve failed experiments and pass small-screen
and real-browser checks. A screenshot is a preview of the actual implementation;
it is never substituted for measured outputs.
