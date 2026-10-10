# Preserved experiment semantics and known limitations

This document describes the supplied code, not an independent implementation of
every experiment or theorem in the paper. Original files are retained byte-for-byte
in the initial Git commit, which the source-comparison tests read directly.

## Data and augmentation

The original scripts pool both dataset partitions, draw a sample without replacement
using NumPy's RandomState sequence at seed 42, and then apply augmentation:

1. Append a constant-zero channel to each original trajectory.
2. Find label-1 trajectories in their sampled order.
3. Copy the first half (rounded down), set their new channel to one, and change
   their labels to zero.
4. Append these copies and retain a parent-index mapping.

This construction is part of the supplied experiment. It changes the classification
problem and must not be treated as generic feature engineering on an unchanged
dataset. The naval preset preserves it; the CLI defaults to selecting only the
training partition. The quickstart also uses only that partition. No published
paper score is asserted by this refactoring.

## Tree and flow

Nodes are zero-indexed. Children of node `j` are `2*j + 1` and `2*j + 2`. A depth
of three allocates 15 nodes, with seven potential decision nodes. Internal nodes
may select a predicate or terminate with a class label. Terminal nodes at the
maximum depth must select a label.

Binary source flow may be zero for a sample the model does not classify correctly.
Flow is conserved through the tree and may leave for the sink only at a node
labelled with that sample's ground-truth class. The objective maximizes total sink
flow, with no tree-size or window-length penalty. The exported tree omits unused
descendants of an already labelled node; this does not prune the optimization model.

## Continuous primitives

For each continuous feature the model selects either `F[a,b](s <= threshold)` or
`F[a,b](s >= threshold)`. Intervals include both endpoints. Their robustness is
`threshold - min(s[a:b+1])` and `max(s[a:b+1]) - threshold`, respectively.

The right branch is the negation of the selected predicate. The refactoring
preserves the original operator family and all intervals satisfying
`0 <= a <= b < horizon` and `b - a <= max_window_span`, including singleton
intervals. No candidate thresholds or tree templates are supplied in advance.

The implementation caches signal extrema to avoid recomputing coefficients at
every tree node. It does not coarsen, filter, or sample the interval family.

## Discrete variants

| Mode | Source script | Equations preserved |
| --- | --- | --- |
| `exist` | `main_belong.py` | Discrete branch value is the minimum of its selection variable and the maximum of the final binary channel over the full horizon. A discrete node selects no temporal interval. |
| `belong` | `main_belong_keep_temporal_feature.py` | Discrete and continuous predicates select an interval. Both the min and max of the binary channel on that interval are equated to the same variable. |

Consequently, `belong` forces that binary channel to be constant within each
selected interval for every training sample, even at a node selecting a continuous
predicate. This follows from the source constraints being unconditional. It is
automatically satisfied by the supplied constant-channel augmentation. The code
retains this constraint; interpreting it as a general eventually-binary predicate
would change the feasible set. Saved-tree prediction rejects a varying binary
channel at an active `belong` node.

## Thresholds, bounds, and equality

- Thresholds have lower bound zero, as in the original Gurobi defaults.
- Every continuous robustness variable is bounded by `[-200, 200]` by default.
- The sign-encoding big-M is 250, with the original zero margin.
- These bounds apply even to unselected continuous predicates. They can constrain
  the feasible model on data with different scales; they are not automatically
  inferred from the data or silently widened.
- At zero robustness both signs are feasible. Different samples can therefore
  take different branches at the same threshold equality in the MILP.

The intended setting, clarified by the author, assumes noisy data and excludes
zero robustness. The zero-margin training equations remain unchanged. Since a
threshold is learned from the samples, noise alone does not guarantee that an
optimizer returns thresholds strictly between sample extrema. Finite-precision
solutions can also sit a few trillionths from a boundary.

That matters for the interpretation of “accuracy.” The objective counts
correctly labelled flow in the optimization model. It need not equal the accuracy
of a deterministic function of the input signals. This refactoring reports both
values, counts prediction ties at tolerance `1e-6`, and makes the prediction tie
policy explicit. `LearnedTree.predict` raises on a tie by default; callers may
select `left` or `right`. Introducing a positive margin or a forced tie policy in
the training model would be a formulation change and has not been done here.

The equality counterexample with identical inputs and opposite labels is outside
the stated nonzero-robustness assumption. It remains an explicit boundary test;
it is not evidence against the method under that assumption. The raw naval
export separately revealed a numerical issue: robustness near `1e-12` could be
routed against its sign within solver feasibility tolerance. Reducing prediction
tolerance to zero did not fix that mismatch.

### Numerically stable export

For a continuous node near a routed boundary, export intersects the original
threshold bounds, **all** continuous robustness bounds (including unselected
features/operations), and the incumbent's left/right routing requirements.
When this interval has a numerically stable interior, it selects its midpoint.
Otherwise it retains the raw threshold and reports the unresolved boundary.
No training margin, additional constraint, window pruning, or changed sample is
introduced. The solver model and raw solution are untouched. The exported tree
may choose a different prediction for unseen signals near the old threshold,
as is possible when selecting a different equally optimal threshold.

`tree.json` is the resulting export; `tree-raw.json` retains the exact solver
thresholds. `summary.json` records changes, raw prediction accuracy, and whether
all admitted sample flows have verified strict routing. Prediction ties remain
explicit. `solve(..., stabilize=False)` disables this step; use it for custom
models with extra constraints or objectives restricting thresholds.

The [full audit](validation.md) independently reconstructs the export's auxiliary
variables and checks every original linear/general constraint and the objective,
in addition to checking predictions. An empty strict interval is never treated
as a successful verification of the nonzero-robustness assumption.

## Repairs and engineering changes

- The final binary channel is addressed by its actual position. The original
  `feature_idx + len(feature)` expression accidentally depended on the character
  count of a feature name and fails for the single-continuous-channel ECG layout.
- Constants in Gurobi min constraints use the explicit `constant` argument.
- Both variants share the same named constraint blocks and configuration types.
- Labels need not be contiguous when building a model; their actual integer IDs
  are used. The discrete-feature augmentation remains specifically binary.
- Solver status and incumbent existence are checked before reading solution values.
- Feasible time-limited incumbents are exported with their bound/gap and status.
- The model retains read-only copies of input data, so caller mutations cannot
  change the data used to score already constructed coefficients.
- Thread count and seeds are explicit. Different solver versions or seeds may
  choose different trees with the same objective.

These repairs preserve the intended channel selection and source equations.
Zero robustness remains outside the stated assumption, and the temporal belong
restriction remains in force. Regression tests compare source and refactored
objectives and constraints on the same actual naval trajectories.
