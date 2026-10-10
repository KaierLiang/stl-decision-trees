# Preserved experiment semantics and known limitations

This document describes the supplied code, not an independent implementation of
every experiment or theorem in the paper. Original files are retained byte-for-byte
under `legacy/` and in the initial Git commit.

## Data and augmentation

The original scripts pool both dataset partitions, draw a sample without replacement
using NumPy's legacy random sequence at seed 42, and then apply augmentation:

1. Append a constant-zero channel to each original trajectory.
2. Find label-1 trajectories in their sampled order.
3. Copy the first half (rounded down), set their new channel to one, and change
   their labels to zero.
4. Append these copies and retain a parent-index mapping.

This construction is part of the supplied experiment. It changes the classification
problem and must not be treated as generic feature engineering on an unchanged
dataset. The legacy presets preserve it; the CLI defaults to selecting only the
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

That last point matters for the interpretation of “accuracy.” The objective counts
correctly labelled flow in the optimization model. It need not equal the accuracy
of a deterministic function of the input signals. This refactoring reports both
values, counts prediction ties at tolerance `1e-6`, and makes the prediction tie
policy explicit. `LearnedTree.predict` raises on a tie by default; callers may
select `left` or `right`. Introducing a positive margin or a forced tie policy in
the training model would be a formulation change and has not been done here.

The follow-up [full audit](validation.md) confirms that this is a correctness
defect for deterministic classification, not merely a formatting or reporting
choice: both original variants can assign total correct flow to identical inputs
with opposite labels. Any deterministic prediction rule must give those inputs
the same label. The test suite retains this counterexample explicitly; it is not
marked as a repaired behavior.

## Repairs and engineering changes

- The final binary channel is addressed by its actual position. The original
  `feature_idx + len(feature)` expression accidentally depended on the character
  count of a feature name and fails for the single-continuous-channel ECG layout.
- Constants in Gurobi min constraints use the explicit `constant` argument.
- Both variants share the same named constraint blocks and configuration types.
- Labels need not be contiguous when building a model; their actual integer IDs
  are used. The legacy augmentation remains specifically binary.
- Solver status and incumbent existence are checked before reading solution values.
- Feasible time-limited incumbents are exported with their bound/gap and status.
- The model retains read-only copies of input data, so caller mutations cannot
  change the data used to score already constructed coefficients.
- Thread count and seeds are explicit. Different solver versions or seeds may
  choose different trees with the same objective.

These repairs preserve the intended channel selection and source equations. They
do not resolve the inherited zero-robustness ambiguity or the temporal belong
restriction. The regression tests compare source and refactored objectives and
model sizes on the same actual naval trajectories.
