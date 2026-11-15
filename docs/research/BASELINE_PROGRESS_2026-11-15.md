# Baseline checkpoint observations — 15 November 2026

Latest assessment at 23:47 UTC: the preserved L0 n3 checkpoint is now at 280,000.
Its raw validation prediction error continues to decrease; the later update below
supersedes the 250,000-step snapshot for current descriptive measurements.

Update at 23:35 UTC: L0 n1 has now also completed 303,000 steps and passed its
actual artifact audit. The active arm is L0 n3. Its frozen 250,000-step checkpoint
is assessed in the update below; the earlier 22:11 capture remains historical.

This report records the reviewed observations captured at 22:11 UTC: completed
L1 n1 and L1 n3 runs, and a frozen L0 n1 checkpoint at 190,000 steps. Training
continues beyond that snapshot; consult the runtime record for current status.
The new analysis reads existing logs and checkpoint metadata on CPU. It executes
no model forward pass, optimizer update, physical probe or planning evaluation.

## What differs between the runs

Every scene contains one red ball moved by actions. The count `n` denotes the
additional white balls, which move autonomously and bounce off walls. Balls pass
through one another; this environment does not test contact dynamics or walkers
reacting to the robot.

| Run in continuation order | White balls | Training objective | Snapshot status |
|---|---:|---|---|
| L1 n1 | 1 | Latent prediction, SIGReg and position-derived Fourier supervision | Completed at 303,000 |
| L1 n3 | 3 | Same supervised recipe | Completed at 303,000 |
| L0 n1 | 1 | Latent prediction and SIGReg; no auxiliary head or position loss | Captured at 190,000 |

The first-to-second comparison changes scene complexity. The supervision
comparison is L0 n1 against L1 n1, with the same underlying model configuration,
data, training schedule and seed. Comparing the current run directly with L1 n3
would also change the number of objects. These are the original predefined
experiments, resumed from preserved checkpoints. The reboot and serial execution
changed process handling; the model recipes were retained.

L1 is the supervised reference. O0/O1, which test an object-based representation
without privileged position supervision, remain pending. The original local
adaptations also remain: predictor width 128 instead of the paper's 64, and an
auxiliary head updated through both encoded and predicted branches, whereas the
paper freezes its parameters on the predicted branch.

## Completed second run: L1 n3

The actual A015 artifact audit and A014 completion review passed. The final and
full checkpoints agree at 303,000 steps, with 198,000 resumed optimizer updates,
no prior optimizer-counter deficit, and complete finite logs. This establishes
artifact integrity and continuity; final A003 inference validation is pending.

| Logged validation quantity | L1 n1 at 303,000 | L1 n3 at 303,000 |
|---|---:|---:|
| Latent prediction MSE | 0.00423250 | 0.00841325 |
| Auxiliary MSE from encoded states | 0.00355318 | 0.00983811 |
| Auxiliary MSE from predicted states | 0.03016425 | 0.08582265 |

L1 n3's latent prediction MSE decreased only 2.40% from 280,000 to 303,000,
with a small rebound from its 300,000-step measurement. The logged objective
was slowing near the end of the scheduled run. That does not establish
convergence or justify extending the schedule.

Its predicted-state auxiliary error is 8.72 times its encoded-state error.
This motivates separate physical future-position measurements. The auxiliary
target is a sum of Fourier features, so these errors are not pixel distances or
proof that each object is individually represented. The n1/n3 raw loss ranking
also does not rank physical prediction or planning ability.

## Earlier arm: L0 n1 at the 22:11 frozen snapshot

The captured checkpoint identifies L0, n1 and step 190,000, with no auxiliary
head. Its last available validation in the captured history is at 180,000.
The latest ten training rows, from 185,500 through 190,000, have median latent
prediction MSE 0.004755 and median throughput 43.1 steps/s.

| Logged validation prediction MSE | Value |
|---|---:|
| L0 n1 at 160,000 | 0.00801856 |
| L0 n1 at 180,000 | 0.00685314 |
| L1 n1 at matched 180,000 | 0.12079418 |

L0's prediction term decreased 14.53% over those 20,000 steps. Its raw value
at 180,000 is 17.63 times lower than L1's at the same step. This observation
does not establish L0 superiority or prove that it has omitted the red ball.
The learned representations differ, and saved normalization statistics are
another confound. L0 may still perform well on the eventual physical tests;
that outcome must be retained if observed.

## Interpretation and next discriminating tests

[Zhu et al., sections 3–4 and Appendix A.10](https://arxiv.org/html/2609.04264v3)
motivate separating prediction loss from physical information and planning:
a representation can avoid global collapse while omitting an important object.
Their finite Fourier target also does not uniquely identify every possible
configuration. The relevant hypothesis here is that L0 can obtain low latent
error while losing information needed to control the red ball. These logs alone
cannot decide that hypothesis.

[Ioffe and Szegedy, section 3.1 and Algorithm 2](https://arxiv.org/pdf/1502.03167v3)
distinguish training minibatch statistics from population statistics used at
inference. Consequently, even matched-step raw losses do not replace the common
train-only calibration and numerical checks planned for the final artifacts.

After the remaining originals finish and pass their individual audits and
literature reviews, the required measurements remain: final A003 validity;
position readouts for controlled and autonomous objects; future-position probes
and action controls; and paired planning on the fixed tasks. The earlier A017
physical gains apply only to L1 n1 at 220,000, not to L1 n3 or this L0 checkpoint.
No robot transfer or scientific performance gate is established by this report.

## Update: L0 n3 at 250,000 steps

The fourth continuation now tests the same three-white-ball scenes as the second
run, L1 n3. It retains the base model, data, schedule and seed while removing the
Fourier position head and its auxiliary losses. This is the original independent
L0 arm, resumed from 155,000, rather than a fine-tune of L1. No recipe was changed
in response to the second run's results.

The preserved checkpoint is at 250,000, with its latest validation at 240,000.
Training continues beyond the captured snapshot. All captured log values are
finite, and the latest ten training rows have median throughput 42.95 steps/s.

| Raw validation latent prediction MSE | L0 n3 | L1 n3 |
|---|---:|---:|
| 200,000 | 0.00230679 | 0.02153805 |
| 220,000 | 0.00269522 | 0.01216016 |
| 240,000 | 0.00147287 | 0.01369141 |

L0's prediction term decreased 45.35% from 220,000 to 240,000, following a rise
over the previous interval. At matched 240,000, L1's raw value is 9.30 times
larger. These are losses in separately learned latent spaces, affected by saved
normalization statistics. They cannot distinguish useful L0 dynamics from a
representation that makes prediction easier by omitting important information.
The completed L0 n1 final latent loss, 0.000737697, has the same interpretive limit.

The primary sources above were revisited before capture and after the result.
The paper's Appendix A.10 also reports weaker environmental-object readouts in
some supervised cases. Accordingly, test controlled and autonomous objects
separately and retain unexpected L0 success. Current/future position readouts,
action controls and paired planning remain the decisive measurements after common
final inference checks. This update establishes no physical or robot performance
gain.

The exact capture and review are under
`outputs/gated/A014/user_assessment_n3_final_l0n3_current/`:

- Result SHA256: `52c8d27bc16b3de2bb319c7da976623d4e78081a7cfaed49133250d4e68a6ad5`.
- Review SHA256: `23f47ab9c9fdd170dbe7a1f1cf91ffba61438e9a31efae5dc045adfb65f3b90a`.
- Checkpoint SHA256: `13ed29efd7241b8a96335da122d64cfb0d710182c527df5a7d4f5653bcc37640`.

An initial summary-script field-name error was preserved and source-checked before
correction. The same snapshot was retained; training and source artifacts were
untouched. Digests, complete log coverage, validation rows and medians were
independently checked. No model forward pass or GPU work was performed.

## Update: L0 n3 at 280,000 steps

The latest available checkpoint was copied without interrupting training and
read for metadata only on CPU. Complete captured logs are finite and cover every
expected training and validation step. No model evaluation was executed.

| Validation step | L0 n3 latent prediction MSE |
|---|---:|
| 240,000 | 0.00147287 |
| 260,000 | 0.00102291 |
| 280,000 | 0.00080837 |

The prediction term decreased 21.0% over the latest interval and 45.1% over
40,000 steps. Matched-step L1 n3 has raw error 0.00862006, 10.66 times larger.
This updates the loss trajectory without changing the scientific conclusion:
separate latent spaces and normalization prevent this ratio from ranking physical
prediction or planning. The sources above were revisited before capture and after
the result. No experiment recipe or next-stage protocol changed.

The capture and exact-result review are preserved under
`outputs/gated/A014/checkpoint_assessment_l0n3_latest/`.
Result SHA256: `b65479c7e76f73415e243778d763222f1d829db63869a24af0b6dbd729e8c374`.
The checkpoint digest, source inode/mtime, input logs and numerical verification
are recorded there. The earlier captures remain intact.

## Preserved evidence

The capture, copied logs/configurations, checkpoint metadata and substantive
post-result literature review are under
`outputs/gated/A014/user_assessment_n3_final_l0n1_current/`.

- `result.json` SHA256:
  `33c4ef37bc259596d77d4984b08941f8368d4753f3d376e497068781f9cf8ae7`
- `review.json` SHA256:
  `402ccaa9cbd3267240ba8d7c75f18dec4aa2b28e36ebd10b8f0090710faf8bb7`
- Captured L0 n1 checkpoint SHA256:
  `a71fc80edf4ff1e809b0e5d0d3146cbc52a554f7c8ba09d963e41bf46a0ff334`

The completed L1 artifact audits are separately preserved under
`outputs/gated/A015/finals/l1_n1_s0/` and `l1_n3_s0/`, with their continuation
reviews under `outputs/gated/A014/runs/`. The capture review passes provenance,
arithmetic and bounded descriptive interpretation only. Its existence does not
pass the pending numerical or scientific gates.
