# TDN architecture proposal: evidence and verification

The [proposal](../../docs/TDN_ARCHITECTURE_PROPOSAL_20261015.md) specifies a
response-preconditioned defect-correction solver, classical and learned
comparators, conditional optimization guarantees, an independent residual
auditor, staged acceptance and a term legend. It is a research design, not
implemented or measured superiority. Historical A PASS and B STOP are unchanged.

- [Primary-source review](primary-source-review.json): relevant sections of
  the three user-supplied papers independently retrieved and read.
- [Retrieval hashes](literature-retrieval.json): eight primary HTML/PDF
  responses; cached source bytes remain in the ignored project-local runtime
  directory. This is a targeted review, not a proof of novelty.
- [Existing numerical checks](readiness-tests.xml): 40 CPU tests passed, no
  failures/errors/skips. These validate restored development capabilities and
  existing numerical identities; they do not test the proposed architecture.
- [Verification](verification.json): source identity, command, environment,
  artifact checks and limitations. No new native Slurm/GPU work was launched.

The adversarial design review identified a short-time expressiveness problem:
a bounded tau-cubed two-sided heat remainder has a diagonal leading term,
while leading splitting error can be off-diagonal. The final proposal includes
explicit sparse commutator features and charges their stencil work. Their
stiff stability and utility remain open experimental questions.
