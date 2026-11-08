# Pure v44: v43 prompt boundary candidate

This is an isolated source candidate. The original v43 and pure v44 checkpoints,
source bundles, and saved evaluation records were not modified. No learning or
model generation was run for this candidate.

## Evidence and change

The actual v43 training ZIP (`Grand_cotraining_v43.zip`, SHA-256
`0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198`)
contains task-specific prompts. Math starts `問題: …\n思考:`, while Dolly
instruction examples start `指示: …`, optionally contain `\n文脈: …`, and end
`\n回答:`. In the inspected rows, `full_text = prompt + " " + target`; the
space is the first answer character the model learns to emit. See
`v43_boundary_examples.json` and `v43_dolly_anchor_samples.json`.

The original pure v44 public-curriculum code used `ユーザー: …\nアシスタント: `
for Numina, Magicoder, and Dolly. The candidate changes only those three
prompt builders: Numina math to `問題: …\n思考:`, Magicoder code to
`指示: …\n回答:`, and Dolly to `指示: …\n文脈: …\n回答:`. It adds the v43 boundary
space as a supervised first target character. Numina tool output is normalized
before this space is inserted, then all role spans shift by one; tool output
remains masked. Source-answer SHA is still computed before the added space.
Question grouping and split identity remain based on the original source
question. Existing v43 anchor rows and their exact boundaries are untouched.

The four Tier 8 fixed prompts use the same math or instruction boundaries.
Tier 1–7 prompt construction, model architecture, ALU-free Python tool runtime,
optimizer, corpus counts, and generation limits were not changed.

## CPU verification

Run `python test_v43_prompt_boundary.py` in this directory. It passed **5/5**
tests on 2026-11-03. The checks used three pinned public source examples,
four actual saved public test records for shape only, and v43's actual Dolly
train/dev/test records. The curriculum fixture admitted the three public
examples and one v43 anchor into training while keeping the other two anchor
records in dev/test. All saved rows preserved full answers, target-side
supervision for the initial space and true EOS, and masking of tool output.
The four held-out public test records were never admitted to training.

This test verifies the data path and prompt shape. It does not show that the
modified model answers better, that the v43 weights transfer without damage,
or that longer Japanese explanations are learned. The English Numina and
Magicoder source texts now follow v43-style task labels, but the label change
is a hypothesis. The candidate must be rebuilt into a separate pure v44
bundle and evaluated before any training or release decision.

## Files

- `prepare_public_v44.py`: copied original with prompt and supervised-space changes.
- `evaluate_v44_eight_tiers.py`: copied original with only Tier 8 prompt changes.
- `test_v43_prompt_boundary.py`: CPU-only data-shape tests.
- `public_format_samples_v44.json`: byte-identical pinned public examples from
  the original source.
- `v43_boundary_examples.json`, `v43_dolly_anchor_samples.json`: small evidence
  extracts from the saved v43 training ZIP.
- `test_result.json`: test and source hashes.
