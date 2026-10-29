# Native Pi execution preflight

`reasonese-native-run` audits the exact full user turns in a compiled native Pi
plan using the existing GPT-5.6 Luna message-QA rubric and YAML cache. It verifies
that the QA inputs match the delivered turns, rejects human TODO placeholders,
and asks the Node runner to verify source artifacts and checkpoint bindings
before constructing a provider transport.

For an explicitly authorized pilot:

```bash
reasonese-native-run --plan out/native-plans/trial.json \
  --out out/native-runs/trial --qa-cache out/native-message-qa.yaml --allow-paid
```

`--allow-paid` is required even with cached QA because the command launches the
assistant continuation. The output directory must be fresh. Exact QA results
reuse the existing cache; changed text, authoring guidance, rubric, or judge
routes require a new verdict. Noncompliant QA stops the continuation.

The command saves `out/native-runs/trial.message-qa.json` before launch. This
receipt includes the SHA256 of the raw plan bytes and ordered content hashes;
the Node runner verifies it again and copies it into the run. Failed QA keeps
the receipt and cache for review. A plan modified while QA is running must be
audited again. These checks establish prompt compliance, not task success or a
completed experiment.

The offline tests inject both the transport and process launcher. They do not
invoke a live judge or run an experimental assistant. Receipts made with either
injected boundary carry `fixture: true`; the production Node runner rejects
them. Injected runs use a separate `*.fixture.yaml` cache so their judgments do
not enter the configured production cache. The CLI provides no fixture override.

QA receipts use exclusive creation. A retry needs a fresh output name even when
the earlier attempt failed before creating its run directory; prior judgments
and failed receipts remain unchanged. Exact content QA caches can still be reused.
