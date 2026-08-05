"""Text-encoder selection benchmark. Read-only with respect to the generation pipeline.

Nothing under `textbench/` is imported by the trainer, the sampler or the conditioned UNet: this
package exists to *choose* an encoder and a report format, and `textenc/` is what production then
uses. Keeping the two apart is deliberate -- a metric added here can never change what a trained
model does.

    corpus     load the report/label corpus from the shard tars
    analysis   dataset statistics (lengths, sections, negation, acquisition content)
    negation   rule-constructed negation minimal pairs
    embed      cache frozen-encoder embeddings once per (encoder, format, split)
    tasks      the five metrics
    runner     the one scoring path -> metrics_matrix.csv

Like `eval/__init__.py`, this re-exports nothing: a heavy dependency in one module must not make
another unimportable (`corpus` and `analysis` need no torch).
"""
