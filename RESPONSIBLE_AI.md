# Responsible AI

## Intended Use

This project is a document-grounded Retrieval-Augmented Generation
knowledge assistant intended to answer questions using an explicitly
indexed document collection.

It is designed as an AI engineering and evaluation project rather than
an authoritative decision-making system.

## Human Oversight

Generated answers should be checked against the cited source documents
before being used for legal, regulatory, financial, employment, safety,
medical or other high-impact decisions.

## Grounding

The system retrieves document evidence before generation and exposes
source, page, chunk and semantic-similarity metadata.

Source attribution is generated deterministically from document metadata
rather than invented by the language model.

## Known Limitations

The language model may produce an incorrect answer even when the correct
evidence has been retrieved.

The project benchmark has demonstrated weaker generation performance for
multi-document synthesis and adversarial questions.

Semantic similarity is not a calibrated probability that an answer is
correct.

## Prompt Injection

Retrieved documents are treated as untrusted content and the generation
prompt explicitly instructs the model not to execute instructions
contained within retrieved text.

This reduces risk but does not guarantee immunity from direct or indirect
prompt injection.

Only trusted source documents should be indexed.

## Data and Privacy

Operational logs intentionally exclude question text, answer text,
retrieved document text and generated prompts.

Secrets are supplied through environment variables and must not be
committed to source control.

## Evaluation

Retrieval and generation are measured separately.

The benchmark includes supported, paraphrased, unsupported,
multi-document and adversarial questions.

Observed failures are retained as regression cases rather than silently
tuned away.