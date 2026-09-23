# regulatory-knowledge-base Specification

## Purpose
Provides a small, reproducible, provenance-tracked corpus of official Brazilian regulatory texts and the fictional product catalog, indexed for hybrid retrieval, so answers can be grounded in and cited back to a specific document and article rather than the LLM's own unverifiable knowledge.

## Requirements

### Requirement: Corpus limited to official public sources with recorded provenance
The system SHALL restrict the regulatory corpus to documents fetched from official public domains, and SHALL record, for every corpus document, its title, norm number, official source URL, retrieval date, version/consolidation date, and a content hash.

#### Scenario: Document provenance is recorded
- **WHEN** a regulatory document is added to the corpus
- **THEN** its manifest entry records title, norm number, official source URL, retrieval date, version/consolidation date, and the sha256 hash of the fetched file

#### Scenario: Non-official source is rejected
- **WHEN** a document's source URL does not belong to one of the configured official domains
- **THEN** the document is rejected from the corpus rather than ingested

### Requirement: Product catalog is a distinct, separately-typed source
The system SHALL index the fictional product catalog's descriptions as retrievable documents, tagged with a source type that distinguishes them from regulatory documents.

#### Scenario: Product catalog document is retrievable and labeled
- **WHEN** a product catalog description is indexed
- **THEN** its chunks are retrievable and carry a source type of `product_catalog`, distinguishing them from regulatory source types

### Requirement: Reproducible corpus fetch and verification
The system SHALL provide a command that re-fetches every manifested document from its recorded official URL and verifies the fetched content against the manifest's recorded hash, without requiring the raw documents to be committed to the repository.

#### Scenario: Fetch verifies hash
- **WHEN** the fetch command re-downloads a manifested document
- **THEN** it compares the downloaded content's hash against the manifest's recorded hash and reports a mismatch rather than silently accepting the content

#### Scenario: Raw documents are not committed
- **WHEN** the corpus is fetched locally
- **THEN** the raw downloaded files land in a gitignored location, and only the manifest is committed

### Requirement: Revoked provisions are excluded from indexed content
The system SHALL NOT include, in any chunk's retrievable content, text that the source document marks as revoked or superseded, and SHALL retain any amendment or revocation annotation only as chunk metadata, never as part of the retrievable content itself.

#### Scenario: Revoked text never appears in a chunk
- **WHEN** a source document marks a provision as revoked or superseded
- **THEN** that provision's text does not appear in any chunk's retrievable content

#### Scenario: Amendment note is metadata, not content
- **WHEN** a source document carries an amendment annotation (e.g. noting that a provision was given new wording or added by a later norm) alongside an in-force provision
- **THEN** the annotation is attached to the chunk as metadata and is not concatenated into the chunk's retrievable content

### Requirement: Extracted text is free of source-rendering artifacts
The system SHALL remove repeated page headers and footers from extracted source text, and SHALL rejoin words split across a line break by hyphenation, so indexed content reads as continuous prose rather than a page-by-page rendering.

#### Scenario: Repeated header or footer is not indexed
- **WHEN** a source document repeats the same header or footer text on multiple pages
- **THEN** that repeated text does not appear in any chunk's retrievable content

#### Scenario: A hyphenated line break is rejoined
- **WHEN** a word is split across a line break by a hyphen in the source document
- **THEN** the extracted text rejoins it into a single word, without the hyphen or an inserted space

### Requirement: Structure-aware chunking of legal texts
The system SHALL split a legal text into chunks along its legal structure (Capítulo, Seção, Artigo, parágrafo, inciso, alínea), SHALL NOT split in the middle of an article when the article fits within the configured chunk size budget, and SHALL split an oversized article by paragraph while repeating the article's header on each resulting chunk.

#### Scenario: An article within budget stays whole
- **WHEN** an article's text fits within the configured chunk size budget
- **THEN** it is indexed as a single chunk, not split across chunk boundaries

#### Scenario: An oversized article splits by paragraph with a repeated header
- **WHEN** an article's text exceeds the configured chunk size budget
- **THEN** it is split at paragraph boundaries, and each resulting chunk repeats the article's header (norm and article reference)

### Requirement: Chunk provenance and hierarchy metadata
The system SHALL attach to every chunk its source document id, norm identifier, article/paragraph reference, hierarchy path, official source URL, and version date.

#### Scenario: Retrieved chunk carries full provenance
- **WHEN** a chunk is retrieved
- **THEN** it carries its document id, norm identifier, article/paragraph reference, hierarchy path, official source URL, and version date

### Requirement: Idempotent, incremental indexing
The system SHALL upsert chunks by a deterministic chunk id derived from their content and position, such that re-running ingestion over an unchanged document produces no duplicate or additional chunks, and SHALL reindex only the documents whose content hash has changed since the last ingestion.

#### Scenario: Re-ingesting an unchanged document is a no-op
- **WHEN** ingestion runs again over a document whose content hash has not changed
- **THEN** no new or duplicate chunks are created for that document

#### Scenario: A changed document triggers reindexing of only that document
- **WHEN** ingestion runs and one document's content hash differs from its last-indexed hash
- **THEN** only that document's chunks are recomputed and upserted; other documents' chunks are left untouched

### Requirement: Hybrid retrieval with reciprocal rank fusion
The system SHALL retrieve candidate chunks using both vector similarity search and Portuguese full-text search that matches regardless of diacritics (accents), and SHALL fuse the two rankings via Reciprocal Rank Fusion into a single ranked result, with the number of results and the fusion weighting configurable.

#### Scenario: Fused ranking reflects both retrieval methods
- **WHEN** a query is retrieved
- **THEN** the returned ranking is the Reciprocal Rank Fusion of the vector-similarity ranking and the full-text-search ranking, not either one alone

#### Scenario: Top-k and fusion weights are configurable
- **WHEN** the configured top-k or fusion weighting is changed
- **THEN** subsequent retrievals use the newly configured values without code changes

#### Scenario: An unaccented query still matches accented content
- **WHEN** a query omits diacritics that the matching indexed content actually has (e.g. querying without accents for a word the corpus stores with accents)
- **THEN** the full-text search still matches that content

### Requirement: Retrieval is scoped by source type per intent
The system SHALL support restricting retrieval to a single source type, and SHALL restrict a product question's retrieval to `product_catalog` documents and a regulatory question's retrieval to `regulation` documents.

#### Scenario: Product question retrieval never surfaces regulatory content
- **WHEN** retrieval runs for a product question
- **THEN** only `product_catalog` chunks are eligible to be returned, never `regulation` chunks

#### Scenario: Regulatory question retrieval never surfaces product-catalog content
- **WHEN** retrieval runs for a regulatory question
- **THEN** only `regulation` chunks are eligible to be returned, never `product_catalog` chunks

### Requirement: Optional reranking stage
The system SHALL support an optional reranking stage applied after fusion, disabled by default, that can be enabled via configuration without changing retrieval call sites.

#### Scenario: Reranking disabled by default
- **WHEN** reranking is not explicitly enabled in configuration
- **THEN** retrieval results are returned in fused-ranking order without a reranking step applied

#### Scenario: Reranking enabled via configuration only
- **WHEN** reranking is enabled in configuration
- **THEN** retrieval results are reordered by the reranking stage before being returned, with no change to the calling code

### Requirement: Retrieval quality evaluation
The system SHALL provide a committed set of Portuguese evaluation questions — most with an expected source document and article, and a smaller subset that are deliberately unanswerable from the corpus and expect a refusal — with every question paraphrased rather than copied verbatim from source article text, and a command that reports recall@k and Mean Reciprocal Rank (MRR) over the answerable questions plus refusal accuracy over the unanswerable ones, against the current index.

#### Scenario: Evaluation reports recall@k and MRR
- **WHEN** the retrieval evaluation command runs against the committed question set's answerable questions
- **THEN** it reports recall@k and MRR computed from whether and where each question's expected document/article appears in the retrieved results

#### Scenario: Evaluation reports refusal accuracy on unanswerable questions
- **WHEN** the retrieval evaluation command runs against the committed question set's deliberately unanswerable questions
- **THEN** it reports refusal accuracy: the fraction of those questions for which the system correctly declines to answer

#### Scenario: Questions are paraphrased, not copied from source text
- **WHEN** an evaluation question is added to the committed set
- **THEN** its wording is a paraphrase of how a customer would actually ask, not a verbatim copy of the cited article's own sentence

### Requirement: Automated tests never require the corpus or embedding model to be fetched
The system SHALL run its automated unit and integration tests using small committed fixtures and a fake embeddings adapter, and SHALL NOT download the corpus or an embedding model as part of CI.

#### Scenario: CI runs without network access to the corpus or model
- **WHEN** the automated test suite runs in CI
- **THEN** it completes using committed fixtures and the fake embeddings adapter, without fetching the corpus or downloading an embedding model
