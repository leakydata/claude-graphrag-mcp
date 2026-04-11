# GraphRag — Knowledge Graph + Vector Search MCP Server

A local MCP (Model Context Protocol) server that gives Claude Code persistent, structured knowledge via a Neo4j knowledge graph with OpenAI vector embeddings. Combines the precision of graph traversal with the fuzziness of semantic vector search.

## Table of Contents

- [Architecture](#architecture)
- [Theory & Background](#theory--background)
- [Ontology Design](#ontology-design)
- [MCP Tools Reference](#mcp-tools-reference)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage Patterns](#usage-patterns)
- [Graph Schema](#graph-schema)
- [Ontology Evolution](#ontology-evolution)
- [Methods & Algorithms](#methods--algorithms)
- [Resources & References](#resources--references)

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  Claude Code     │────▶│  MCP Server (Python)  │────▶│  Neo4j       │
│  (any project)   │◀────│  stdio transport       │     │  graph + vec │
└─────────────────┘     └──────────┬───────────┘     └──────────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │ OpenAI API    │
                            │ embeddings    │
                            └──────────────┘
```

- **Claude Code** connects to the MCP server via stdio (configured globally in `~/.claude/settings.json`)
- **MCP Server** (`server.py`) handles all logic: chunking, embedding, storage, retrieval
- **Neo4j** stores the knowledge graph (nodes, relationships) and vector indexes
- **OpenAI** provides `text-embedding-3-large` embeddings (256 dims, ~$0.13/1M tokens)

## Theory & Background

### Why Knowledge Graphs + RAG?

Standard RAG (Retrieval Augmented Generation) uses flat vector search: embed a question, find similar text chunks. This works for simple lookups but fails when:

- **Answers require connecting multiple pieces of information** — "What services depend on the auth module that had the compliance issue?" requires traversing relationships, not just finding similar text.
- **The same entity appears in many contexts** — vector search returns scattered chunks; a graph connects them into a coherent picture.
- **You need structured reasoning** — "List all Python libraries used by the data pipeline team" requires graph queries, not fuzzy similarity.

### KAG: Knowledge Augmented Generation

KAG (Luo et al., 2024) goes beyond RAG by maintaining both structured knowledge (triples) and unstructured text (chunks), cross-linked. The key insight from KAG:

- **"Friendly" knowledge**: Schema-constrained, high-precision facts stored as graph triples (entities + relationships)
- **"Unfriendly" knowledge**: Schema-free text chunks with embeddings for fuzzy retrieval
- **Mutual indexing**: Cross-links between triples and source chunks so you can trace any fact back to its origin

This project implements this dual approach:
- `Chunk` nodes hold embedded text for vector search
- `Entity` nodes hold structured facts for graph traversal
- `EXTRACTED_FROM` relationships link entities back to source chunks
- `HAS_CHUNK` relationships link documents to their chunks

### Microsoft GraphRAG Insights

Microsoft's GraphRAG adds a hierarchical community layer:
- Entities are clustered into `Community` nodes using the Leiden algorithm
- Community summaries enable **global queries** ("What are the main themes across all documents?")
- This is especially powerful for large corpora where individual entity queries miss the forest for the trees

### Neo4j LLM Graph Builder Insights

Neo4j's approach emphasizes:
- **Schema-guided extraction dramatically improves precision** over schema-free extraction
- **Document→Chunk→Entity provenance chain** must always be maintained for traceability
- Vector indexes on both chunks and entities enable hybrid retrieval

## Ontology Design

### The Core Principle

> **Start constrained, expand deliberately.** An unconstrained graph becomes spaghetti. Define allowed node types and relationship types upfront, use a catch-all for outliers, and promote recurring catch-all patterns into the schema.

### Node Labels

#### Structural (RAG plumbing — always present)

| Label | Description |
|-------|-------------|
| `Document` | Source document that was ingested |
| `Chunk` | Text chunk with embedding |
| `Community` | Cluster of related entities |

#### General-Purpose

| Label | Description |
|-------|-------------|
| `Entity` | **Catch-all** for unclassified entities (review periodically) |
| `Person` | People, authors, contributors |
| `Organization` | Companies, institutions, teams |
| `Location` | Geographic places |
| `Event` | Named events, milestones, releases |
| `Concept` | Abstract ideas, topics, paradigms |
| `Tool` | Software tools, utilities, applications |

#### Software/Code Domain

| Label | Description |
|-------|-------------|
| `Repository` | Code repositories |
| `Module` | Packages, libraries, logical code units |
| `Class` | OOP classes, structs, types |
| `Function` | Functions, methods, procedures |
| `API` | Endpoints, interfaces, protocols |
| `Technology` | Languages, frameworks, platforms |
| `Error` | Bugs, exceptions, issues |
| `Configuration` | Config files, env variables |

#### Research/Academic Domain

| Label | Description |
|-------|-------------|
| `Paper` | Academic publications |
| `Dataset` | Named datasets |
| `Method` | Algorithms, techniques, architectures |
| `Metric` | Evaluation measures, benchmarks |
| `Finding` | Key results, claims, conclusions |
| `Theory` | Theoretical frameworks |

### Relationship Types

#### Structural

| Type | Description |
|------|-------------|
| `HAS_CHUNK` | Document → Chunk |
| `NEXT` | Chunk → Chunk (ordering) |
| `EXTRACTED_FROM` | Entity → Chunk (provenance) |
| `IN_COMMUNITY` | Entity → Community |

#### General-Purpose (~15 core types — keep this tight)

| Type | Description |
|------|-------------|
| `RELATED_TO` | Generic (use when nothing specific fits) |
| `PART_OF` | Composition/membership |
| `HAS_PART` | Inverse of PART_OF |
| `LOCATED_IN` | Spatial containment |
| `MEMBER_OF` | Belongs to organization/group |
| `CREATED_BY` | Authorship |
| `OCCURRED_AT` | Temporal placement |
| `CAUSES` | Causal |
| `DEPENDS_ON` | Dependency |
| `SIMILAR_TO` | Similarity/equivalence |
| `DERIVED_FROM` | Origin, fork |
| `MENTIONS` | Reference |
| `DESCRIBES` | Documentation |

#### Software/Code

`IMPORTS`, `CALLS`, `EXTENDS`, `IMPLEMENTS`, `CONFIGURES`, `DEPLOYS_TO`, `THROWS`, `TESTS`, `DOCUMENTS`, `USES`

#### Research/Academic

`CITES`, `EVALUATES_ON`, `PROPOSES`, `OUTPERFORMS`, `USES_METHOD`, `TRAINED_ON`, `REPLICATES`

### When to Add New Types

1. **Check catch-all first**: If you see >5 `Entity` nodes with the same `subtype` property, promote it to a full label.
2. **Check relationships**: If you're using `RELATED_TO` repeatedly for the same pattern, create a specific relationship type.
3. **Never delete types**: Deprecate by merging into parent types.
4. **Document new types** in `ontology.py` with a description.

## MCP Tools Reference

### `ingest_document(text, source, metadata?)`
Ingest text into the knowledge graph. Chunks the text at sentence boundaries, generates embeddings, stores as `Document` → `Chunk` chain.

- **text**: Full text content
- **source**: Label for origin (filename, URL, description)
- **metadata**: Optional JSON string of extra properties

### `store_fact(subject, predicate, obj, context?)`
Store a knowledge triple as graph nodes and relationship.

- **subject**: Source entity name
- **predicate**: Relationship type (see ontology)
- **obj**: Target entity name
- **context**: Optional description

### `query(question, top_k?)`
Hybrid search: embeds the question, runs vector similarity on chunks AND entities, then traverses graph relationships from matching entities.

- **question**: Natural language query
- **top_k**: Results to return (default 5)

### `get_neighbors(entity, depth?, limit?)`
Graph exploration starting from a named entity.

- **entity**: Starting entity name
- **depth**: Hops to traverse (default 2, max 5)
- **limit**: Max paths (default 25)

### `cypher_query(query, params?)`
Execute raw Cypher queries. Safety guard blocks destructive operations without RETURN clauses.

- **query**: Cypher query string
- **params**: Optional JSON parameters

### `list_entities(pattern?, limit?)`
List entities, optionally filtered by name substring.

### `list_documents(limit?)`
List all ingested documents with chunk counts.

### `delete_document(source)`
Remove a document and all its chunks.

### `graph_stats()`
Node counts, relationship type counts.

## Setup & Installation

### Prerequisites
- Neo4j Community Edition (installed via apt)
- Python 3.11+
- uv package manager
- OpenAI API key

### Install

```bash
# Neo4j (already installed as systemd service)
sudo systemctl start neo4j
sudo systemctl status neo4j

# Python dependencies
cd /home/scholyx/Documents/code/GraphRag
uv sync

# Set OpenAI key (add to ~/.bashrc for persistence)
export OPENAI_API_KEY="sk-your-key-here"
```

### Neo4j Access
- **Bolt**: `bolt://localhost:7687`
- **Browser**: `http://localhost:7474`
- **Credentials**: `neo4j` / `graphrag2024`

## Configuration

### MCP Registration (global — all Claude Code instances)

In `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "graphrag": {
      "command": "/home/scholyx/Documents/code/GraphRag/.venv/bin/python",
      "args": ["/home/scholyx/Documents/code/GraphRag/server.py"],
      "cwd": "/home/scholyx/Documents/code/GraphRag"
    }
  }
}
```

### Environment Variables

In `.env` (loaded by python-dotenv):
```
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag2024
```

### Embedding Config

In `server.py`:
- **Model**: `text-embedding-3-large`
- **Dimensions**: 256 (reduced from 3072 — saves cost/storage, minimal quality loss)
- **Chunk size**: 1500 characters with 200 character overlap
- **Similarity**: Cosine

## Usage Patterns

### Ingest a document
```
Use ingest_document to store this research paper...
```

### Store structured facts
```
Store that Neo4j depends on the JVM, and that GraphRAG was created by Microsoft Research.
```

### Query knowledge
```
What do we know about knowledge graph ontology design?
```

### Explore the graph
```
Show me everything connected to "Neo4j" within 2 hops.
```

### Run Cypher directly
```
Run a Cypher query to find all entities that have more than 3 relationships.
```

### Inspect the graph
```
Show me graph stats. List all entities matching "Python".
```

## Graph Schema

### Neo4j Indexes

| Index | Type | Target |
|-------|------|--------|
| `chunk_embedding` | VECTOR (256d, cosine) | `Chunk.embedding` |
| `entity_embedding` | VECTOR (256d, cosine) | `Entity.embedding` |
| `chunk_id` | RANGE (unique) | `Chunk.id` |
| `entity_name` | RANGE (unique) | `Entity.name` |
| `document_id` | RANGE (unique) | `Document.id` |

### Node Properties

**Document**: `id`, `source`, `metadata` (JSON), `chunk_count`
**Chunk**: `id`, `text`, `source`, `index`, `embedding` (vector)
**Entity**: `name`, `embedding` (vector), optional `subtype`

## Ontology Evolution

### Design Philosophy: Use-Case-Driven Modeling

From the Neo4j Data Modeling MCP Server team (Alex Gilmore & Jesus Barrasa):

> Don't model the world upfront. Start with specific questions you want the graph to answer, model for those, then extend iteratively. Each iteration should take minutes, not hours.

**The workflow:**
1. **Define use cases first** — what questions will this graph answer?
2. **Provide sample data** — let the schema emerge from real data + use cases
3. **Generate first draft** — create a minimal model that covers the use cases
4. **Validate** — check for duplicates, missing nodes, broken references
5. **Visualize & review** — look at the model, identify gaps
6. **Iterate** — add new data sources, extend the model, repeat

**Anti-patterns to avoid:**
- Modeling entities/relationships that no use case requires ("it might be useful later")
- Making something a separate node when a property would suffice (e.g., TubeLine as a node vs. a property on Station — only separate it if you need to query/traverse through it)
- Over-engineering before you have data flowing

### Process for adding new domains

1. **Identify recurring patterns** in catch-all `Entity` nodes (run `cypher_query` to check)
2. **Define new labels** with descriptions in `ontology.py`
3. **Define new relationship types** — keep them specific and directional
4. **Update extraction prompts** if using LLM-based extraction
5. **Backfill existing data** by re-labeling matching `Entity` nodes

### Normalization

Run periodically to keep the graph clean:
- **Entity deduplication**: Find entities with similar names (embedding similarity > 0.95) and merge
- **Relationship normalization**: Map ad-hoc relationship types to the blessed set
- **Catch-all review**: Promote recurring `Entity` subtypes to proper labels

### OWL/Turtle Formalization

The ontology can be exported to OWL (W3C standard) for portability and versioning:
- **Node types** → OWL Classes
- **Relationships** → OWL Object Properties
- **Node properties** → OWL Datatype Properties

Standard serialization formats: Turtle (human-readable), RDF/XML, JSON-LD. Tools like WebProtege can import/visualize OWL ontologies. The Neo4j Data Modeling MCP Server supports OWL/Turtle import and export directly.

## Methods & Algorithms

### Chunking Strategy

**Sentence-boundary chunking with overlap**:
1. Target chunk size: 1500 characters
2. Overlap: 200 characters (prevents losing context at boundaries)
3. Break points: sentence-ending punctuation (`. `, `? `, `! `, `\n\n`)
4. Fallback: hard break at chunk_size if no sentence boundary found

Why not token-based? Character-based is simpler, faster, and good enough for this use case. Token-based chunking matters when you're optimizing for a specific model's context window, which isn't the bottleneck here.

### Embedding

**OpenAI `text-embedding-3-large` at 256 dimensions**:
- Full model produces 3072-dim vectors
- Matryoshka representation learning means you can truncate to fewer dimensions with minimal quality loss
- 256 dims ≈ 93% of full quality at ~8% of the storage/compute cost
- Cosine similarity for search

### Hybrid Retrieval

Query flow:
1. Embed the question
2. **Vector search on chunks**: Find top-k most similar text chunks
3. **Vector search on entities**: Find top-k most similar entities
4. **Graph traversal from matched entities**: Follow relationships to gather connected context
5. Return combined results ranked by similarity score

This gives you both:
- **Fuzzy recall** (vector) — finds relevant content even with different wording
- **Structured precision** (graph) — follows explicit relationships for connected facts

### Provenance Chain

Every piece of knowledge traces back to its source:
```
Document → HAS_CHUNK → Chunk → EXTRACTED_FROM ← Entity
```

This means you can always answer "where did this fact come from?" by traversing back to the source document.

## Resources & References

### Primary Inspirations
- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag — community-based summarization for global queries
- **Neo4j LLM Graph Builder**: https://github.com/neo4j-labs/llm-graph-builder — schema-guided extraction + hybrid search
- **KAG** (Luo et al., 2024): https://arxiv.org/abs/2409.13731 — dual friendly/unfriendly knowledge with mutual indexing

### Supporting Tools
- **Neo4j Data Modeling MCP Server**: https://github.com/neo4j-labs/neo4j-data-modeling-mcp-server — schema validation, mermaid visualization, OWL export, iterative model refinement. Presented by Alex Gilmore & Jesus Barrasa (Neo4j). Video: https://www.youtube.com/watch?v=rVsiZ8UXULg
- **Neo4j GraphRAG Python**: https://github.com/neo4j/neo4j-graphrag-python
- **OpenSPG** (KAG's framework): https://github.com/OpenSPG/openspg
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

### Key Papers
- "Graph RAG: Unlocking LLM discovery on narrative private data" (Microsoft Research)
- "KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation" (Luo et al., 2024)

---

## File Structure

```
GraphRag/
├── server.py          # MCP server — all tools and Neo4j interaction
├── ontology.py        # Node labels, relationship types, extraction prompts
├── pyproject.toml     # uv/pip project config
├── .env               # API keys and Neo4j credentials (not in git)
├── .env.example       # Template for .env
├── .gitignore
└── README.md          # This file
```
