"""
GraphRag MCP Server — Knowledge Graph + Vector Search powered by Neo4j and OpenAI embeddings.

Provides tools for ingesting documents, storing facts, and querying knowledge
via hybrid graph traversal + vector similarity search.
"""

import os
import json
import hashlib
import logging
import subprocess
from textwrap import dedent
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag2024")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 256  # cheap + fast, still good quality
CHUNK_SIZE = 1500  # characters per chunk
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
openai_client: Optional[OpenAI] = None
neo4j_driver = None


def get_openai():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client


def ensure_neo4j_running():
    """Start Neo4j if it's not already running."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "neo4j"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip() != "active":
            logger.info("Neo4j not running, starting it...")
            subprocess.run(["sudo", "systemctl", "start", "neo4j"], timeout=30)
            # Wait for it to be ready
            import time
            for _ in range(30):
                try:
                    d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                    d.verify_connectivity()
                    d.close()
                    logger.info("Neo4j is ready.")
                    return
                except Exception:
                    time.sleep(1)
            raise RuntimeError("Neo4j failed to start within 30 seconds")
    except FileNotFoundError:
        pass  # systemctl not available, assume neo4j is managed externally


def get_driver():
    global neo4j_driver
    if neo4j_driver is None:
        ensure_neo4j_running()
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        neo4j_driver.verify_connectivity()
        init_schema(neo4j_driver)
    return neo4j_driver


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------
def init_schema(driver):
    """Create constraints and vector index if they don't exist."""
    with driver.session() as session:
        # Constraints
        session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT document_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.id IS UNIQUE"
        )

        # Vector index on Chunk embeddings
        try:
            session.run(dedent(f"""\
                CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {EMBEDDING_DIMS},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """))
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Vector index creation note: {e}")

        # Vector index on Entity embeddings
        try:
            session.run(dedent(f"""\
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {EMBEDDING_DIMS},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """))
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Entity vector index creation note: {e}")

    logger.info("Schema initialized.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed(text: str) -> list[float]:
    """Get embedding for a text string."""
    resp = get_openai().embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS
    )
    return resp.data[0].embedding


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation near the end
            for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n", "\n\n"]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end + 100)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def make_chunk_id(source: str, index: int) -> str:
    return hashlib.sha256(f"{source}:{index}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "GraphRag",
    instructions="Knowledge Graph + Vector Search for persistent knowledge storage and retrieval. "
    "Use ingest_document to add knowledge, query to search it, store_fact for triples, "
    "get_neighbors to explore the graph.",
)


@mcp.tool()
def ingest_document(text: str, source: str, metadata: str = "{}") -> str:
    """Ingest a document into the knowledge graph.

    Chunks the text, generates embeddings, and stores everything in Neo4j.
    Use this to add knowledge from documents, code files, research papers, notes, etc.

    Args:
        text: The full text content to ingest.
        source: A label for where this came from (e.g. filename, URL, description).
        metadata: Optional JSON string of extra metadata to store on the document node.
    """
    driver = get_driver()
    meta = json.loads(metadata) if metadata else {}
    chunks = chunk_text(text)
    doc_id = hashlib.sha256(f"{source}:{text[:200]}".encode()).hexdigest()[:16]

    with driver.session() as session:
        # Create Document node
        session.run(
            "MERGE (d:Document {id: $id}) "
            "SET d.source = $source, d.metadata = $metadata, d.chunk_count = $count",
            id=doc_id, source=source, metadata=json.dumps(meta), count=len(chunks)
        )

        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(source, i)
            embedding = embed(chunk)

            session.run(
                "MERGE (c:Chunk {id: $id}) "
                "SET c.text = $text, c.source = $source, c.index = $index, "
                "    c.embedding = $embedding "
                "WITH c "
                "MATCH (d:Document {id: $doc_id}) "
                "MERGE (d)-[:HAS_CHUNK]->(c)",
                id=chunk_id, text=chunk, source=source, index=i,
                embedding=embedding, doc_id=doc_id
            )

            # Link sequential chunks
            if i > 0:
                prev_id = make_chunk_id(source, i - 1)
                session.run(
                    "MATCH (prev:Chunk {id: $prev_id}), (curr:Chunk {id: $curr_id}) "
                    "MERGE (prev)-[:NEXT]->(curr)",
                    prev_id=prev_id, curr_id=chunk_id
                )

    return f"Ingested '{source}': {len(chunks)} chunks stored with embeddings."


@mcp.tool()
def store_fact(
    subject: str,
    predicate: str,
    obj: str,
    context: str = "",
    subject_type: str = "Entity",
    obj_type: str = "Entity",
    source: str = "",
) -> str:
    """Store a knowledge triple (subject -[predicate]-> object) in the graph.

    Creates typed nodes and a relationship between them. Nodes get both
    a specific label (e.g. Person, Chemical) AND the base Entity label
    for unified querying. Only embeds new entities to avoid overwriting
    and wasting API calls.

    Args:
        subject: The source entity name (e.g. "Capsaicin", "Jianwei Chen").
        predicate: The relationship type (e.g. "TARGETS", "AUTHORED", "CAUSES").
        obj: The target entity name (e.g. "TRPV1", "Neurogenic Inflammation").
        context: Description of this relationship. Be specific and quantitative.
        subject_type: Entity type for subject (e.g. "Chemical", "Person", "Paper", "Method"). Default "Entity".
        obj_type: Entity type for object (e.g. "Concept", "Condition", "Organism"). Default "Entity".
        source: Optional document source name (e.g. "src-chen-2024-trpv1"). Links entities back to the document for provenance.
    """
    driver = get_driver()
    predicate_clean = predicate.upper().replace(" ", "_").replace("-", "_")

    # Normalize type labels — title case, no spaces
    subject_type = subject_type.strip().title().replace(" ", "")
    obj_type = obj_type.strip().title().replace(" ", "")

    created = []

    with driver.session() as session:
        for entity_name, entity_type in [(subject, subject_type), (obj, obj_type)]:
            # Check if entity already exists
            exists = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e.name AS name",
                name=entity_name
            ).single()

            if exists:
                # Entity exists — add type label if not already present, skip embedding
                if entity_type and entity_type != "Entity":
                    session.run(
                        f"MATCH (e:Entity {{name: $name}}) SET e:{entity_type}",
                        name=entity_name
                    )
            else:
                # New entity — embed and create with labels
                emb = embed(entity_name)
                if entity_type and entity_type != "Entity":
                    session.run(
                        f"MERGE (e:Entity {{name: $name}}) "
                        f"SET e.embedding = $embedding "
                        f"SET e:{entity_type}",
                        name=entity_name, embedding=emb
                    )
                else:
                    session.run(
                        "MERGE (e:Entity {name: $name}) "
                        "SET e.embedding = $embedding",
                        name=entity_name, embedding=emb
                    )
                created.append(entity_name)

        # Create relationship
        session.run(
            f"MATCH (s:Entity {{name: $subject}}), (o:Entity {{name: $obj}}) "
            f"MERGE (s)-[r:{predicate_clean}]->(o) "
            f"SET r.context = $context",
            subject=subject, obj=obj, context=context
        )

        # Link entities to source document for provenance
        if source:
            for entity_name in [subject, obj]:
                session.run(
                    "MATCH (e:Entity {name: $entity}), (d:Document {source: $source}) "
                    "MERGE (e)-[:EXTRACTED_FROM]->(d)",
                    entity=entity_name, source=source
                )

    subj_label = f":{subject_type}" if subject_type != "Entity" else ""
    obj_label = f":{obj_type}" if obj_type != "Entity" else ""
    new_flag = f" (new: {', '.join(created)})" if created else ""
    return f"Stored: ({subject}{subj_label}) -[{predicate_clean}]-> ({obj}{obj_label}){new_flag}" + (f" | {context}" if context else "")


@mcp.tool()
def query(question: str, top_k: int = 5) -> str:
    """Query the knowledge graph using hybrid vector + graph search.

    Embeds the question, finds the most similar chunks and entities via vector search,
    then traverses the graph to gather related context.

    Args:
        question: Natural language question to search for.
        top_k: Number of results to return (default 5).
    """
    driver = get_driver()
    q_embedding = embed(question)
    results = []

    with driver.session() as session:
        # Vector search on chunks
        chunk_results = session.run(
            dedent("""\
                CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
                YIELD node, score
                RETURN node.text AS text, node.source AS source, score
                ORDER BY score DESC
            """),
            k=top_k, embedding=q_embedding
        )
        for record in chunk_results:
            results.append({
                "type": "chunk",
                "text": record["text"],
                "source": record["source"],
                "score": round(record["score"], 4)
            })

        # Vector search on entities
        try:
            entity_results = session.run(
                dedent("""\
                    CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
                    YIELD node, score
                    MATCH (node)-[r]-(related)
                    RETURN node.name AS entity, type(r) AS relationship,
                           related.name AS related_entity, r.context AS context, score
                    ORDER BY score DESC
                    LIMIT $limit
                """),
                k=top_k, embedding=q_embedding, limit=top_k * 3
            )
            for record in entity_results:
                results.append({
                    "type": "relationship",
                    "entity": record["entity"],
                    "relationship": record["relationship"],
                    "related_entity": record["related_entity"],
                    "context": record["context"],
                    "score": round(record["score"], 4)
                })
        except Exception as e:
            logger.debug(f"Entity search note: {e}")

    if not results:
        return "No results found. The knowledge graph may be empty — try ingesting some documents first."

    return json.dumps(results, indent=2)


@mcp.tool()
def get_neighbors(entity: str, depth: int = 2, limit: int = 25) -> str:
    """Explore the graph around a specific entity.

    Returns all nodes and relationships within N hops of the given entity.

    Args:
        entity: The entity name to start from.
        depth: How many hops to traverse (default 2, max 5).
        limit: Maximum number of paths to return (default 25).
    """
    driver = get_driver()
    depth = min(depth, 5)

    with driver.session() as session:
        result = session.run(
            dedent(f"""\
                MATCH path = (e:Entity {{name: $name}})-[*1..{depth}]-(connected)
                RETURN [n IN nodes(path) | n.name] AS node_names,
                       [r IN relationships(path) | type(r)] AS rel_types
                LIMIT $limit
            """),
            name=entity, limit=limit
        )

        paths = []
        for record in result:
            names = record["node_names"]
            rels = record["rel_types"]
            # Build a readable path string
            path_parts = []
            for i, name in enumerate(names):
                path_parts.append(f"({name})")
                if i < len(rels):
                    path_parts.append(f"-[{rels[i]}]->")
            paths.append(" ".join(path_parts))

    if not paths:
        return f"No entity found with name '{entity}'. Check the name or try a query first."

    return f"Graph neighborhood of '{entity}' (depth={depth}):\n" + "\n".join(paths)


@mcp.tool()
def list_entities(pattern: str = "", limit: int = 50) -> str:
    """List entities in the knowledge graph, optionally filtered by name pattern.

    Args:
        pattern: Optional substring to filter entity names (case-insensitive). Leave empty for all.
        limit: Max entities to return (default 50).
    """
    driver = get_driver()

    with driver.session() as session:
        if pattern:
            result = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($pattern) "
                "RETURN e.name AS name ORDER BY name LIMIT $limit",
                pattern=pattern, limit=limit
            )
        else:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name AS name ORDER BY name LIMIT $limit",
                limit=limit
            )

        entities = [record["name"] for record in result]

    if not entities:
        return "No entities found." + (f" (filter: '{pattern}')" if pattern else "")

    return f"Entities ({len(entities)}):\n" + "\n".join(f"  - {e}" for e in entities)


@mcp.tool()
def list_documents(limit: int = 50) -> str:
    """List all ingested documents.

    Args:
        limit: Max documents to return (default 50).
    """
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            "MATCH (d:Document) "
            "RETURN d.source AS source, d.chunk_count AS chunks, d.id AS id "
            "ORDER BY d.source LIMIT $limit",
            limit=limit
        )
        docs = [
            {"source": r["source"], "chunks": r["chunks"], "id": r["id"]}
            for r in result
        ]

    if not docs:
        return "No documents ingested yet."

    lines = [f"Documents ({len(docs)}):"]
    for d in docs:
        lines.append(f"  - {d['source']} ({d['chunks']} chunks)")
    return "\n".join(lines)


@mcp.tool()
def delete_document(source: str) -> str:
    """Delete a document and all its chunks from the knowledge graph.

    Args:
        source: The source label of the document to delete (as shown by list_documents).
    """
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            "MATCH (d:Document {source: $source})-[:HAS_CHUNK]->(c:Chunk) "
            "DETACH DELETE c, d "
            "RETURN count(c) AS deleted",
            source=source
        )
        record = result.single()
        count = record["deleted"] if record else 0

    if count == 0:
        return f"No document found with source '{source}'."
    return f"Deleted document '{source}' and {count} chunks."


@mcp.tool()
def cypher_query(query: str, params: str = "{}") -> str:
    """Execute a raw Cypher query against the Neo4j knowledge graph.

    Use this for ad-hoc exploration, complex queries, schema inspection,
    or anything the other tools don't cover.

    Args:
        query: A Cypher query string (e.g. "MATCH (n) RETURN labels(n), count(n)").
        params: Optional JSON string of query parameters (e.g. '{"name": "Python"}').
    """
    driver = get_driver()
    parsed_params = json.loads(params) if params else {}

    # Block destructive operations for safety
    upper = query.upper().strip()
    if any(kw in upper for kw in ["DROP", "DELETE", "REMOVE", "DETACH"]):
        if not any(safe in upper for safe in ["RETURN"]):
            return "Blocked: destructive queries without RETURN clause are not allowed. Use delete_document for safe deletion."

    with driver.session() as session:
        result = session.run(query, **parsed_params)
        records = []
        for record in result:
            records.append(dict(record))

    if not records:
        return "Query returned no results."

    # Serialize — Neo4j types need special handling
    def serialize(obj):
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
            return list(obj)
        if hasattr(obj, 'items'):
            return dict(obj)
        return str(obj)

    try:
        return json.dumps(records, indent=2, default=serialize)
    except TypeError:
        return "\n".join(str(r) for r in records)


@mcp.tool()
def graph_stats() -> str:
    """Get statistics about the knowledge graph — node counts, relationship counts, etc."""
    driver = get_driver()

    with driver.session() as session:
        stats = {}
        for label in ["Document", "Chunk", "Entity"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
            stats[label] = result.single()["count"]

        rel_result = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC LIMIT 20"
        )
        rels = {r["type"]: r["count"] for r in rel_result}

    lines = [
        "Knowledge Graph Stats:",
        f"  Documents: {stats['Document']}",
        f"  Chunks:    {stats['Chunk']}",
        f"  Entities:  {stats['Entity']}",
        f"  Relationships:"
    ]
    for rel_type, count in rels.items():
        lines.append(f"    {rel_type}: {count}")
    if not rels:
        lines.append("    (none)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts — reusable workflow templates for any MCP client
# ---------------------------------------------------------------------------

@mcp.prompt(
    name="ingest_source",
    title="Ingest a Source into the Knowledge Graph",
    description="Guides you through reading a source, extracting entities and relationships, "
    "and storing everything in the knowledge graph with proper provenance.",
)
def ingest_source_prompt(
    source_text: str,
    source_name: str,
    domains: str = "general, software, research",
) -> list[dict]:
    """Parameterized prompt for ingesting a source document.

    Args:
        source_text: The text content to ingest.
        source_name: A label for the source (filename, URL, title).
        domains: Comma-separated domains to use for extraction (general, software, research).
    """
    domain_list = [d.strip() for d in domains.split(",")]

    # Build entity/relationship type lists from ontology
    from ontology import (
        GENERAL_LABELS, SOFTWARE_LABELS, RESEARCH_LABELS,
        GENERAL_RELATIONSHIPS, SOFTWARE_RELATIONSHIPS, RESEARCH_RELATIONSHIPS,
        DIRECTION_GUIDE,
    )

    labels = dict(GENERAL_LABELS)
    rels = dict(GENERAL_RELATIONSHIPS)
    if "software" in domain_list:
        labels.update(SOFTWARE_LABELS)
        rels.update(SOFTWARE_RELATIONSHIPS)
    if "research" in domain_list:
        labels.update(RESEARCH_LABELS)
        rels.update(RESEARCH_RELATIONSHIPS)

    entity_types = "\n".join(f"  - **{k}**: {v}" for k, v in labels.items())
    rel_types = "\n".join(f"  - **{k}**: {v}" for k, v in rels.items())

    return [
        {
            "role": "user",
            "content": f"""Please ingest the following source into the knowledge graph.

## Source
**Name:** {source_name}

**Content:**
{source_text}

## Instructions

Follow these steps carefully:

### Step 1: Check existing entities
Call `list_entities` to see what's already in the graph. This prevents duplicates and helps you create cross-references to existing knowledge.

### Step 2: Ingest the document
Call `ingest_document` with the full text and source name. This chunks the text and creates embeddings for vector search.

### Step 3: Extract entities
Read through the text and identify important entities. Use these allowed entity types (pick the most specific):

{entity_types}

If an entity doesn't fit any type, use "Entity" as the label.

**Entity naming rules:**
- Normalize names: proper case, no abbreviations unless that IS the name (CGRP, TRPV1 stay uppercase)
- Merge synonyms: "B. bassiana" and "Beauveria bassiana" -> "Beauveria bassiana". Use the full canonical form.
- Be selective — extract what matters for cross-document connections, not every noun
- Prefer specific types: Chemical over Entity for "Substance P", Organism over Entity for "E. coli", Condition over Concept for "Allergic Rhinitis"

### Step 4: Extract relationships
For each pair of related entities, identify the relationship. Use these allowed types:

{rel_types}

If no specific type fits, use "RELATED_TO" with a descriptive context.

{DIRECTION_GUIDE}

### Step 5: Store facts
For each entity-relationship-entity triple, call `store_fact` with:
- subject: the source entity (the Subject in the natural sentence)
- predicate: the relationship type
- obj: the target entity (the Object in the natural sentence)
- context: a specific, quantitative description. "91.67% mortality at day 4" not "high mortality".

If an entity from this source already exists in the graph (from Step 1), reuse its exact name to create cross-document connections.

### Step 6: Report
Summarize what was stored: number of chunks, entities extracted, relationships created, and any cross-references to existing entities.""",
        }
    ]


@mcp.prompt(
    name="query_knowledge",
    title="Query the Knowledge Graph",
    description="Guides you through answering a question using hybrid vector search "
    "and graph traversal across the knowledge graph.",
)
def query_knowledge_prompt(question: str) -> list[dict]:
    """Parameterized prompt for querying the knowledge graph.

    Args:
        question: The question to answer using the knowledge graph.
    """
    return [
        {
            "role": "user",
            "content": f"""Answer this question using the knowledge graph:

**Question:** {question}

## Instructions

Follow these steps to find a comprehensive answer:

### Step 1: Semantic search
Call `query` with the question. This searches both text chunks (vector similarity) and entities (graph + vector). Review the results for relevant information.

### Step 2: Explore the graph
If the query results mention specific entities, call `get_neighbors` on the most relevant ones to discover related information that vector search might have missed. Follow relationship chains for multi-hop reasoning.

### Step 3: Targeted search (if needed)
If the initial results are insufficient:
- Try `list_entities` with relevant name patterns to find entities you might have missed
- Use `cypher_query` for structured queries (e.g., "find all entities of type Technology that have DEPENDS_ON relationships")

### Step 4: Synthesize
Combine findings from vector search and graph traversal into a coherent answer. Cite the sources (document names) where the information came from.

If the knowledge graph doesn't contain enough information to answer the question, say so clearly and suggest what could be ingested to fill the gap.""",
        }
    ]


@mcp.prompt(
    name="review_ontology",
    title="Review and Improve the Knowledge Graph Ontology",
    description="Analyzes the current state of the knowledge graph and suggests "
    "schema improvements — new entity types, relationship cleanup, deduplication.",
)
def review_ontology_prompt() -> list[dict]:
    """Prompt for reviewing and improving the graph ontology."""
    return [
        {
            "role": "user",
            "content": """Review the current state of the knowledge graph and suggest improvements.

## Instructions

### Step 1: Get graph statistics
Call `graph_stats` to see overall counts.

### Step 2: Inspect entity types
Use `cypher_query` to analyze the graph structure:

```cypher
MATCH (e:Entity) RETURN labels(e) AS labels, count(e) AS count ORDER BY count DESC
```

Check for entities still using the catch-all "Entity" label that could be promoted to proper types.

### Step 3: Inspect relationships
```cypher
MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC
```

Look for:
- Overuse of RELATED_TO (should be replaced with specific types)
- Duplicate or near-duplicate relationship types
- Missing relationship types that would improve queries

### Step 4: Check for duplicate entities
```cypher
MATCH (e:Entity) WITH toLower(e.name) AS lname, collect(e.name) AS names, count(*) AS cnt WHERE cnt > 1 RETURN names, cnt ORDER BY cnt DESC
```

### Step 5: Check relationship directions
```cypher
MATCH (s:Entity)-[r]->(t:Entity) RETURN s.name AS from, type(r) AS rel, t.name AS to LIMIT 50
```

Verify relationships read naturally as sentences: "Subject VERB Object".
Flag any backwards relationships (e.g. Paper AUTHORED Person should be Person AUTHORED Paper).

### Step 6: Check for entities that should use specific types
Look for entities currently labeled as generic "Entity" or "Concept" that should be:
- **Chemical** (compounds, molecules, drugs, neuropeptides)
- **Organism** (species, strains, cell lines)
- **Condition** (diseases, disorders, syndromes)
- **Method** (techniques, protocols, assays)

### Step 7: Report
Provide a summary with:
- Current graph size and health
- Entities that should be re-labeled to more specific types
- Relationships with wrong direction that need fixing
- Relationships that need normalization (e.g. CREATED_BY should be AUTHORED)
- Duplicate entities that should be merged
- Suggested new types or relationships based on patterns you see
- Any other schema improvements""",
        }
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
