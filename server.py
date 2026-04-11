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
    """Get embedding for a single text string."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or export it.")
    resp = get_openai().embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS
    )
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in a single API call."""
    if not texts:
        return []
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or export it.")
    resp = get_openai().embeddings.create(
        input=texts, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS
    )
    # Sort by index to maintain order
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    # Clamp overlap to prevent stalling
    overlap = min(overlap, chunk_size - 1)

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n", "\n\n"]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end + 100)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Guarantee forward progress
        new_start = end - overlap
        if new_start <= start:
            new_start = start + 1
        start = new_start
    return chunks


def make_chunk_id(source: str, index: int) -> str:
    return hashlib.sha256(f"{source}:{index}".encode()).hexdigest()[:16]


import re

# Strict pattern for Cypher identifiers — only alphanumeric + underscore
_CYPHER_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sanitize_cypher_label(label: str) -> str:
    """Validate a string is safe to use as a Cypher label/relationship type.
    Raises ValueError if it contains anything other than alphanumeric + underscore."""
    if not _CYPHER_IDENT_RE.match(label):
        raise ValueError(f"Invalid Cypher identifier: '{label}'. Only alphanumeric and underscores allowed.")
    return label


def parse_json_arg(value: str, arg_name: str) -> dict | list:
    """Parse a JSON string with a clear error message on failure."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid JSON in '{arg_name}': {e}. Received: {value[:200]}")


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "GraphRag",
    instructions="Knowledge Graph + Vector Search for persistent knowledge storage and retrieval. "
    "Use ingest_document to add knowledge, query to search it, store_fact/store_facts for triples, "
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
    meta = parse_json_arg(metadata, "metadata") if metadata else {}
    chunks = chunk_text(text)
    doc_id = hashlib.sha256(f"{source}:{text[:200]}".encode()).hexdigest()[:16]

    # Batch embed all chunks in one API call
    embeddings = embed_batch(chunks)

    with driver.session() as session:
        # Create Document node
        session.run(
            "MERGE (d:Document {id: $id}) "
            "SET d.source = $source, d.metadata = $metadata, d.chunk_count = $count",
            id=doc_id, source=source, metadata=json.dumps(meta), count=len(chunks)
        )

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = make_chunk_id(source, i)

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
    subject_type = subject_type.strip().title().replace(" ", "")
    obj_type = obj_type.strip().title().replace(" ", "")

    # Validate all identifiers before building Cypher
    try:
        sanitize_cypher_label(predicate_clean)
        if subject_type != "Entity":
            sanitize_cypher_label(subject_type)
        if obj_type != "Entity":
            sanitize_cypher_label(obj_type)
    except ValueError as e:
        return f"Rejected: {e}"

    created = []

    with driver.session() as session:
        for entity_name, entity_type in [(subject, subject_type), (obj, obj_type)]:
            exists = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e.name AS name",
                name=entity_name
            ).single()

            if exists:
                if entity_type and entity_type != "Entity":
                    session.run(
                        f"MATCH (e:Entity {{name: $name}}) SET e:{entity_type}",
                        name=entity_name
                    )
            else:
                emb = embed(f"{entity_name} ({entity_type})")
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

        # Link entities to source document AND its chunks for provenance
        if source:
            for entity_name in [subject, obj]:
                # Link to document
                session.run(
                    "MATCH (e:Entity {name: $entity}), (d:Document {source: $source}) "
                    "MERGE (e)-[:EXTRACTED_FROM]->(d)",
                    entity=entity_name, source=source
                )
                # Link to chunks from that document for chunk-level traceability
                session.run(
                    "MATCH (e:Entity {name: $entity}), (c:Chunk {source: $source}) "
                    "MERGE (e)-[:MENTIONED_IN]->(c)",
                    entity=entity_name, source=source
                )

    subj_label = f":{subject_type}" if subject_type != "Entity" else ""
    obj_label = f":{obj_type}" if obj_type != "Entity" else ""
    new_flag = f" (new: {', '.join(created)})" if created else ""
    return f"Stored: ({subject}{subj_label}) -[{predicate_clean}]-> ({obj}{obj_label}){new_flag}" + (f" | {context}" if context else "")


@mcp.tool()
def store_facts(facts_json: str) -> str:
    """Store multiple knowledge triples in a single batch operation.

    Much more efficient than calling store_fact repeatedly — uses batch embedding
    and a single database transaction. Use this when ingesting multiple facts
    from a single source.

    Args:
        facts_json: JSON array of fact objects. Each object should have:
            - subject (str): Source entity name
            - predicate (str): Relationship type
            - obj (str): Target entity name
            - context (str): Description of the relationship
            - subject_type (str, optional): Entity type for subject. Default "Entity"
            - obj_type (str, optional): Entity type for object. Default "Entity"
            - source (str, optional): Document source name for provenance

            Example: [{"subject": "Capsaicin", "predicate": "TARGETS", "obj": "TRPV1",
                       "context": "TRPV1 agonist", "subject_type": "Chemical",
                       "obj_type": "Concept", "source": "src-chen-2024-trpv1"}]
    """
    driver = get_driver()
    facts = parse_json_arg(facts_json, "facts_json")

    if not isinstance(facts, list) or not facts:
        return "No facts provided. Expected a JSON array of fact objects."

    # Validate all identifiers upfront before touching the database
    entity_types_map = {}
    for i, fact in enumerate(facts):
        for key in ("subject", "predicate", "obj"):
            if key not in fact:
                return f"Fact #{i} missing required field '{key}'."

        predicate_clean = fact["predicate"].upper().replace(" ", "_").replace("-", "_")
        try:
            sanitize_cypher_label(predicate_clean)
        except ValueError as e:
            return f"Fact #{i}: {e}"

        st = fact.get("subject_type", "Entity").strip().title().replace(" ", "")
        ot = fact.get("obj_type", "Entity").strip().title().replace(" ", "")

        for label in (st, ot):
            if label != "Entity":
                try:
                    sanitize_cypher_label(label)
                except ValueError as e:
                    return f"Fact #{i}: {e}"

        if fact["subject"] not in entity_types_map:
            entity_types_map[fact["subject"]] = st
        if fact["obj"] not in entity_types_map:
            entity_types_map[fact["obj"]] = ot

    # All validated — proceed with database operations
    with driver.session() as session:
        # Find which entities already exist
        all_entity_names = set()
        for fact in facts:
            all_entity_names.add(fact["subject"])
            all_entity_names.add(fact["obj"])

        existing = set()
        for name in all_entity_names:
            result = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e.name AS name",
                name=name
            ).single()
            if result:
                existing.add(name)

        new_entities = all_entity_names - existing

        # Batch embed all new entities in one API call
        new_entities_list = sorted(new_entities)
        if new_entities_list:
            embed_texts = [f"{name} ({entity_types_map.get(name, 'Entity')})" for name in new_entities_list]
            embeddings = embed_batch(embed_texts)

            for name, emb in zip(new_entities_list, embeddings):
                entity_type = entity_types_map.get(name, "Entity")
                if entity_type and entity_type != "Entity":
                    session.run(
                        f"MERGE (e:Entity {{name: $name}}) "
                        f"SET e.embedding = $embedding "
                        f"SET e:{entity_type}",
                        name=name, embedding=emb
                    )
                else:
                    session.run(
                        "MERGE (e:Entity {name: $name}) "
                        "SET e.embedding = $embedding",
                        name=name, embedding=emb
                    )

        # Update type labels for existing entities
        for name in existing:
            entity_type = entity_types_map.get(name, "Entity")
            if entity_type and entity_type != "Entity":
                session.run(
                    f"MATCH (e:Entity {{name: $name}}) SET e:{entity_type}",
                    name=name
                )

        # Create all relationships
        rel_count = 0
        for fact in facts:
            predicate_clean = fact["predicate"].upper().replace(" ", "_").replace("-", "_")
            context = fact.get("context", "")
            session.run(
                f"MATCH (s:Entity {{name: $subject}}), (o:Entity {{name: $obj}}) "
                f"MERGE (s)-[r:{predicate_clean}]->(o) "
                f"SET r.context = $context",
                subject=fact["subject"], obj=fact["obj"], context=context
            )
            rel_count += 1

            # Provenance links — document + chunk level
            source = fact.get("source", "")
            if source:
                for entity_name in [fact["subject"], fact["obj"]]:
                    session.run(
                        "MATCH (e:Entity {name: $entity}), (d:Document {source: $source}) "
                        "MERGE (e)-[:EXTRACTED_FROM]->(d)",
                        entity=entity_name, source=source
                    )
                    session.run(
                        "MATCH (e:Entity {name: $entity}), (c:Chunk {source: $source}) "
                        "MERGE (e)-[:MENTIONED_IN]->(c)",
                        entity=entity_name, source=source
                    )

    return (
        f"Batch stored: {rel_count} relationships, "
        f"{len(new_entities)} new entities (embedded), "
        f"{len(existing)} existing entities (reused). "
        f"Total entities: {len(all_entity_names)}."
    )


@mcp.tool()
def query(question: str, top_k: int = 5) -> str:
    """Query the knowledge graph using hybrid vector + graph search.

    Embeds the question, finds the most similar chunks and entities via vector search,
    then automatically traverses 1 hop from matched entities to include connected
    facts and relationships in the results.

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

        # Vector search on entities + auto-traverse 1 hop
        try:
            # First, find matching entities
            entity_matches = session.run(
                dedent("""\
                    CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
                    YIELD node, score
                    RETURN node.name AS entity, labels(node) AS labels, score
                    ORDER BY score DESC
                """),
                k=top_k, embedding=q_embedding
            )

            matched_entities = []
            for record in entity_matches:
                labels = [l for l in record["labels"] if l != "Entity"]
                matched_entities.append({
                    "name": record["entity"],
                    "type": labels[0] if labels else "Entity",
                    "score": round(record["score"], 4)
                })

            # For each matched entity, traverse 1 hop to get connected facts
            for match in matched_entities:
                # Outgoing relationships
                outgoing = session.run(
                    dedent("""\
                        MATCH (e:Entity {name: $name})-[r]->(target:Entity)
                        RETURN e.name AS from, type(r) AS rel, target.name AS to,
                               labels(target) AS target_labels, r.context AS context
                    """),
                    name=match["name"]
                )
                for record in outgoing:
                    target_labels = [l for l in record["target_labels"] if l != "Entity"]
                    results.append({
                        "type": "graph_fact",
                        "from": record["from"],
                        "relationship": record["rel"],
                        "to": record["to"],
                        "to_type": target_labels[0] if target_labels else "Entity",
                        "context": record["context"],
                        "score": match["score"],
                        "match": match["name"],
                    })

                # Incoming relationships
                incoming = session.run(
                    dedent("""\
                        MATCH (source:Entity)-[r]->(e:Entity {name: $name})
                        RETURN source.name AS from, type(r) AS rel, e.name AS to,
                               labels(source) AS source_labels, r.context AS context
                    """),
                    name=match["name"]
                )
                for record in incoming:
                    source_labels = [l for l in record["source_labels"] if l != "Entity"]
                    results.append({
                        "type": "graph_fact",
                        "from": record["from"],
                        "from_type": source_labels[0] if source_labels else "Entity",
                        "relationship": record["rel"],
                        "to": record["to"],
                        "context": record["context"],
                        "score": match["score"],
                        "match": match["name"],
                    })

        except Exception as e:
            logger.debug(f"Entity search note: {e}")

    if not results:
        return "No results found. The knowledge graph may be empty — try ingesting some documents first."

    # Deduplicate graph facts (same fact can appear from multiple matched entities)
    seen = set()
    deduped = []
    for r in results:
        if r["type"] == "graph_fact":
            key = (r["from"], r["relationship"], r["to"])
            if key in seen:
                continue
            seen.add(key)
        deduped.append(r)

    return json.dumps(deduped, indent=2)


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
                "RETURN e.name AS name, labels(e) AS labels ORDER BY name LIMIT $limit",
                pattern=pattern, limit=limit
            )
        else:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, labels(e) AS labels ORDER BY name LIMIT $limit",
                limit=limit
            )

        entities = []
        for record in result:
            labels = [l for l in record["labels"] if l != "Entity"]
            type_str = f" [{labels[0]}]" if labels else ""
            entities.append(f"{record['name']}{type_str}")

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
def cypher_query(query: str, params: str = "{}", allow_write: bool = False) -> str:
    """Execute a raw Cypher query against the Neo4j knowledge graph.

    Read-only by default. Use this for ad-hoc exploration, complex queries,
    schema inspection, or anything the other tools don't cover.

    Args:
        query: A Cypher query string (e.g. "MATCH (n) RETURN labels(n), count(n)").
        params: Optional JSON string of query parameters (e.g. '{"name": "Python"}').
        allow_write: Set to true to allow destructive operations (DELETE, DROP, REMOVE). Default false.
    """
    driver = get_driver()
    parsed_params = parse_json_arg(params, "params") if params else {}

    # Block destructive operations unless explicitly allowed
    upper = query.upper().strip()
    write_keywords = ["DROP", "DELETE", "REMOVE", "DETACH", "CREATE", "MERGE", "SET"]
    if any(kw in upper for kw in write_keywords) and not allow_write:
        return (
            "Blocked: this query contains write operations. "
            "Set allow_write=true to permit, or use the dedicated tools "
            "(store_fact, store_facts, delete_document) for safe mutations."
        )

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

        # Count typed entities
        typed_result = session.run(
            "MATCH (e:Entity) WITH labels(e) AS labs "
            "UNWIND labs AS lab WITH lab WHERE lab <> 'Entity' "
            "RETURN lab AS type, count(*) AS count ORDER BY count DESC"
        )
        typed = {r["type"]: r["count"] for r in typed_result}

        rel_result = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC LIMIT 20"
        )
        rels = {r["type"]: r["count"] for r in rel_result}

    lines = [
        "Knowledge Graph Stats:",
        f"  Documents: {stats['Document']}",
        f"  Chunks:    {stats['Chunk']}",
        f"  Entities:  {stats['Entity']}",
    ]
    if typed:
        lines.append("  Entity types:")
        for t, c in typed.items():
            lines.append(f"    {t}: {c}")
    lines.append("  Relationships:")
    for rel_type, count in rels.items():
        lines.append(f"    {rel_type}: {count}")
    if not rels:
        lines.append("    (none)")

    return "\n".join(lines)


@mcp.tool()
def merge_entities(keep: str, merge: str) -> str:
    """Merge two duplicate entities into one.

    Keeps the 'keep' entity and transfers all relationships from 'merge' to it,
    then deletes the 'merge' entity. Use this for deduplication — when the same
    real-world entity has multiple nodes (e.g. "TRPV1 receptor" and "TRPV1").

    Args:
        keep: The entity name to keep (the canonical name).
        merge: The entity name to merge into 'keep' (will be deleted).
    """
    driver = get_driver()

    with driver.session() as session:
        # Verify both entities exist
        keep_node = session.run(
            "MATCH (e:Entity {name: $name}) RETURN e.name AS name, labels(e) AS labels",
            name=keep
        ).single()
        merge_node = session.run(
            "MATCH (e:Entity {name: $name}) RETURN e.name AS name, labels(e) AS labels",
            name=merge
        ).single()

        if not keep_node:
            return f"Entity '{keep}' not found."
        if not merge_node:
            return f"Entity '{merge}' not found."

        # Copy type labels from merge to keep
        merge_labels = [l for l in merge_node["labels"] if l != "Entity"]
        for label in merge_labels:
            try:
                sanitize_cypher_label(label)
                session.run(f"MATCH (e:Entity {{name: $name}}) SET e:{label}", name=keep)
            except ValueError:
                pass

        # Transfer outgoing relationships
        out_result = session.run(
            "MATCH (m:Entity {name: $merge})-[r]->(target) "
            "WHERE target.name <> $keep "
            "RETURN type(r) AS rel_type, target.name AS target, r.context AS context",
            merge=merge, keep=keep
        )
        out_transferred = 0
        for record in out_result:
            rel_type = record["rel_type"]
            try:
                sanitize_cypher_label(rel_type)
            except ValueError:
                continue
            session.run(
                f"MATCH (k:Entity {{name: $keep}}), (t:Entity {{name: $target}}) "
                f"MERGE (k)-[r:{rel_type}]->(t) "
                f"SET r.context = $context",
                keep=keep, target=record["target"], context=record["context"]
            )
            out_transferred += 1

        # Transfer incoming relationships
        in_result = session.run(
            "MATCH (source)-[r]->(m:Entity {name: $merge}) "
            "WHERE source.name <> $keep "
            "RETURN type(r) AS rel_type, source.name AS source, r.context AS context",
            merge=merge, keep=keep
        )
        in_transferred = 0
        for record in in_result:
            rel_type = record["rel_type"]
            try:
                sanitize_cypher_label(rel_type)
            except ValueError:
                continue
            session.run(
                f"MATCH (s:Entity {{name: $source}}), (k:Entity {{name: $keep}}) "
                f"MERGE (s)-[r:{rel_type}]->(k) "
                f"SET r.context = $context",
                source=record["source"], keep=keep, context=record["context"]
            )
            in_transferred += 1

        # Transfer EXTRACTED_FROM and MENTIONED_IN provenance
        session.run(
            "MATCH (m:Entity {name: $merge})-[r:EXTRACTED_FROM]->(d) "
            "MATCH (k:Entity {name: $keep}) "
            "MERGE (k)-[:EXTRACTED_FROM]->(d)",
            merge=merge, keep=keep
        )
        session.run(
            "MATCH (m:Entity {name: $merge})-[r:MENTIONED_IN]->(c) "
            "MATCH (k:Entity {name: $keep}) "
            "MERGE (k)-[:MENTIONED_IN]->(c)",
            merge=merge, keep=keep
        )

        # Delete the merged entity and all its relationships
        session.run(
            "MATCH (m:Entity {name: $merge}) DETACH DELETE m",
            merge=merge
        )

    return (
        f"Merged '{merge}' into '{keep}': "
        f"{out_transferred} outgoing + {in_transferred} incoming relationships transferred. "
        f"'{merge}' deleted."
    )


@mcp.tool()
def find_duplicates(similarity_threshold: float = 0.92) -> str:
    """Find potential duplicate entities by name similarity and embedding distance.

    Returns pairs of entities that might be the same real-world thing.
    Use merge_entities to combine confirmed duplicates.

    Args:
        similarity_threshold: Minimum cosine similarity to consider as duplicate (default 0.92).
    """
    driver = get_driver()
    duplicates = []

    with driver.session() as session:
        # Check for case-insensitive name matches first
        case_dupes = session.run(
            "MATCH (e:Entity) "
            "WITH toLower(e.name) AS lname, collect(e.name) AS names, count(*) AS cnt "
            "WHERE cnt > 1 RETURN names, cnt ORDER BY cnt DESC"
        )
        for record in case_dupes:
            duplicates.append({
                "type": "exact_match",
                "names": record["names"],
                "reason": "Same name (case-insensitive)"
            })

        # Check for embedding similarity between all entity pairs
        entities = session.run(
            "MATCH (e:Entity) WHERE e.embedding IS NOT NULL "
            "RETURN e.name AS name, labels(e) AS labels"
        )
        entity_list = [{"name": r["name"], "labels": r["labels"]} for r in entities]

        # For each entity, find its nearest neighbors by embedding
        for entity in entity_list:
            similar = session.run(
                dedent("""\
                    MATCH (e:Entity {name: $name})
                    CALL db.index.vector.queryNodes('entity_embedding', 5, e.embedding)
                    YIELD node, score
                    WHERE node.name <> $name AND score >= $threshold
                    RETURN node.name AS similar_name, score
                """),
                name=entity["name"], threshold=similarity_threshold
            )
            for record in similar:
                pair = tuple(sorted([entity["name"], record["similar_name"]]))
                dup_entry = {
                    "type": "embedding_similarity",
                    "names": list(pair),
                    "similarity": round(record["score"], 4),
                    "reason": f"Cosine similarity {record['score']:.4f}"
                }
                # Avoid duplicate pairs
                if dup_entry not in duplicates:
                    duplicates.append(dup_entry)

    if not duplicates:
        return "No potential duplicates found."

    # Deduplicate pairs
    seen_pairs = set()
    unique_dupes = []
    for d in duplicates:
        key = tuple(sorted(d["names"]))
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_dupes.append(d)

    lines = [f"Potential duplicates ({len(unique_dupes)}):"]
    for d in unique_dupes:
        lines.append(f"  - {' / '.join(d['names'])} — {d['reason']}")
    lines.append("\nUse merge_entities(keep='canonical_name', merge='duplicate_name') to merge.")

    return "\n".join(lines)


@mcp.tool()
def get_schema() -> str:
    """Get the current graph schema — node labels, relationship types, and property keys.

    Use this before writing Cypher queries to know what labels, relationships,
    and properties exist in the graph. Essential for text-to-Cypher translation.
    """
    driver = get_driver()

    with driver.session() as session:
        # Node labels and their counts
        label_result = session.run(
            "MATCH (n) UNWIND labels(n) AS label "
            "RETURN label, count(*) AS count ORDER BY count DESC"
        )
        labels = {r["label"]: r["count"] for r in label_result}

        # Relationship types and their counts
        rel_result = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
        rels = {r["type"]: r["count"] for r in rel_result}

        # Property keys per label (sample from first 100 nodes of each type)
        label_props = {}
        for label in labels:
            props_result = session.run(
                f"MATCH (n:{label}) WITH n LIMIT 100 "
                f"UNWIND keys(n) AS key "
                f"WITH key WHERE key <> 'embedding' "
                f"RETURN DISTINCT key ORDER BY key"
            )
            label_props[label] = [r["key"] for r in props_result]

        # Relationship patterns: which labels connect via which relationship types
        pattern_result = session.run(
            "MATCH (s)-[r]->(t) "
            "WITH labels(s) AS sl, type(r) AS rel, labels(t) AS tl "
            "RETURN DISTINCT sl, rel, tl LIMIT 100"
        )
        patterns = []
        for r in pattern_result:
            src = [l for l in r["sl"] if l != "Entity"]
            tgt = [l for l in r["tl"] if l != "Entity"]
            src_str = src[0] if src else "Entity"
            tgt_str = tgt[0] if tgt else "Entity"
            patterns.append(f"(:{src_str})-[:{r['rel']}]->(:{tgt_str})")

    lines = ["Graph Schema:", "", "Node Labels:"]
    for label, count in labels.items():
        props = label_props.get(label, [])
        props_str = f" — properties: {', '.join(props)}" if props else ""
        lines.append(f"  {label} ({count}){props_str}")

    lines.append("")
    lines.append("Relationship Types:")
    for rel, count in rels.items():
        lines.append(f"  {rel} ({count})")

    lines.append("")
    lines.append("Patterns:")
    for p in sorted(set(patterns)):
        lines.append(f"  {p}")

    return "\n".join(lines)


@mcp.tool()
def find_path(from_entity: str, to_entity: str, max_depth: int = 6) -> str:
    """Find the shortest path between two entities in the knowledge graph.

    This is a classic graph query that vector search cannot do. Use it to
    discover how two seemingly unrelated entities are connected.

    Args:
        from_entity: The starting entity name.
        to_entity: The target entity name.
        max_depth: Maximum path length to search (default 6, max 10).
    """
    driver = get_driver()
    max_depth = min(max_depth, 10)

    with driver.session() as session:
        # Shortest path
        result = session.run(
            dedent(f"""\
                MATCH path = shortestPath(
                    (a:Entity {{name: $from}})-[*..{max_depth}]-(b:Entity {{name: $to}})
                )
                RETURN [n IN nodes(path) | n.name] AS node_names,
                       [r IN relationships(path) | type(r)] AS rel_types,
                       [r IN relationships(path) | r.context] AS contexts,
                       length(path) AS hops
            """),
            **{"from": from_entity, "to": to_entity}
        )

        record = result.single()
        if not record:
            return f"No path found between '{from_entity}' and '{to_entity}' within {max_depth} hops."

        names = record["node_names"]
        rels = record["rel_types"]
        contexts = record["contexts"]
        hops = record["hops"]

        # Build readable path with contexts
        path_parts = []
        for i, name in enumerate(names):
            path_parts.append(f"({name})")
            if i < len(rels):
                ctx = f" [{contexts[i]}]" if contexts[i] else ""
                path_parts.append(f" -[{rels[i]}]{ctx}-> ")

        path_str = "".join(path_parts)

        # Also find alternative paths
        alt_result = session.run(
            dedent(f"""\
                MATCH path = (a:Entity {{name: $from}})-[*..{max_depth}]-(b:Entity {{name: $to}})
                WITH path, length(path) AS hops
                ORDER BY hops
                SKIP 1 LIMIT 3
                RETURN [n IN nodes(path) | n.name] AS node_names,
                       [r IN relationships(path) | type(r)] AS rel_types,
                       length(path) AS hops
            """),
            **{"from": from_entity, "to": to_entity}
        )

        alt_paths = []
        for alt in alt_result:
            alt_names = alt["node_names"]
            alt_rels = alt["rel_types"]
            parts = []
            for i, name in enumerate(alt_names):
                parts.append(f"({name})")
                if i < len(alt_rels):
                    parts.append(f" -[{alt_rels[i]}]-> ")
            alt_paths.append("".join(parts))

    lines = [
        f"Shortest path ({hops} hops):",
        path_str,
    ]
    if alt_paths:
        lines.append(f"\nAlternative paths ({len(alt_paths)}):")
        for ap in alt_paths:
            lines.append(f"  {ap}")

    return "\n".join(lines)


@mcp.tool()
def find_similar_entities(entity_name: str, top_k: int = 5) -> str:
    """Find entities most similar to a given entity by embedding distance.

    Useful for discovering related concepts, finding potential duplicates,
    or exploring the neighborhood of an entity in semantic space (not graph space).

    Args:
        entity_name: The entity to find similar entities to.
        top_k: Number of similar entities to return (default 5).
    """
    driver = get_driver()

    with driver.session() as session:
        # Get the entity's embedding
        entity = session.run(
            "MATCH (e:Entity {name: $name}) RETURN e.embedding AS embedding",
            name=entity_name
        ).single()

        if not entity or not entity["embedding"]:
            return f"Entity '{entity_name}' not found or has no embedding."

        # Vector search using the entity's own embedding
        results = session.run(
            dedent("""\
                CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
                YIELD node, score
                WHERE node.name <> $name
                RETURN node.name AS name, labels(node) AS labels, score
                ORDER BY score DESC
            """),
            k=top_k + 1, embedding=entity["embedding"], name=entity_name
        )

        similar = []
        for record in results:
            labels = [l for l in record["labels"] if l != "Entity"]
            type_str = f" [{labels[0]}]" if labels else ""
            similar.append({
                "name": record["name"],
                "type": type_str,
                "similarity": round(record["score"], 4)
            })

    if not similar:
        return f"No similar entities found for '{entity_name}'."

    lines = [f"Entities similar to '{entity_name}':"]
    for s in similar[:top_k]:
        lines.append(f"  - {s['name']}{s['type']} (similarity: {s['similarity']})")

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

### Step 3: Extract entities and relationships
Read through the text and identify important entities and their relationships.

**Entity types** (pick the most specific):
{entity_types}

**Relationship types** (pick the most specific):
{rel_types}

{DIRECTION_GUIDE}

### Step 4: Store facts in batch
Build a JSON array of all facts and call `store_facts` once (much faster than individual `store_fact` calls). Each fact needs:
- subject, predicate, obj, context, subject_type, obj_type, source

**Rules:**
- Normalize names: proper case, abbreviations that ARE the name stay uppercase (CGRP, TRPV1)
- Merge synonyms to canonical form: "B. bassiana" -> "Beauveria bassiana"
- Be specific and quantitative in context: "91.67% mortality at day 4" not "high mortality"
- Check relationship direction reads as natural sentence
- Prefer specific entity types: Chemical not Entity, Condition not Concept
- Reuse exact entity names from Step 1 for cross-document connections
- Set source to the document source name for provenance tracking

### Step 5: Report
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
Call `query` with the question. This now automatically:
- Searches text chunks by vector similarity
- Searches entities by vector similarity
- Traverses 1 hop from matched entities to include connected facts

Review all results — chunk matches for raw text, graph_fact matches for structured knowledge.

### Step 2: Deep exploration (if needed)
If Step 1 found relevant entities but you need more depth:
- Call `get_neighbors` on key entities to traverse further (2+ hops)
- Use `list_entities` with patterns to find related entities
- Use `cypher_query` for structured queries (e.g. "MATCH (c:Chemical)-[:TARGETS]->(t) RETURN c.name, t.name")

### Step 3: Synthesize
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
Call `graph_stats` to see overall counts and entity type distribution.

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
