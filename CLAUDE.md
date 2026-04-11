# GraphRag — Instructions for Claude

## What This Is

A local knowledge graph (Neo4j) + vector search (OpenAI embeddings) system that gives you persistent, queryable memory across conversations. You have MCP tools to store and retrieve knowledge.

## Available MCP Tools

### graphrag (your knowledge store)
- `graph_stats` — check what's in the graph
- `ingest_document(text, source)` — chunk + embed text (no entity extraction)
- `store_fact(subject, predicate, obj, context)` — store a knowledge triple
- `query(question)` — hybrid vector + graph search
- `get_neighbors(entity, depth)` — explore graph around an entity
- `list_entities(pattern)` / `list_documents()` — browse the graph
- `cypher_query(query)` — run raw Cypher
- `delete_document(source)` — remove a document

### neo4j-data-modeling (schema design)
- `validate_data_model` / `validate_node` / `validate_relationship` — schema validation
- `get_mermaid_config_str` — visualize a model
- `export_to_owl_turtle` / `export_to_arrows_json` — export ontology
- `list_example_data_models` / `get_example_data_model` — reference models

## Ingestion Workflow

The user prefers a collaborative, conversational approach:

1. **User shares a source** (paper, doc, code, URL)
2. **Read and discuss it together** — understand the content before extracting
3. **You extract entities and relationships** using your judgment + the ontology in `ontology.py`
4. **Call `ingest_document`** to store chunks with embeddings
5. **Call `store_fact`** for each entity/relationship you extract
6. **The user does NOT do bulk unattended ingestion** — at most 5-10 papers, one at a time or with subagents for parallel processing

## Entity Extraction Guidelines

- Use the ontology in `ontology.py` for allowed node labels and relationship types
- Normalize entity names: proper case, merge duplicates ("Python 3" and "python" -> "Python")
- Abbreviations that ARE the name stay uppercase: CGRP, TRPV1, TRP
- Use specific entity types: Chemical (not Entity) for "Substance P", Organism for "E. coli", Condition for "Allergic Rhinitis"
- Be selective — extract what matters for the user's work, not every noun
- Use `RELATED_TO` as catch-all relationship only when nothing specific fits
- Always provide `context` on `store_fact` calls — be specific and quantitative where possible
- Check relationship direction: (Subject)-[VERB]->(Object) should read as a natural sentence
- Use AUTHORED not CREATED_BY for authorship: (Person)-[AUTHORED]->(Paper)
- Check `list_entities` before creating new entities to reuse existing names and create cross-references

## Neo4j Access
- Bolt: `bolt://localhost:7687`
- Browser: `http://localhost:7474`
- Credentials: `neo4j` / `graphrag2024`
- If Neo4j is down, the MCP server will auto-start it via systemctl

## Key Files
- `server.py` — MCP server implementation
- `ontology.py` — allowed node labels, relationship types, extraction prompts
- `.env` — API keys (never commit)
