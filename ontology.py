"""
GraphRag Ontology — Defines allowed node labels, relationship types, and extraction rules.

Design principles (from Microsoft GraphRAG, Neo4j LLM Graph Builder, KAG/OpenSPG):
- Start with a constrained seed ontology for precision
- Allow catch-all Entity nodes for things that don't fit
- Periodically review catch-alls and promote recurring types
- Keep relationship types tight (15-20 core)
- Maintain document→chunk→entity provenance chain always
- Both structured triples AND text chunks, cross-linked
"""

# ---------------------------------------------------------------------------
# Node Labels — organized by domain
# ---------------------------------------------------------------------------

# Core structural nodes (RAG plumbing — always present)
STRUCTURAL_LABELS = {
    "Document": "Source document that was ingested",
    "Chunk": "Text chunk from a document, with embedding",
    "Community": "Cluster of related entities (for global queries)",
}

# General-purpose entity types
GENERAL_LABELS = {
    "Entity": "Catch-all for unclassified named entities (review periodically)",
    "Person": "People, authors, contributors, users",
    "Organization": "Companies, institutions, teams, groups",
    "Location": "Geographic places, addresses, regions",
    "Event": "Named events, incidents, milestones, releases",
    "Concept": "Abstract ideas, topics, themes, paradigms",
    "Tool": "Software tools, utilities, applications",
}

# Software/Code domain
SOFTWARE_LABELS = {
    "Repository": "Code repositories",
    "Module": "Logical code units, packages, libraries",
    "Class": "OOP classes, structs, types",
    "Function": "Functions, methods, procedures",
    "API": "Endpoints, interfaces, protocols",
    "Technology": "Languages, frameworks, platforms, runtimes",
    "Error": "Bugs, exceptions, error types, issues",
    "Configuration": "Config files, env variables, settings",
}

# Research/Academic domain
RESEARCH_LABELS = {
    "Paper": "Academic publications, articles, preprints",
    "Dataset": "Named datasets used in research or ML",
    "Method": "Algorithms, techniques, approaches, architectures",
    "Metric": "Evaluation measures, benchmarks, scores",
    "Finding": "Key results, claims, conclusions",
    "Theory": "Theoretical frameworks, hypotheses",
}

# All entity labels (excluding structural)
ALL_ENTITY_LABELS = {**GENERAL_LABELS, **SOFTWARE_LABELS, **RESEARCH_LABELS}

# ---------------------------------------------------------------------------
# Relationship Types — organized by domain
# ---------------------------------------------------------------------------

# Structural relationships (RAG plumbing)
STRUCTURAL_RELATIONSHIPS = {
    "HAS_CHUNK": "Document → Chunk (provenance)",
    "NEXT": "Chunk → Chunk (sequential ordering)",
    "EXTRACTED_FROM": "Entity → Chunk (provenance — where entity was found)",
    "IN_COMMUNITY": "Entity → Community (clustering)",
}

# General-purpose relationships (the core ~15)
GENERAL_RELATIONSHIPS = {
    "RELATED_TO": "Generic association (use when no specific type fits)",
    "PART_OF": "Composition/membership (X is part of Y)",
    "HAS_PART": "Inverse of PART_OF",
    "LOCATED_IN": "Spatial containment",
    "MEMBER_OF": "Person/entity belongs to organization/group",
    "CREATED_BY": "Authorship, creation",
    "OCCURRED_AT": "Temporal placement of events",
    "CAUSES": "Causal relationship",
    "DEPENDS_ON": "Dependency (general)",
    "SIMILAR_TO": "Similarity or equivalence",
    "DERIVED_FROM": "Origin, inheritance, fork",
    "MENTIONS": "Reference without deep relationship",
    "DESCRIBES": "Documentation, explanation",
}

# Software/Code relationships
SOFTWARE_RELATIONSHIPS = {
    "IMPORTS": "Module/package import",
    "CALLS": "Function/method invocation",
    "EXTENDS": "Inheritance, subclassing",
    "IMPLEMENTS": "Interface implementation",
    "CONFIGURES": "Configuration relationship",
    "DEPLOYS_TO": "Deployment target",
    "THROWS": "Exception/error production",
    "TESTS": "Test coverage",
    "DOCUMENTS": "Documentation relationship",
    "USES": "General usage of a tool/technology/library",
}

# Research/Academic relationships
RESEARCH_RELATIONSHIPS = {
    "CITES": "Citation between papers",
    "EVALUATES_ON": "Evaluation on a dataset/benchmark",
    "PROPOSES": "Paper proposes a method/theory",
    "OUTPERFORMS": "Comparative superiority",
    "USES_METHOD": "Application of a method",
    "TRAINED_ON": "Model training data",
    "REPLICATES": "Replication of prior work",
}

ALL_RELATIONSHIPS = {
    **STRUCTURAL_RELATIONSHIPS,
    **GENERAL_RELATIONSHIPS,
    **SOFTWARE_RELATIONSHIPS,
    **RESEARCH_RELATIONSHIPS,
}

# ---------------------------------------------------------------------------
# Extraction prompt template — used when LLM extracts entities from text
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_PROMPT = """\
Extract entities and relationships from the following text.

## Allowed Entity Types
{entity_types}

If an entity doesn't fit any type above, use "Entity" with a descriptive `subtype` property.

## Allowed Relationship Types
{relationship_types}

If a relationship doesn't fit any type above, use "RELATED_TO" with a descriptive `context`.

## Rules
1. Entity names should be normalized: proper case, no abbreviations unless the abbreviation IS the name.
2. Merge duplicates: "Python 3", "Python", "python" → "Python".
3. Be specific with relationship types — prefer specific over RELATED_TO.
4. Every entity MUST have a type from the allowed list.
5. Return JSON array of objects with this structure:

For entities:
{{"type": "entity", "label": "Person", "name": "Guido van Rossum", "properties": {{"subtype": null}}}}

For relationships:
{{"type": "relationship", "source": "Guido van Rossum", "target": "Python", "relationship": "CREATED_BY", "context": "Creator of the Python programming language"}}

## Text
{text}

## Output (JSON array only, no explanation)
"""


def get_extraction_prompt(text: str, domains: list[str] | None = None) -> str:
    """Build an entity extraction prompt for the given text and domains.

    Args:
        text: The text to extract from.
        domains: Optional list of domains to include ("general", "software", "research").
                 Defaults to all.
    """
    domains = domains or ["general", "software", "research"]

    labels = dict(GENERAL_LABELS)
    rels = dict(GENERAL_RELATIONSHIPS)

    if "software" in domains:
        labels.update(SOFTWARE_LABELS)
        rels.update(SOFTWARE_RELATIONSHIPS)
    if "research" in domains:
        labels.update(RESEARCH_LABELS)
        rels.update(RESEARCH_RELATIONSHIPS)

    entity_types = "\n".join(f"- **{k}**: {v}" for k, v in labels.items())
    relationship_types = "\n".join(f"- **{k}**: {v}" for k, v in rels.items())

    return ENTITY_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        relationship_types=relationship_types,
        text=text,
    )


def validate_relationship_type(rel_type: str) -> str:
    """Normalize and validate a relationship type. Returns the closest match or RELATED_TO."""
    normalized = rel_type.upper().replace(" ", "_").replace("-", "_")
    if normalized in ALL_RELATIONSHIPS:
        return normalized
    # Fuzzy fallback — check if it's a substring match
    for known in ALL_RELATIONSHIPS:
        if normalized in known or known in normalized:
            return known
    return normalized  # Allow it but it won't be in the "blessed" set


def validate_entity_label(label: str) -> str:
    """Normalize and validate an entity label. Returns the closest match or 'Entity'."""
    normalized = label.strip().title().replace(" ", "")
    if normalized in ALL_ENTITY_LABELS:
        return normalized
    if normalized in STRUCTURAL_LABELS:
        return normalized
    # Check case-insensitive
    for known in ALL_ENTITY_LABELS:
        if normalized.lower() == known.lower():
            return known
    return "Entity"  # Catch-all
