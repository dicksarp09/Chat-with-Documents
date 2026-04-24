from typing import List, Dict, Any, Optional


SYSTEM_PROMPT = """You are an expert document analysis AI. Your role is to analyze documents and provide structured, accurate insights.

CRITICAL RULES FOR FAITHFULNESS:
1. NEVER fabricate information - only use facts from the provided context
2. EVERY statement MUST be directly supported by evidence from the context
3. If information is not explicitly in the context, you MUST say "Not found in context"
4. Use node IDs to cite specific evidence for each claim
5. When uncertain, qualify statements as "inferred from context" not "stated"

Output format: Valid JSON only. No markdown, no explanations outside JSON structure."""


SUMMARY_PROMPT = """Analyze the following document context and provide a comprehensive summary.

Context:
{context}

Instructions:
1. Provide a clear, concise summary of the main content
2. Identify the document type and main theme
3. Highlight the most important information
4. Cite evidence using node IDs from the source

Output as JSON:
{{
    "summary": "2-3 sentence summary of the document",
    "document_type": "type of document (e.g., contract, report, policy)",
    "key_theme": "main theme or subject",
    "evidence": ["node_id_1", "node_id_2"]
}}
"""


KEY_POINTS_PROMPT = """Extract the most important key points from the following document context.

Context:
{context}

Instructions:
1. Extract 5-10 key points that capture the essential information
2. Categorize each key point appropriately
3. Provide specific evidence from the source for each point
4. Focus on factual, verifiable information

Output as JSON:
{{
    "key_points": [
        {{
            "text": "the key point text",
            "category": "category name",
            "evidence": ["node_id_1", "node_id_2"]
        }}
    ]
}}
"""


RISK_DETECTION_PROMPT = """Analyze the following context and identify potential risks.

Context:
{context}

Instructions:
1. Identify all potential risks mentioned or implied
2. Categorize risks (legal, financial, operational, compliance, etc.)
3. Assess severity level (high, medium, low)
4. Provide specific evidence for each risk identified

Output as JSON:
{{
    "risks": [
        {{
            "description": "description of the risk",
            "severity": "high|medium|low",
            "category": "risk category",
            "evidence": ["node_id_1", "node_id_2"]
        }}
    ]
}}
"""


OBLIGATION_EXTRACTION_PROMPT = """Extract all obligations from the following document context.

Context:
{context}

Instructions:
1. Identify all obligations, duties, and requirements
2. Determine which party is responsible for each obligation
3. Note any deadlines or timeframes mentioned
4. Provide evidence for each obligation

Output as JSON:
{{
    "obligations": [
        {{
            "description": "description of the obligation",
            "party": "responsible party",
            "deadline": "deadline if specified, null otherwise",
            "evidence": ["node_id_1", "node_id_2"]
        }}
    ]
}}
"""


ACTION_GENERATION_PROMPT = """Based on the following context, generate recommended actions.

Context:
{context}

Instructions:
1. Generate actionable tasks based on the document content
2. Prioritize actions as high, medium, or low
3. Explain the reason for each action
4. Provide evidence supporting each recommended action

Output as JSON:
{{
    "actions": [
        {{
            "task": "specific action to take",
            "priority": "high|medium|low",
            "reason": "why this action is needed",
            "evidence": ["node_id_1", "node_id_2"]
        }}
    ]
}}
"""


QUERY_ANSWER_PROMPT = """Answer the query using ONLY information from the provided context.

Query: {query}

Context:
{context}

STRICT RULES - VIOLATION = WRONG ANSWER:
1. Use the EXACT [source: node_id] markers from context to cite evidence
2. Do NOT fabricate or guess node IDs - only use ones that appear in the context
3. Do NOT add information not present in context
4. Do NOT generalize or paraphrase beyond what context states
5. If query asks about something not in context → respond: "Not found in context"
6. If partial information exists → answer with what exists, note what's missing

Question type detection:
- "Who/What/Where/When" → Extraction (specific facts needed)
- "Why/How does" → Reasoning (causal explanation needed)
- "What should/How to" → Recommendation (action steps needed)
- "Summarize/What are" → Summary (overview needed)

Output JSON (ALL fields required):
{{
    "answer": "YOUR ANSWER - cite sources using format [source: node_id]. Use ONLY IDs from context.",
    "evidence": ["list of node_ids from context"],
    "confidence": "explicitly_mentioned|partial|inferred|not_found",
    "key_points": ["specific details extracted from context"]
}}

Evidence examples (GOOD):
- "Vulnerable communities in sub-Saharan Africa are at risk [source: abc123]"
- "Climate change disrupts crop yields [source: def456]"

Non-evidence examples (BAD):
- "People in poor regions suffer" (no source)
- "xyz123" (node ID not in context)
- "The document suggests solutions" (too vague)
"""


FULL_ANALYSIS_PROMPT = """Perform a COMPREHENSIVE and DETAILED analysis of the following document context. Include ALL specific details.

Context:
{context}

Instructions:
1. Summarize the document with comprehensive details
2. Extract ALL key points with specific information
3. Identify risks with detailed descriptions
4. Extract ALL obligations with specific details
5. Generate recommended actions with thorough reasoning
6. Every insight must include evidence node references
7. Include specific names, dates, technologies, accomplishments, and quantifiable results

Output as JSON:
{{
    "summary": "comprehensive document summary with specific details",
    "key_points": [
        {{
            "text": "detailed key point text with specific information",
            "category": "category",
            "evidence": ["node_id_1"]
        }}
    ],
    "risks": [
        {{
            "description": "detailed risk description",
            "severity": "high|medium|low",
            "category": "risk category",
            "evidence": ["node_id_1"]
        }}
    ],
    "obligations": [
        {{
            "description": "detailed obligation description",
            "party": "responsible party",
            "deadline": "deadline or null",
            "evidence": ["node_id_1"]
        }}
    ],
    "actions": [
        {{
            "task": "detailed action task",
            "priority": "high|medium|low",
            "reason": "thorough reason for action",
            "evidence": ["node_id_1"]
        }}
    ]
}}
"""


COMPRESSION_PROMPT = """Given the following context and query, extract ONLY the most relevant information.

Query: {query}

Context:
{context}

CRITICAL: Preserve the EXACT text from context - do not paraphrase or summarize.
This is used for answer generation, so accuracy is essential.

Instructions:
1. Extract sentences DIRECTLY relevant to answering the query
2. Keep the exact wording - copy from context verbatim
3. Preserve the [source: node_id] markers in the context - these are REQUIRED for citations
4. Preserve factual information and key details exactly as stated
5. Remove only redundant or clearly irrelevant content
6. Maintain logical flow where possible

Output format: Compressed context with [source: id] markers. No JSON."""


def get_summary_prompt(context: str) -> str:
    return SUMMARY_PROMPT.format(context=context)


def get_key_points_prompt(context: str) -> str:
    return KEY_POINTS_PROMPT.format(context=context)


def get_risk_detection_prompt(context: str) -> str:
    return RISK_DETECTION_PROMPT.format(context=context)


def get_obligation_prompt(context: str) -> str:
    return OBLIGATION_EXTRACTION_PROMPT.format(context=context)


def get_action_prompt(context: str) -> str:
    return ACTION_GENERATION_PROMPT.format(context=context)


def get_query_prompt(query: str, context: str) -> str:
    return QUERY_ANSWER_PROMPT.format(query=query, context=context)


def get_full_analysis_prompt(context: str) -> str:
    return FULL_ANALYSIS_PROMPT.format(context=context)


def get_compression_prompt(query: str, context: str) -> str:
    return COMPRESSION_PROMPT.format(query=query, context=context)
