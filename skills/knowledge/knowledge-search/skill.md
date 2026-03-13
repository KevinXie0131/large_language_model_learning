---
name: knowledge-search
description: |
  Knowledge base search assistant. Searches and retrieves information from local knowledge files
  stored in the knowledge directory (Finance_info, School_info, etc.).
  Reads markdown documents, CSV data files, and other structured/unstructured files to answer user questions.
  Finance_info contains company reports: 三一重工 (Sany Heavy Industry), 航天动力 (Aerospace Power / 陕西航天动力高科技).
  School_info contains academic data: students, teachers, courses, scores, classes.
  Trigger this skill when the user mentions "search", "find info", "look up", "query", "knowledge base",
  "finance info", "school info", "student", "teacher", "course", "score", "company report",
  "annual report", "三一重工", "Sany", "航天动力", "aerospace power",
  or asks questions that could be answered from knowledge files.
---

# Knowledge Base Search Assistant

> Search, retrieve, and synthesize information from local knowledge files to answer user questions accurately.

## Knowledge Base Structure

All knowledge files are stored under:
```
.claude/skills/knowledge/
├── Finance_info/               # Financial reports, company filings (Markdown)
│   ├── 三一重工.md             # Sany Heavy Industry report
│   └── 航天动力.md             # Shaanxi Aerospace Power Hi-Tech 2024 H1 report
└── School_info/                # Academic data (CSV)
    ├── students.csv
    ├── teachers.csv
    ├── courses.csv
    ├── scores.csv
    └── classes.csv
```

## Workflow

### Step 1: Understand the Query

Parse the user's question to identify:
- **Domain**: Which knowledge category (Finance, School, or cross-domain)
- **Entities**: Specific names, IDs, subjects, dates mentioned
- **Intent**: Lookup, comparison, aggregation, summary, or analysis

### Step 2: Locate Relevant Files

Scan the knowledge directory to find matching files:

```
Knowledge directory: .claude/skills/knowledge/
```

- For financial questions → read files in `Finance_info/`
- For school/academic questions → read relevant CSV files in `School_info/`
- If unclear which domain → scan all subdirectories

Use Glob to discover files, then Read to load content.

### Step 3: Extract & Answer

**For CSV data (School_info):**
- Parse CSV structure: understand column headers and data types
- Filter rows matching the query criteria
- Perform calculations if needed (averages, counts, rankings, etc.)
- Present results in a clear table or summary

**For Markdown documents (Finance_info):**
- Search for relevant sections using keywords from the query
- Extract specific data points, figures, or passages
- Summarize findings with citations to the source section

### Step 3b: Web Search Fallback

If the information **cannot be found** in the local knowledge base:

1. **Inform the user** that the answer was not found locally
2. **Use WebSearch** to search the web for the answer
3. **Use WebFetch** if needed to retrieve details from a specific URL found in search results
4. **Clearly label** web-sourced information as coming from the web (not from the local knowledge base)
5. **Cross-reference** web results with local data when partially relevant local data exists

> Priority order: Local knowledge base → Web search → Tell user the information is unavailable

### Step 4: Present Results

Format the response based on query type:

| Query Type | Response Format |
|-----------|----------------|
| Single fact lookup | Direct answer with source reference |
| Data query (filter/aggregate) | Markdown table + summary |
| Comparison | Side-by-side table or bullet points |
| Summary request | Structured summary with key highlights |
| Complex analysis | Refer to huashu-data-pro skill for deep analysis |

## Rules

- **Never fabricate data** — only return information found in the knowledge files
- **Cite sources** — mention which file the information came from
- **Handle missing data gracefully** — if the answer is not in the knowledge base, fall back to web search using WebSearch/WebFetch tools before saying the information is unavailable
- **For complex analysis** — if the user needs charts, reports, or deep analysis, suggest using the huashu-data-pro skill
- **Language** — respond in the same language the user uses
- **Be precise** — for numerical data, provide exact figures from the source
- **Distinguish sources** — clearly indicate whether information came from the local knowledge base or from a web search
