---
layout: default
title: "Node Documentation"
nav_order: 3
---

# Medical Agent Nodes Documentation

This document provides detailed technical documentation for all nodes in the medical conversation system.

## Node Architecture Overview

All nodes follow the PocketFlow framework pattern with three main methods:
- `prep(shared)`: Prepare data from shared store
- `exec(prep_res)`: Execute core logic (usually LLM calls)
- `post(shared, prep_res, exec_res)`: Store results and determine routing

## Node Detailed Specifications

### 1. IngestQuery

**Purpose**: Entry point and input normalization

**Methods**:
- `prep(shared)`: Extracts `role` and `input` from shared store
- `exec(inputs)`: Normalizes and validates input data
- `post(shared, prep_res, exec_res)`: Stores processed `query` and `role`

**Shared Store Updates**:
- `shared["role"]`: User role identifier
- `shared["query"]`: Normalized query string

**Routing**: Always returns "default" → MainDecisionAgent

**Example**:
```python
# Input: shared = {"role": "patient_dental", "input": "  Tôi bị đau răng  "}
# Output: shared["query"] = "Tôi bị đau răng", shared["role"] = "patient_dental"
```

---

### 2. MainDecisionAgent

**Purpose**: Central classification and intelligent routing

**Methods**:
- `prep(shared)`: Reads `query` and `role` for classification
- `exec(inputs)`: Uses LLM with structured prompt for input classification
- `post(shared, prep_res, exec_res)`: Stores classification results and routes

**Classification Types**:
- `medical_question`: Queries requiring medical knowledge
- `greeting`: Welcome/hello messages  
- `topic_suggest`: Requests for topic suggestions
- `statement`: General statements
- `nonsense`: Unclear/invalid inputs

**LLM Prompt Features**:
- Role-aware classification
- Confidence scoring
- RAG question generation
- Reasoning explanation

**Shared Store Updates**:
- `shared["input_type"]`: Classification result
- `shared["classification_confidence"]`: Confidence level
- `shared["classification_reason"]`: Classification reasoning
- `shared["rag_questions"]`: Additional questions for enhanced retrieval

**Routing Logic**:
- `medical_question` → "retrieve_kb"
- `greeting` → "greeting"  
- Other types → "topic_suggest"

**Example Classification**:
```yaml
type: medical_question
confidence: high
reason: User is asking about dental pain symptoms
rag_questions:
  - "Triệu chứng đau răng thường gặp"
  - "Cách xử lý đau răng tại nhà"
```

---

### 3. RetrieveFromKB

**Purpose**: Knowledge base search and retrieval

**Methods**:
- `prep(shared)`: Combines `query` with `rag_questions` for comprehensive search
- `exec(inputs)`: Performs TF-IDF based similarity search in role-specific KB
- `post(shared, prep_res, exec_res)`: Stores retrieval results and scores

**Search Features**:
- Role-based knowledge filtering
- TF-IDF vectorization with cosine similarity
- Top-K retrieval (default: 7 results)
- Performance timing logs

**Knowledge Base Structure**:
Each KB entry contains:
- `ĐỀ MỤC`: Main topic category
- `CHỦ ĐỀ CON`: Subtopic
- `MÃ SỐ`: Reference code
- `CÂU HỎI`: Question
- `CÂU TRẢ LỜI`: Answer
- `keywords`: Search keywords

**Shared Store Updates**:
- `shared["retrieved"]`: List of retrieved KB entries with scores
- `shared["retrieval_score"]`: Best similarity score
- `shared["need_clarify"]`: Boolean for clarification need

**Routing**: Always returns "default" → ScoreDecisionNode

---

### 4. ScoreDecisionNode

**Purpose**: Quality-based routing decision maker

**Methods**:
- `prep(shared)`: Reads `input_type` and `retrieval_score`
- `exec(inputs)`: Evaluates score against threshold for routing decision
- `post(shared, prep_res, exec_res)`: Sets response context and routes

**Decision Logic**:
```python
if input_type == "medical_question":
    if retrieval_score >= threshold:
        return "compose_answer"  # High confidence
    else:
        return "clarify"        # Low confidence
else:
    return "clarify"           # Non-medical queries
```

**Threshold Configuration**: Configurable via `get_score_threshold()` utility

**Shared Store Updates**:
- `shared["response_context"]`: Context for response generation

**Routing Options**:
- `compose_answer`: High-confidence medical responses
- `clarify`: Low-confidence or clarification needed
- `topic_suggest`: Topic suggestions

---

### 5. ComposeAnswer

**Purpose**: Generate comprehensive medical answers

**Methods**:
- `prep(shared)`: Gathers role, query, retrieved KB data, and conversation history
- `exec(inputs)`: Uses LLM with role-specific persona to generate structured answer
- `post(shared, prep_res, exec_res)`: Stores formatted response

**LLM Features**:
- Role-specific persona adaptation
- Knowledge base context integration
- Conversation history consideration
- Structured YAML output validation

**Persona Configuration**:
Each role has specific persona settings:
```python
PERSONA_BY_ROLE = {
    "patient_dental": {
        "persona": "Bác sĩ nha khoa",
        "audience": "bệnh nhân nha khoa", 
        "tone": "Thái độ thân thiện, Ngôn ngữ thân thiện, không dùng từ chuyên môn"
    }
}
```

**Output Structure**:
```yaml
explanation: "Detailed medical explanation..."
suggestion_questions:
  - "Follow-up question 1"
  - "Follow-up question 2"
```

**Shared Store Updates**:
- `shared["answer_obj"]`: Complete structured response
- `shared["explain"]`: Main explanation text
- `shared["suggestion_questions"]`: Follow-up suggestions

**Error Handling**: Fallback to generic response if LLM fails or returns invalid format

---

### 6. ClarifyQuestionNode

**Purpose**: Handle low-confidence medical queries

**Methods**:
- `prep(shared)`: Reads role, query, retrieved data, and RAG questions
- `exec(inputs)`: Generates clarification message and related question suggestions
- `post(shared, prep_res, exec_res)`: Stores clarification response

**Question Generation Strategy**:
- Use questions from retrieved KB entries if available
- Fall back to random sampling from role-specific KB
- Provide 5+ suggestion questions for exploration

**Response Format**:
```python
{
    "explain": "Clarification message explaining the situation",
    "suggestion_questions": ["Related question 1", "Related question 2", ...],
    "preformatted": True
}
```

**Shared Store Updates**:
- `shared["answer_obj"]`: Clarification response object
- `shared["explain"]`: Clarification message
- `shared["suggestion_questions"]`: Related question suggestions

---

### 7. TopicSuggestResponse

**Purpose**: Provide topic exploration and suggestions

**Methods**:
- `prep(shared)`: Reads user role and query context
- `exec(inputs)`: Generates role-specific topic suggestions via random KB sampling
- `post(shared, prep_res, exec_res)`: Stores topic suggestion response

**Topic Generation**:
- Random sampling from role-specific knowledge base
- 10 diverse topic suggestions for exploration
- Role-appropriate content filtering

**Response Format**:
```python
{
    "explain": "Friendly introduction to topic suggestions",
    "suggestion_questions": ["Topic 1", "Topic 2", ..., "Topic 10"],
    "preformatted": True
}
```

**Use Cases**:
- User explicitly requests topic suggestions
- General conversation starters
- Exploration of available medical knowledge areas

---

### 8. GreetingResponse

**Purpose**: Handle greeting messages and welcome interactions

**Methods**:
- `prep(shared)`: Reads role and query for context
- `exec(inputs)`: Sets greeting context flag
- `post(shared, prep_res, exec_res)`: Stores welcome message

**Response**:
- Friendly welcome message
- Sets up transition to topic exploration
- Always routes to TopicSuggestResponse for follow-up

**Shared Store Updates**:
- `shared["explain"]`: Welcome greeting message

**Routing**: Always returns "default" → TopicSuggestResponse

---

## Node Communication Patterns

### Data Flow Between Nodes

1. **IngestQuery** → **MainDecisionAgent**
   - Passes: normalized query, user role
   
2. **MainDecisionAgent** → **RetrieveFromKB**
   - Passes: classification results, RAG questions
   
3. **RetrieveFromKB** → **ScoreDecisionNode**
   - Passes: retrieved knowledge, similarity scores
   
4. **ScoreDecisionNode** → **ComposeAnswer/ClarifyQuestionNode**
   - Passes: routing decision based on confidence

### Error Handling Patterns

All nodes implement graceful error handling:
- LLM call failures fall back to default responses
- Invalid data structures trigger fallback behaviors
- Logging provides visibility into failure modes

### Performance Considerations

- **Retrieval Timing**: KB search performance is logged for optimization
- **LLM Caching**: Consider implementing caching for repeated queries
- **Batch Processing**: Single requests processed individually (no batching)

## Extension Points

### Adding New Node Types

To add new nodes:
1. Inherit from `Node` class
2. Implement `prep`, `exec`, `post` methods
3. Update flow routing in `flow.py`
4. Add appropriate logging

### Customizing Existing Nodes

- **Prompts**: Modify prompts in `utils/prompts.py`
- **Personas**: Update role configurations in `utils/role_ENUM.py`
- **Thresholds**: Adjust score thresholds in `utils/helpers.py`
- **KB Structure**: Modify knowledge base format in `utils/kb.py`
