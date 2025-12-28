# Memory Management Architecture

## Overview

The memory management system has been refactored into a modular architecture with one orchestrator node and three specialized worker nodes. This design enables parallel execution and better separation of concerns.

## Node Architecture

```
User Query
    ↓
RetrieveFromMemory (get top 10 relevant memories)
    ↓
MemoryManager (orchestrator - decides operations via LLM)
    ↓
    ├── AddMemory (INSERT operations) ──┐
    ├── UpdateMemory (UPDATE operations) ┼─→ Execute in parallel
    └── DeleteMemory (DELETE operations) ┘
    ↓
Continue flow
```

## Nodes Description

### 1. RetrieveFromMemory (Existing - Enhanced)
- **Purpose**: Retrieve top 10 most relevant memories based on user query
- **Input**: User query, context, role
- **Output**: List of relevant memories with IDs and scores
- **Enhancement**: Now uses LLM to create optimized memory retrieval query

### 2. MemoryManager (New - Orchestrator)
- **Purpose**: Analyze conversation and existing memories to decide operations
- **Input**:
  - User query
  - AI response
  - Retrieved memories (from RetrieveFromMemory)
  - Context and role
- **LLM Task**: Decide which operations to perform:
  - `INSERT`: Add new memories
  - `UPDATE`: Modify existing memories
  - `DELETE`: Remove outdated/incorrect memories
- **Output**: Structured operations list
  ```python
  {
    "operations": {
      "insert": [{"content": "..."}],
      "update": [{"memory_id": "...", "content": "..."}],
      "delete": [{"memory_id": "..."}]
    },
    "reason": "...",
    "importance": "high|medium|low"
  }
  ```

### 3. AddMemory (Worker)
- **Purpose**: Execute INSERT operations
- **Input**: List of insert operations from MemoryManager
- **Process**: Create new memory entries in Qdrant
- **Output**: Results with success count and details

### 4. UpdateMemory (Worker)
- **Purpose**: Execute UPDATE operations
- **Input**: List of update operations with memory IDs
- **Process**: Update existing memory entries using point_id
- **Output**: Results with success count and details

### 5. DeleteMemory (Worker)
- **Purpose**: Execute DELETE operations
- **Input**: List of delete operations with memory IDs
- **Process**: Batch delete memory entries from Qdrant
- **Output**: Results with success count and details

## Flow Integration

### Example Flow Configuration

```python
# In your MedFlow or similar flow:

# Step 1: Retrieve relevant memories
retrieve_memory = RetrieveFromMemory()

# Step 2: Orchestrator decides operations
memory_manager = MemoryManager()

# Step 3: Worker nodes execute operations (can run in parallel)
add_memory = AddMemory()
update_memory = UpdateMemory()
delete_memory = DeleteMemory()

# Connect nodes
flow.add_node("retrieve_memory", retrieve_memory)
flow.add_node("memory_manager", memory_manager)
flow.add_node("add_memory", add_memory)
flow.add_node("update_memory", update_memory)
flow.add_node("delete_memory", delete_memory)

# Define edges
flow.add_edge("retrieve_memory", "memory_manager")
flow.add_edge("memory_manager", "add_memory")
flow.add_edge("memory_manager", "update_memory")
flow.add_edge("memory_manager", "delete_memory")
```

## Async Parallel Batch Execution

All nodes now use **async/await** for non-blocking execution:

### MemoryManager (AsyncNode)
- Uses `AsyncNode` for async LLM calls
- Decides all operations in a single LLM call
- No blocking while waiting for LLM response

### Worker Nodes (AsyncParallelBatchNode)
- **AddMemory**: Uses `AsyncParallelBatchNode`
  - Executes all INSERT operations **in parallel**
  - Each insert runs concurrently using `asyncio.gather`
  - Uses `run_in_executor` to avoid blocking on sync Qdrant calls

- **UpdateMemory**: Uses `AsyncParallelBatchNode`
  - Executes all UPDATE operations **in parallel**
  - Each update runs concurrently using `asyncio.gather`
  - Uses `run_in_executor` for non-blocking Qdrant updates

- **DeleteMemory**: Uses `AsyncNode` with batch delete
  - Collects all DELETE IDs and executes in **one batch call**
  - More efficient than individual deletes
  - Uses `run_in_executor` for non-blocking Qdrant delete

### Why Parallel Execution is Safe
The worker nodes can run in parallel because:
- They operate on different operations (no conflicts)
- Each has its own subset of data from MemoryManager
- They all write to different memory entries (different point IDs)
- INSERT creates new IDs (no collision)
- UPDATE uses existing IDs (no overlap with INSERT)
- DELETE removes existing IDs (no overlap with INSERT)

## Benefits

1. **Scalability**: Worker nodes can run concurrently
2. **Separation of Concerns**: Each node has a single responsibility
3. **Maintainability**: Easy to modify/debug individual operations
4. **Flexibility**: Can easily add new operation types
5. **Observability**: Clear logging at each stage

## Shared State

Data passed between nodes via `shared` state:

```python
shared = {
  # From RetrieveFromMemory
  "relevant_memories": [...],

  # From MemoryManager
  "memory_operations": {
    "insert": [...],
    "update": [...],
    "delete": [...]
  },
  "memory_manager_reason": "...",
  "memory_importance": "high",

  # From Worker Nodes
  "add_memory_result": {...},
  "update_memory_result": {...},
  "delete_memory_result": {...}
}
```

## Example Use Cases

### Case 1: New User Information
```
User: "Tôi 25 tuổi, làm kỹ sư phần mềm"
Retrieved Memories: []

MemoryManager Decision:
  INSERT: ["Người dùng 25 tuổi, nghề kỹ sư phần mềm"]

AddMemory: Creates new entry
```

### Case 2: Update Existing Information
```
User: "Giờ tôi 26 tuổi rồi"
Retrieved Memories: [
  {id: "abc-123", content: "Người dùng 25 tuổi, nghề kỹ sư"}
]

MemoryManager Decision:
  UPDATE: [{memory_id: "abc-123", content: "Người dùng 26 tuổi, nghề kỹ sư"}]

UpdateMemory: Updates entry abc-123
```

### Case 3: Correction (Delete + Insert)
```
User: "Không, tôi không phải bác sĩ, tôi là kỹ sư"
Retrieved Memories: [
  {id: "xyz-456", content: "Người dùng là bác sĩ"}
]

MemoryManager Decision:
  DELETE: [{memory_id: "xyz-456"}]
  INSERT: ["Người dùng là kỹ sư phần mềm"]

DeleteMemory: Removes xyz-456
AddMemory: Creates new entry
```

### Case 4: Complex Multi-Operation
```
User: "Tôi đã chuyển từ Hà Nội vào Sài Gòn, bỏ nghề giáo viên làm kỹ sư"
Retrieved Memories: [
  {id: "m1", content: "Người dùng sống ở Hà Nội"},
  {id: "m2", content: "Người dùng là giáo viên"}
]

MemoryManager Decision:
  UPDATE: [{memory_id: "m1", content: "Người dùng sống ở Sài Gòn (chuyển từ Hà Nội)"}]
  UPDATE: [{memory_id: "m2", content: "Người dùng là kỹ sư (trước đây là giáo viên)"}]

UpdateMemory: Updates both m1 and m2 in batch
```

## Migration from Old SaveToMemory

The old `SaveToMemory` node can be deprecated or kept for backward compatibility. The new architecture provides:
- Better observability (separate logs for each operation type)
- Parallel execution capability
- Clearer separation of LLM decision-making vs execution

## Performance Considerations

### Optimization Strategies
1. **Single LLM Call**: MemoryManager makes only 1 LLM call to decide all operations
2. **Parallel Execution**:
   - AddMemory processes all INSERTs concurrently via `asyncio.gather`
   - UpdateMemory processes all UPDATEs concurrently via `asyncio.gather`
   - All 3 worker nodes can run in parallel if operations exist
3. **Batch Delete**: DeleteMemory deletes all IDs in one Qdrant API call
4. **Non-Blocking IO**: All nodes use `run_in_executor` to avoid blocking on sync Qdrant calls
5. **Optimized Retrieval**: RetrieveFromMemory returns top 10 (configurable) for optimal context vs performance

### Performance Example
If MemoryManager decides:
- 5 INSERT operations
- 3 UPDATE operations
- 2 DELETE operations

**Sequential execution would take**: T_insert1 + T_insert2 + ... + T_update1 + ... + T_delete

**Parallel execution takes**: max(T_insert_batch, T_update_batch, T_delete_batch)

Where:
- `T_insert_batch` = max time among 5 parallel inserts
- `T_update_batch` = max time among 3 parallel updates
- `T_delete_batch` = time for batch delete of 2 IDs

**Speedup**: Can be 5-10x faster for multiple operations!

## Error Handling

Each node handles errors independently:
- Failed INSERT: Logs error, continues with other inserts
- Failed UPDATE: Logs error, continues with other updates
- Failed DELETE: Batch fails, but logs which IDs were attempted
- MemoryManager LLM failure: Falls back to skip all operations

## Logging

Clear emoji-based logging for easy debugging:
- 🧠 RetrieveFromMemory
- 🎯 MemoryManager
- ➕ AddMemory
- 🔄 UpdateMemory
- 🗑️ DeleteMemory
