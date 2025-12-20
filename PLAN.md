# PCL Interpreter Implementation Plan

## Overview
This document tracks the implementation progress of the PCL (Pototo Core Language) interpreter, a dataflow-based interpreter that uses a producer/consumer protocol with guards and extents.

## Architecture

### Core Concepts
- **Operators**: Stateless, correspond to program syntax (e.g., `Literal`, `Var`, `VarRef`, `Lambda`)
- **Producers/Consumers**: Runtime stateful objects created from operators via `subscribe()`
- **Guards**: Represent regions (subsets of extent) via predicates. Monotonically growing.
- **Extents**: Represent the set of values a term can take on (its type)
- **VarScope**: Variable scope for looking up variables by name, with parent chaining for nested scopes
- **ColumnValue**: Columnar data representation with `parent_indices` for alignment across nesting levels

### Producer/Consumer Protocol
- `Consumer::notify(yield_guard)` - Producer notifies consumer that data is ready
- `Producer::get()` - Consumer requests data synchronously (returns `ColumnValue`)
- `Producer::release(obsolete_guard)` - Consumer retracts interest in a region

### Key Design Decisions
1. **Blanket implementations**: `Rc<RefCell<P>>` implements `Producer` when `P: Producer`, and `Rc<RefCell<C>>` implements `Consumer` when `C: Consumer`
2. **Variable system**: Variables are split into `Var` (operator) and `VarSub` (runtime state)
3. **VarScope**: Linked list structure for variable lookup with parent chaining; also tracks innermost scan for alignment
4. **Bound vs Scanning**: Lambda variables can be bound (from Application) or scanning (from aggregation)
5. **Scans as Joins**: Nested scanning lambdas execute as joins; predicates determine join strategy

## Completed ✅

### Step 1: Core Types
- [x] `Guard` enum with predicates (Equality, Membership, Inequality, And, Or, Function, Record)
- [x] `Extent` enum (Base, Function, Record, Union)
- [x] `Value` enum for runtime data representation
- [x] `FuncBinding` for function input-output pairs

### Step 2: Producer/Consumer Traits
- [x] `Consumer` trait with `notify()` method
- [x] `Producer` trait with `get()` and `release()` methods
- [x] `Operator` trait with `extent()` and `subscribe()` methods
- [x] Blanket implementations for `Rc<RefCell<>>` wrappers

### Step 3: Literal Operator
- [x] `Literal` operator implementation
- [x] `LiteralProducer` implementation
- [x] Immediate notification on subscribe
- [x] Tests for integer and string literals

### Step 4: Variable System
- [x] `Var` operator (name, definition, extent, predicate)
- [x] `VarSub` (implements both `Producer` and `Consumer`)
  - Stores yield guard (monotonically growing)
  - Manages list of consumers
  - Stores release guard for variable references
- [x] `VarRef` operator (looks up variable by name in VarScope)
- [x] `VarRefSub` producer (filters data based on intent guard)
- [x] `VarScope` for variable lookup with parent chaining
- [x] Basic variable test

### Step 5: Lambda Operator ✅
- [x] `Lambda` struct with variable and body
- [x] `extent()` implementation (computes function type from domain/codomain)
- [x] `subscribe()` implementation with domain/codomain guard splitting
- [x] `LambdaProducer` with proper notification handling from both variable and body
- [x] `release()` with guard splitting and propagation
- [x] Tests (identity, constant function, release, nested scope, function guards, notifications)

### Step 6: Columnar Values & Alignment ✅
- [x] Implement `ColumnValue` struct with `values` and `parent_indices`
- [x] Helper methods: `single()`, `from_values()`, `with_parent_indices()`, `expand()`
- [x] Update `Producer::get()` to return `ColumnValue` instead of `Value`
- [x] Update all existing producers (`LiteralProducer`, `VarSub`, `VarRefSub`, `LambdaProducer`)
- [x] Update tests for columnar values

## In Progress 🚧

### Step 7: Variable Binding Modes
- [x] Refactor `Var` to remove static `definition` field
- [x] Add `VarSource` enum to `VarSub` (Bound vs Scanning modes)
- [x] Update `Var::create_subscription()` with yield_guard parameter
- [x] Update Lambda with `subscribe_with_binding()` for Bound mode
- [x] Update tests for new API
- [ ] Add `innermost_scan` tracking to `VarScope`
- [ ] Implement alignment logic in `VarRefSub::get()`

### Step 8: Application Operator
- [ ] `Application` operator implementation
- [ ] Binds argument to lambda's variable (sets Bound mode)
- [ ] Bidirectional release flow (domain and codomain guards)
- [ ] Depends-image generation for codomain intent guard
- [ ] Tests

## TODO 📋

### High Priority - Core Operators
8. **Implement Cartesian product join**
   - Default when scanning variable has no correlation predicate
   - Generate `parent_indices` for cross-product pattern

9. **Memo operator**
    - Cache function bindings
    - Yield guard tracking
    - Obsolete guard management

10. **Integration tests**
    - Implement test-data operator.
    - Write tests that wire up operators and test the flow of data through them.

11. **Implement hash join**
   - When predicate is equality on outer variable (e.g., `t2.fk = t1.pk`)
   - Build hash table on outer variable values
   - Probe with inner variable, emit matching pairs with `parent_indices`

12. **Implement JoinStrategy selection**
    - Parse predicate to identify correlations with outer variables
    - Choose appropriate join strategy (Cartesian, Hash, etc.)

### Medium Priority - Other Operators
13. **Records operator**
    - Split guards for field subscriptions
    - Zip data from fields (now columnar)
    - Handle alignment when some fields ready but others aren't

14. **Let-bindings**
    - Existentially quantified variables (always bound)
    - Scope management

15. **Aggregation operators (sum, count, etc.)**
    - Consume lambda in scanning mode
    - Aggregate columnar values respecting `parent_indices` grouping

### Low Priority / Future
16. **Pattern matching**
    - Union handling
    - Case analysis

17. **Dependent records**
    - Type-level dependencies

18. **Cycle handling**
    - Ensure termination in cyclic dataflow graphs
    - Convergence guarantees

### Deferred Challenges ⏳
19. **Streaming joins**
    - Incremental join execution as yield guards advance
    - Symmetric hash join or similar for true streaming
    - See design.md "Open Challenges" section

20. **Guard expression evaluation**
    - How to evaluate complex expressions in guards (e.g., `t2.fk = t1.pk + 1`)
    - Requires dataflow machinery, but guards configure dataflow - circular dependency
    - See design.md "Open Challenges" section

21. **Multi-level nesting optimization**
    - Efficient composition of parent_indices through multiple levels
    - Precompute transitive indices vs. recompute on demand

## Architecture Notes

### Columnar Value Structure
```rust
struct ColumnValue {
    values: Vec<ScalarValue>,
    // Indices into parent level's batch (for alignment)
    // None if top-level or independent
    parent_indices: Option<Vec<usize>>,
}
```

### Variable Binding Modes
| Mode | When | Behavior |
|------|------|----------|
| **Bound** | Lambda applied (`(\x. body) arg`) | VarSub wraps producer from argument |
| **Scanning** | Lambda aggregated (`sum(\x. body)`) | VarSub iterates over extent, executes joins |

### Variable System Flow (Updated)
1. `Lambda::subscribe()` creates `VarSub` for lambda variable
   - If called via Application: variable is **bound** to argument's producer
   - If called via aggregation: variable is **scanning** over its extent
2. Adds variable to new `VarScope` with parent scope chained
3. If scanning, sets `innermost_scan` in VarScope
4. Subscribes to body with new scope
5. When body contains `VarRef`, it looks up variable in scope
6. `VarRef::subscribe()` creates `VarRefSub` with:
   - Reference to the `VarSub`
   - Reference to `innermost_scan` (for alignment if outer variable)
7. `VarRefSub::get()` expands outer variables using `parent_indices` from innermost scan

### Alignment Example
For `sum(\t1. sum(\t2. v(t1) + v(t2)))`:
```
t1 (outer scan): values=[A,B], parent_indices=None
t2 (inner scan): values=[1,2,3,4], parent_indices=[0,0,1,1]

When v(t1) is accessed inside inner lambda:
- VarRefSub sees innermost_scan = t2's VarSub
- Expands t1 values using t2's parent_indices
- Returns [A,A,B,B] (aligned with t2)

Now + operator can zip aligned values:
- v(t1): [vA, vA, vB, vB]
- v(t2): [v1, v2, v3, v4]
- Result: [vA+v1, vA+v2, vB+v3, vB+v4]
```

### Join Execution
Scanning variables with predicates referencing outer variables execute as joins:
- **Cartesian**: No predicate → cross product, regular parent_indices pattern
- **Hash join**: Equality predicate → build hash table on outer, probe with inner
- **Filter**: Other predicates → Cartesian + filter (or specialized index)

### Guard Monotonicity
- The contract of `notify()` guarantees that guards are monotonically growing
- `VarSub` stores a single yield guard (not a vec)
- Guards are unioned when updated (though current implementation just replaces - needs fix)

### Memory Management
- Using `Rc<RefCell<>>` for shared ownership of subscriptions
- TODO: Verify no memory leaks from Rc cycles
- Release guards used for garbage collection

## Testing Status
- [x] Literal tests (int, string) - updated for ColumnValue
- [x] Basic variable test - updated for ColumnValue
- [x] Lambda tests (extent, identity, constant, release, function guards, nested scope, notifications) - updated for ColumnValue
- [ ] Application tests
- [ ] ColumnValue alignment tests (expand, parent_indices)
- [ ] Join execution tests
- [ ] Integration tests

## Next Steps (Immediate)
1. Refactor `Var` to remove static definition, add `VarSource` enum to `VarSub`
2. Add `innermost_scan` tracking to `VarScope`
3. Implement alignment logic in `VarRefSub::get()` using `ColumnValue::expand()`
4. Implement `Application` operator (binds argument to lambda variable)
5. Implement basic Cartesian product join for scanning mode

## Questions / Open Issues
1. Should `Consumer::notify()` take `Guard` by reference instead of by value?
2. Memory leak concerns with Rc cycles - need to verify release guards prevent leaks
3. How does Application know to bind vs. let lambda scan? (Answer: Application always binds; aggregations like `sum` trigger scanning)
4. For streaming joins: when can we emit partial results? Need incremental join design.
5. For guard evaluation: how to handle guards with complex expressions that need dataflow to evaluate?

