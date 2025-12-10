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

## In Progress 🚧

### Step 5: Lambda Operator
- [x] `Lambda` struct with variable and body
- [x] Basic `subscribe()` implementation
- [ ] `extent()` implementation (needs to compute function type from domain/codomain)
- [ ] Proper handling of domain/codomain guards
- [ ] Tests

### Step 6: Application Operator
- [ ] `Application` operator implementation
- [ ] Bidirectional release flow (domain and codomain guards)
- [ ] Depends-image generation for codomain intent guard
- [ ] Tests

## TODO 📋

### High Priority - Columnar Values & Alignment
1. **Implement ColumnValue**
   - Replace scalar `Value` with columnar `ColumnValue` representation
   - Add `parent_indices: Option<Vec<usize>>` for alignment tracking
   - Update `Producer::get()` to return `ColumnValue`

2. **Refactor Var to remove static definition**
   - Remove `definition` field from `Var` operator
   - Binding happens dynamically via Application or remains unbound for scans

3. **Implement VarSource enum in VarSub**
   - `Bound(Box<dyn Producer>)` - wraps producer from Application
   - `Scanning { extent, predicate, correlations }` - iterates over extent
   - VarSub uses appropriate source based on how lambda is used

4. **Implement alignment in VarRefSub**
   - Track `innermost_scan: Option<Rc<RefCell<VarSub>>>`
   - On `get()`: if referencing outer variable, expand using inner scan's `parent_indices`
   - Enables vectorized operations on aligned batches

5. **Extend VarScope to track innermost scan**
   - Add `innermost_scan` field to VarScope
   - Lambda sets this when its variable is scanning
   - VarRef lookups receive alignment context

### High Priority - Core Operators
6. **Implement Application operator**
   - Binds argument to lambda's variable (sets to Bound mode)
   - Handle function application with proper guard propagation
   - Implement bidirectional release flow as specified in design doc

7. **Complete Lambda::extent()**
   - Compute function extent from variable extent (domain) and body extent (codomain)
   - Store it in Lambda struct to avoid recomputation

### Medium Priority - Joins
8. **Implement Cartesian product join**
   - Default when scanning variable has no correlation predicate
   - Generate `parent_indices` for cross-product pattern

9. **Implement hash join**
   - When predicate is equality on outer variable (e.g., `t2.fk = t1.pk`)
   - Build hash table on outer variable values
   - Probe with inner variable, emit matching pairs with `parent_indices`

10. **Implement JoinStrategy selection**
    - Parse predicate to identify correlations with outer variables
    - Choose appropriate join strategy (Cartesian, Hash, etc.)

### Medium Priority - Other Operators
11. **Records operator**
    - Split guards for field subscriptions
    - Zip data from fields (now columnar)
    - Handle alignment when some fields ready but others aren't

12. **Let-bindings**
    - Existentially quantified variables (always bound)
    - Scope management

13. **Aggregation operators (sum, count, etc.)**
    - Consume lambda in scanning mode
    - Aggregate columnar values respecting `parent_indices` grouping

### Low Priority / Future
14. **Pattern matching**
    - Union handling
    - Case analysis

15. **Memo operator**
    - Cache function bindings
    - Yield guard tracking
    - Obsolete guard management

16. **Dependent records**
    - Type-level dependencies

17. **Cycle handling**
    - Ensure termination in cyclic dataflow graphs
    - Convergence guarantees

### Deferred Challenges ⏳
18. **Streaming joins**
    - Incremental join execution as yield guards advance
    - Symmetric hash join or similar for true streaming
    - See design.md "Open Challenges" section

19. **Guard expression evaluation**
    - How to evaluate complex expressions in guards (e.g., `t2.fk = t1.pk + 1`)
    - Requires dataflow machinery, but guards configure dataflow - circular dependency
    - See design.md "Open Challenges" section

20. **Multi-level nesting optimization**
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
- [x] Literal tests (int, string)
- [x] Basic variable test
- [ ] Lambda tests
- [ ] Application tests
- [ ] ColumnValue and alignment tests
- [ ] Join execution tests
- [ ] Integration tests

## Next Steps (Immediate)
1. Implement `ColumnValue` struct and update `Producer::get()` return type
2. Refactor `Var` to remove static definition, add `VarSource` enum to `VarSub`
3. Add `innermost_scan` tracking to `VarScope`
4. Implement alignment logic in `VarRefSub::get()`
5. Implement `Application` operator (binds argument to lambda variable)
6. Implement basic Cartesian product join for scanning mode

## Questions / Open Issues
1. Should `Consumer::notify()` take `Guard` by reference instead of by value?
2. Memory leak concerns with Rc cycles - need to verify release guards prevent leaks
3. How does Application know to bind vs. let lambda scan? (Answer: Application always binds; aggregations like `sum` trigger scanning)
4. For streaming joins: when can we emit partial results? Need incremental join design.
5. For guard evaluation: how to handle guards with complex expressions that need dataflow to evaluate?

