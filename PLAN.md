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

### Producer/Consumer Protocol
- `Consumer::notify(yield_guard)` - Producer notifies consumer that data is ready
- `Producer::get()` - Consumer requests data synchronously
- `Producer::release(obsolete_guard)` - Consumer retracts interest in a region

### Key Design Decisions
1. **Blanket implementations**: `Rc<RefCell<P>>` implements `Producer` when `P: Producer`, and `Rc<RefCell<C>>` implements `Consumer` when `C: Consumer`
2. **Variable system**: Variables are split into `Var` (operator) and `VarSub` (runtime state)
3. **VarScope**: Linked list structure for variable lookup with parent chaining

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

### High Priority
1. **Complete Lambda::extent()**
   - Compute function extent from variable extent (domain) and body extent (codomain)
   - Store it in Lambda struct to avoid recomputation

2. **Implement data filtering in VarRefSub**
   - Filter data from VarSub based on intent guard
   - Currently returns full value

3. **Implement Application operator**
   - Handle function application with proper guard propagation
   - Implement bidirectional release flow as specified in design doc

4. **Fix Var::get_last_subscription() workaround**
   - Currently using a temporary storage mechanism
   - Should be replaced with proper architecture

### Medium Priority
5. **Records operator**
   - Split guards for field subscriptions
   - Zip data from fields
   - Handle alignment when some fields ready but others aren't

6. **Let-bindings**
   - Existentially quantified variables
   - Scope management

7. **Pattern matching**
   - Union handling
   - Case analysis

### Low Priority / Future
8. **Memo operator**
   - Cache function bindings
   - Yield guard tracking
   - Obsolete guard management

9. **Dependent records**
   - Type-level dependencies

10. **Cycle handling**
    - Ensure termination in cyclic dataflow graphs
    - Convergence guarantees

## Architecture Notes

### Variable System Flow
1. `Lambda::subscribe()` creates `VarSub` for lambda variable
2. Adds it to new `VarScope` with parent scope chained
3. Subscribes to body with new scope
4. When body contains `VarRef`, it looks up variable in scope
5. `VarRef::subscribe()` adds consumer to `VarSub`'s consumers vec
6. Returns `VarRefSub` that filters data

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
- [ ] Integration tests

## Next Steps (Immediate)
1. Fix `VarSub::notify()` to use union instead of replacement for yield guard
2. Implement `Lambda::extent()` properly
3. Add data filtering to `VarRefSub::get()`
4. Start Application operator implementation

## Questions / Open Issues
1. Should `Consumer::notify()` take `Guard` by reference instead of by value?
2. How to handle data filtering efficiently? Need to understand Value structure better
3. How to properly compute Lambda extent without storing it (or should we store it)?
4. Memory leak concerns with Rc cycles - need to verify release guards prevent leaks

