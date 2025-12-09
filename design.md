Pototo is a programming language that implements a new programming paradigm.
It abstracts over low level concepts like memory, threads, and connections, enabling programmers to focus on the logic of their program, non-functional requirements, and high-level architectural decisions.
The denotational semantics of the language is a pure, dependently-typed, functional language.
The operational semantics is very unlike the lambda calculus: rather than operating via term-wise beta reduction, function terms implement a producer/consumer interface which allows the runtime to implement streaming dataflow semantics, with pipelining, parallelization, and vectorization.
Progress is tracked by sending puntuations through this producer/consumer interface.


* Python syntax
    * Programs are written in a Pythonic syntax that uses for-comprehensions for the definition of collection-level logic. This Python syntax is lowered to the Pototo Core Language (PCL), where it is typechecked and interpreted.
    * Supported: functions, type annotations, assignment, augmented assignment, operators, assignment expressions, attributes, kwargs, ...
    * Likely need some customizations (defer for now):
        * Multi-line generator expressions, so generators don't always have to be wrapped in a def
        * A way to define records. Could reuse Python classes, but they're really imperative and full of boilerplate.
    * Need to lower to PCL. Will defer until after hashing out the fundamental pieces of PCL and the interpreter.
        * Retain location info through lowering
        * Big challenge is converting imperative loops into functional expressions. In particular, defining the indexing for these definitions is a big unknown.
        * Converting comprehensions & generators should be easier: just make them lambdas, with some additional indexing for each `yield`

* Pototo Core Language (PCL)
    * Should be really simple: literals, variables, records, unions, lambdas, let-binding, application, pattern matching, type annotations.
    * Will want a type checker, but can defer implementing it until later, and just make sure terms typecheck when experimenting.
    * To execute PCL, we convert each term in the AST into a dataflow operator and wire them up to execute.
        * The interpreter walks the AST, converting AST nodes into dataflow operators, and kicking off computation by subscribing to a dataflow operator when forced.
        * Each operator has an **extent**, the set of values the term can take on, and corresponding exactly to the term's type.
        * Execution proceeds by working in terms of **regions**, which are subsets of the extent: operators can ask for regions, inform about the availability of a region, and revoke their interest in a region.
        * A region is represented by a **guard** which is a data structure respresenting a set of predicates, such as equalities, set memberships, and inequalities.

    * Dataflow operators implement a producer/consumer protocol:
        * Subscribe(Guard, Consumer, VarScope): the consumer registers interest with the producer in a region of the producer's extent using an **intent guard**. The `VarScope` parameter provides variable context for operators that reference variables.
        * Notify(Guard): the producer notifies the consumer that data is ready for it to retrieve, along with a **yield guard** specifying a region that will not see any further data.
        * Get: the consumer requests the data that is ready from the producer and gets it synchronously. The data structure that represents data is determined by the operator's type: records have multiple fields, unions have multiple cases, and lambdas are collections. These types can be nested, and the data structure explicitly tracks that nesting. 
        * Release(Guard): the consumer retracts interest in a sub-region of its subscription with the producer in the form of an **obsolete guard**.

    * **Operator vs Runtime State**: There is an important distinction between operators and runtime state:
        * **Operators** are stateless and correspond to program syntax. They implement the `Operator` trait and include: `Literal`, `Variable`, `VariableRef`, `Lambda`, `Application`, etc. Each operator has an extent (its type) but no runtime state.
        * **Producers and Consumers** are runtime stateful objects created when `subscribe()` is called on an operator. They implement the `Producer` and `Consumer` traits respectively, and maintain all the runtime state needed for dataflow execution (yield guards, consumers lists, etc.).
        * When `Operator::subscribe()` is called, it creates the appropriate producer/consumer objects and wires them up according to the operator's semantics.

    * **Guard Monotonicity Contract**: The contract of `Consumer::notify()` guarantees that yield guards are **monotonically growing**. That is, each call to `notify()` must provide a yield guard that is a superset (or equal to) all previous yield guards for that consumer. This contract allows implementations to store a single yield guard rather than tracking all historical guards, and enables efficient guard management throughout the dataflow graph. 

# PCL operators:
## Literals
Subscribe calls Notify on the consumer immediately. Notify calls Get. Get returns a constant. Release is a no-op.

## Variables

Variables are split into operators and runtime state:

### Variable Operator
A `Variable` operator represents a variable definition. It holds:
    * The variable's name
    * The operator that defines this variable (its definition)
    * The extent of this variable (may be restricted by predicates)
    * A predicate guard that restricts the variable's extent (applied to guards before propagating to the definition)

The variable is owned and managed by the operator that defines it (record, lambda, let-binding, pattern match).

### VariableSubscription (Runtime State)
A `VariableSubscription` is created when `Variable::subscribe()` is called. It implements both `Producer` and `Consumer`:
    * As a **Consumer**: Receives notifications from the variable's definition, updates the yield guard (monotonically growing), and forwards notifications to all registered consumers
    * As a **Producer**: Provides data from the definition and handles release requests
    * Maintains a list of all consumers that have subscribed to this variable (multiple `VariableRef`s can reference the same variable)
    * Stores a release guard for use by variable references

When release is called on a function, its domain release guard is stored in the `VariableSubscription` for use within the function body.

### VariableRef Operator
A `VariableRef` operator represents a reference to a variable. It holds:
    * The name of the variable being referenced
    * The extent (cached from the variable when found)

### VariableRefSubscription (Runtime State)
A `VariableRefSubscription` is created when `VariableRef::subscribe()` is called. It implements `Producer`:
    * Filters data from the `VariableSubscription` based on its intent guard
    * When `release()` is called, returns the stored release guard from the `VariableSubscription` rather than invoking release on the subscription itself (since the lambda would have already invoked it)

### Variable Lookup: VarScope
Variables are looked up by name using a `VarScope` structure:
    * `VarScope` is a linked list structure that maps variable names to their `VariableSubscription` objects
    * Supports parent chaining for nested scopes (e.g., lambdas within lambdas)
    * When looking up a variable, the scope searches up the parent chain if not found in the current scope
    * `VarScope` is passed through `subscribe()` calls to enable variable lookup

### Variable System Flow

The variable system works as follows:

1. **Variable Definition**: When `Variable::subscribe()` is called:
    * Creates a `VariableSubscription` 
    * Adds the original consumer (from the subscribe call) to the subscription's consumers list
    * Uses the subscription itself (wrapped in `Rc<RefCell<>>`) as the consumer for the definition operator
    * Subscribes to the definition operator
    * Returns the subscription as a producer

2. **Variable Reference**: When `VariableRef::subscribe()` is called:
    * Looks up the variable name in the provided `VarScope` (searching up the parent chain if needed)
    * Adds itself as a consumer to the found `VariableSubscription`'s consumers list
    * Creates and returns a `VariableRefSubscription` that filters data based on the intent guard

3. **Notification Flow**: When the definition operator notifies:
    * Notification goes to `VariableSubscription::notify()`
    * The yield guard is updated 
    * All registered consumers (including all `VariableRefSubscription`s) are notified with the updated yield guard

4. **Data Access**: When a `VariableRefSubscription` calls `get()`:
    * It retrieves data from the `VariableSubscription`
    * Filters the data based on its intent guard 
    * Returns the filtered data

5. **Release Flow**: When a `VariableRefSubscription` calls `release()`:
    * Returns the stored release guard from the `VariableSubscription`
    * Does not propagate release to the definition (the lambda handles that)

### Quantification
Variables are quantified as either existential or universal. Let-bindings, records, and patterns are existentially quantified, which means they have a single definition. Lambda variables are universally quantified, which means they can be bound to many definitions. Quantification applies within each variable's scope (e.g., the part of the dataflow graph corresponding to the lambda's body). Additional bookkeeping within the dataflow graph is required to keep variables aligned as data flows through the graph (I think this can be orchestrated by the lambda operator).

For example, consider `sum(let one = 1 in λ row . let field = col1 row + 1 in one + (col2 row) + field)`. To evaluate the sum, we must iterate through the possible values of `row`. Inside of the lambda, the value of `one` remains constant as `row` and `field` change. As the dataflow operator for `one + (col2 row) + field` executes, something must ensure that `Get` on each of the variables in that expression return values that correspond to each other (specifically, `one` must return all `1`s as the other fields vary). When there are multiple universally-quantified variables in scope, the bookkeping must iterate through all valid combinations of those variables (i.e. perform a join). Predicates on variables are specified as proposition arguments to lambdas using dependent types.

## Records
A record is a map of field names to field definitions. Each field's definition is a dataflow operator. Subscribe splits the provided guard into a guard for each field, and calls subscribe on each corresponding field's operator. Notify is called by each field's operator when ready. Get zips together the data for the record's fields (TODO: how do we handle alignment when some fields of a record are ready, but others aren't? Maybe we only return when all subscribed fields are available). Release splits the obsolete guard and propagates the subguards to each field (TODO: how to handle correlated guards?).

## Application
Applies a function to an argument by resolving the argument and propagating it to the function. 

Subscribe receives an intent guard for the function's codomain. It uses the function's dep-rel to generate the depends-image of the codomain intent guard, and subscribes to the argument with the resulting domain intent guard. It then subscribes to the function with the combined domain/codomain intent guards (NOTE: the choice of eager/lazy evaluation is decided here by whether subscribe is invoked on the function before or after the argument notifies). 

Notify is forwarded from the argument to the function. Get gets from the argument and function, and returns the corresponding elements of the codomain. 

Release is implemented as the following pseudocode:
```
def release(g_c):
    g_d1 = f.pre(g_c) # generate an initial obsolete guard for the domain
    g_d2 = arg.release(g_d1) # release the argument, and get back an expanded obsolete guard for the domain.
    # g_d1 ⊂ g_d2 , but g_d2 can be larger if `arg` has a variable with its own obsolete guard
    g_fn = fn.release(fn.extent().create_from_parts(g_d2, g_c)) # release the function by combining the domain and codomain guards into a single unified guard
    # TODO: what if g_fn.split_domain() is a strict superset of g_d2? Would this process ever need to be iterated to guarantee release of all resources?
    return g_fn.split_codomain()
```
This approach allows obsolescence to flow bidirectionally within the lambda's body. 
It flows upstream from the top node of the body starting with the lambda's codomain obsolete guard. 
It flows downstream from the variables with the lambda's domain obsolete guard. 
These flows meet at application nodes, where obsolescence information comes from the function's codomain _and_ its domain (if the function argument uses a variable from an enclosing lambda).
To handle this, we first flow obsolescence upstream to the argument by generating an obsolence guard as the preimage of the codomain guard, which returns an updated domain obsolescence guard.
Then, we use that updated domain guard together with the original codomain guard to release the function.
Most of the time, this complex flow boils down to "get a domain guard from the argument and use it on the function" (i.e. unidirectional flow), but supporting this bidirectional flow is critical to ensure garbage collection doesn't leak resources as complex programs are composed together.


## Lambdas
A lambda is a (universally quantified) variable and a body, which can be applied to another term to replace the variable with that term. To accomplish this, the lambda operator manages a Variable. 

Subscribe splits its intent guard into domain and codomain intent guards, then calls subscribe on its variable and body with these guards. 

Notify comes from the variable, and is handled by the Variable. Get returns a collection of function bindings. 

Release splits the obsolete guard on domain and codomain, and calls release on the Variable and the body with the corresponding guard. The Variable stores its obsolete guard until the release call on the body returns so that variable references in the body can return expanded obsolete guards as needed.


## Memos
A memo is a function that stores its bindings so they don't have to be recomputed in the future. 
It takes a function as an argument and proxies all requests to/from it, but with additional logic to store, get, and release bindings.
The memo manages subscriptions to its argument in a way that guarantees that the same bindings are never computed twice. (TODO: would we want different kinds of memos, e.g. LRU memo with fixed size?)
Subscriptions to the memo keep track of which bindings are already stored in the memo (and can therefore notify), and which are still outstanding.
To implement this, the memo keeps a yield guard to track how much of extent is ready, as well as an obsolete guard to track which parts have expired. 

Subscribe first checks whether the current yield guard covers the intent guard. If it does not, it looks at the memo's subscription's intent guard to see if it can be reused. If not, it cancels that subscription and creates a new memo subscription for the union of the previous and new intent guards. It the creates and returns a new subscription for the given intent guard.

Notify:
* Calls Get immediately and stores all the returned bindings
* Calls Release with the yield guard to release its interest in that region of the extent
* Issues notify calls for all downstream subscriptions with overlapping intent guards

Get returns a handle to the data structure currently held in the memo, filtered to the subscription's intent guard. This data structure is indexed by the memo's domain, and can be filtered efficiently along that axis, or scanned fully.
No data is returned for regions that have been released by all subscriptions.

Release drops bindings for the given obsolete guard. It does not propagate upstream because memo releases data as soon as it is Notified.


## Let-bindings: 

## Unions, dependent records: TODO

## Pattern matching: 


# How to handle Cycles
Many of the above algorithms will not terminate if there's a cycle in the dataflow graph. How do we need to modify them to ensure termination? In which cases can we ensure convergence rather than simply truncating the iteration?


# PCL Proof of Concept
    

* PCL interpreter
    * Convert to dataflow operators and schedule.
    * Follow pull/push/progress protocol.
    * TODO: which cases can't be converted into dataflow?