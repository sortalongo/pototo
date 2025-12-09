//! PCL (Pototo Core Language) Interpreter
//!
//! This module implements the dataflow-based interpreter for PCL.
//! Execution proceeds via a producer/consumer protocol using guards and extents.

use std::collections::HashMap;

/// A Guard represents a region (subset of an extent) via a set of predicates.
/// Guards are used to:
/// - Specify intent (what region a consumer is interested in)
/// - Track yield (what region is ready and won't see further data)
/// - Track obsolescence (what region is no longer needed)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Guard {
    /// The universal guard representing the entire extent
    Universal,
    /// An empty guard representing no region
    Empty,
    /// A guard representing equality: variable == value
    Equality { variable: String, value: Value },
    /// A guard representing set membership: variable ∈ set
    Membership {
        variable: String,
        values: Vec<Value>,
    },
    /// A guard representing inequality: variable != value
    Inequality { variable: String, value: Value },
    /// A conjunction of guards (all must be satisfied)
    And(Vec<Guard>),
    /// A disjunction of guards (at least one must be satisfied)
    Or(Vec<Guard>),
    /// A guard for a function type: combines domain and codomain guards
    Function {
        domain: Box<Guard>,
        codomain: Box<Guard>,
    },
    /// A guard for a record type: maps field names to their guards
    Record(HashMap<String, Guard>),
}

impl Guard {
    /// Create an empty guard
    pub fn empty() -> Self {
        Guard::Empty
    }

    /// Create a universal guard
    pub fn universal() -> Self {
        Guard::Universal
    }

    /// Check if this guard is empty (represents no region)
    pub fn is_empty(&self) -> bool {
        matches!(self, Guard::Empty)
    }

    /// Check if this guard is universal (represents entire extent)
    pub fn is_universal(&self) -> bool {
        matches!(self, Guard::Universal)
    }

    /// Intersect two guards (conjunction)
    pub fn intersect(self, other: Guard) -> Guard {
        match (self, other) {
            (Guard::Empty, _) | (_, Guard::Empty) => Guard::Empty,
            (Guard::Universal, g) | (g, Guard::Universal) => g,
            (Guard::And(mut guards), g) => {
                guards.push(g);
                Guard::And(guards)
            }
            (g, Guard::And(mut guards)) => {
                guards.insert(0, g);
                Guard::And(guards)
            }
            (g1, g2) => Guard::And(vec![g1, g2]),
        }
    }

    /// Union two guards (disjunction)
    pub fn union(self, other: Guard) -> Guard {
        match (self, other) {
            (Guard::Empty, g) | (g, Guard::Empty) => g,
            (Guard::Universal, _) | (_, Guard::Universal) => Guard::Universal,
            (Guard::Or(mut guards), g) => {
                guards.push(g);
                Guard::Or(guards)
            }
            (g, Guard::Or(mut guards)) => {
                guards.insert(0, g);
                Guard::Or(guards)
            }
            (g1, g2) => Guard::Or(vec![g1, g2]),
        }
    }

    /// Split a function guard into domain and codomain guards
    pub fn split_function(&self) -> Option<(Guard, Guard)> {
        match self {
            Guard::Function { domain, codomain } => Some((*domain.clone(), *codomain.clone())),
            Guard::Universal => {
                // Universal function guard means universal domain and codomain
                Some((Guard::Universal, Guard::Universal))
            }
            _ => None,
        }
    }

    /// Split a record guard into field guards
    pub fn split_record(&self) -> Option<HashMap<String, Guard>> {
        match self {
            Guard::Record(fields) => Some(fields.clone()),
            Guard::Universal => {
                // Universal record guard means universal for all fields
                // This is a placeholder - in practice we'd need the record schema
                Some(HashMap::new())
            }
            _ => None,
        }
    }

    /// Create a function guard from domain and codomain guards
    pub fn from_function_parts(domain: Guard, codomain: Guard) -> Self {
        Guard::Function {
            domain: Box::new(domain),
            codomain: Box::new(codomain),
        }
    }

    /// Create a record guard from field guards
    pub fn from_record_parts(fields: HashMap<String, Guard>) -> Self {
        Guard::Record(fields)
    }
}

/// An Extent represents the set of values a term can take on (its type).
/// Each operator has an extent that corresponds exactly to its type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Extent {
    /// A base type (e.g., integer, string, boolean)
    Base(BaseType),
    /// A function type: domain -> codomain
    Function {
        domain: Box<Extent>,
        codomain: Box<Extent>,
    },
    /// A record type: map of field names to their extents
    Record(HashMap<String, Extent>),
    /// A union type: one of several possible extents
    Union(Vec<Extent>),
}

impl Extent {
    /// Create a function extent from domain and codomain
    pub fn function(domain: Extent, codomain: Extent) -> Self {
        Extent::Function {
            domain: Box::new(domain),
            codomain: Box::new(codomain),
        }
    }

    /// Create a record extent from field extents
    pub fn record(fields: HashMap<String, Extent>) -> Self {
        Extent::Record(fields)
    }

    /// Split a function extent into domain and codomain
    pub fn split_function(&self) -> Option<(&Extent, &Extent)> {
        match self {
            Extent::Function { domain, codomain } => Some((domain, codomain)),
            _ => None,
        }
    }

    /// Split a record extent into field extents
    pub fn split_record(&self) -> Option<&HashMap<String, Extent>> {
        match self {
            Extent::Record(fields) => Some(fields),
            _ => None,
        }
    }

    /// Create a guard from parts (for function types: domain + codomain guards)
    pub fn create_guard_from_parts(&self, parts: Vec<Guard>) -> Guard {
        match self {
            Extent::Function { .. } => {
                if parts.len() == 2 {
                    Guard::from_function_parts(parts[0].clone(), parts[1].clone())
                } else {
                    Guard::Universal
                }
            }
            Extent::Record(_) => {
                // For records, parts should be a map of field names to guards
                // This is a simplified version - in practice we'd need proper mapping
                Guard::Universal
            }
            _ => {
                if parts.len() == 1 {
                    parts[0].clone()
                } else {
                    Guard::Universal
                }
            }
        }
    }
}

/// Base types in PCL
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BaseType {
    Int,
    String,
    Bool,
    Unit,
}

/// Values in PCL
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(i64),
    String(String),
    Bool(bool),
    Unit,
    /// A function value (collection of bindings)
    Function(Vec<FunctionBinding>),
    /// A record value
    Record(HashMap<String, Value>),
}

/// A function binding represents a single input-output pair for a function
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionBinding {
    pub input: Value,
    pub output: Value,
}

// ============================================================================
// Producer/Consumer Protocol
// ============================================================================

/// A Consumer receives notifications when data is ready.
/// The consumer is notified by the producer with a yield guard indicating
/// what region is ready and won't see further data.
pub trait Consumer {
    /// Notify the consumer that data is ready.
    /// The `yield_guard` specifies a region that is ready and will not see
    /// any further data.
    // TODO: should we take Guard by ref?
    fn notify(&mut self, yield_guard: Guard);
}

/// Blanket implementation: Rc<RefCell<C>> implements Consumer when C does.
impl<C: Consumer> Consumer for Rc<RefCell<C>> {
    fn notify(&mut self, yield_guard: Guard) {
        self.borrow_mut().notify(yield_guard)
    }
}

/// A Producer provides data and handles release requests.
/// The producer is created by an operator's `subscribe` method and allows
/// the consumer to retrieve data and release regions.
pub trait Producer {
    /// Get the data that is ready.
    /// Returns the data structure representing the values in the ready region.
    /// The structure depends on the operator's type (records have fields,
    /// functions are collections, etc.).
    fn get(&mut self) -> Value;

    /// Release interest in a region.
    /// The `obsolete_guard` specifies a sub-region of the subscription that
    /// is no longer needed. Returns an expanded obsolete guard that may be
    /// larger if the producer has additional obsolescence information (e.g.,
    /// from variables with their own obsolete guards).
    fn release(&mut self, obsolete_guard: Guard) -> Guard;
}

/// Blanket implementation: Rc<RefCell<P>> implements Producer when P does.
impl<P: Producer> Producer for Rc<RefCell<P>> {
    fn get(&mut self) -> Value {
        self.borrow_mut().get()
    }

    fn release(&mut self, obsolete_guard: Guard) -> Guard {
        self.borrow_mut().release(obsolete_guard)
    }
}

/// A dataflow operator that can be subscribed to.
/// Operators implement this trait to provide a subscription interface.
/// The `subscribe` method takes an intent guard (specifying what region the
/// consumer is interested in) and a consumer, and returns a producer that
/// allows the consumer to get data and release regions.
pub trait Operator {
    /// Get the extent (type) of this operator.
    fn extent(&self) -> &Extent;

    /// Subscribe to this operator with an intent guard and consumer.
    /// Returns a producer that allows the consumer to get data and release regions.
    ///
    /// # Arguments
    /// * `intent_guard` - The region of the operator's extent that the consumer
    ///   is interested in
    /// * `consumer` - The consumer that will receive notifications when data is ready
    ///
    /// # Arguments
    /// * `var_scope` - The variable scope for looking up variables
    ///
    /// # Returns
    /// A producer that provides access to the data and allows releasing regions
    fn subscribe(
        &mut self,
        intent_guard: Guard,
        consumer: Box<dyn Consumer>,
        var_scope: Option<VarScope>,
    ) -> Box<dyn Producer>;
}

// ============================================================================
// Literal Operator
// ============================================================================

/// A literal operator represents a constant value.
/// According to the design: Subscribe calls Notify on the consumer immediately.
/// Notify calls Get. Get returns a constant. Release is a no-op.
pub struct Literal {
    value: Value,
    extent: Extent,
}

impl Literal {
    /// Create a new literal operator from a value.
    pub fn new(value: Value) -> Self {
        let extent = Self::extent_for_value(&value);
        Literal { value, extent }
    }

    /// Determine the extent for a given value.
    fn extent_for_value(value: &Value) -> Extent {
        match value {
            Value::Int(_) => Extent::Base(BaseType::Int),
            Value::String(_) => Extent::Base(BaseType::String),
            Value::Bool(_) => Extent::Base(BaseType::Bool),
            Value::Unit => Extent::Base(BaseType::Unit),
            Value::Function(bindings) => {
                // For a function literal, we need to infer the domain and codomain
                // from the bindings. For now, we'll use a simplified approach.
                // TODO: Properly infer function types from bindings
                if bindings.is_empty() {
                    Extent::function(Extent::Base(BaseType::Unit), Extent::Base(BaseType::Unit))
                } else {
                    // Infer from first binding as a placeholder
                    let domain = Self::extent_for_value(&bindings[0].input);
                    let codomain = Self::extent_for_value(&bindings[0].output);
                    Extent::function(domain, codomain)
                }
            }
            Value::Record(fields) => {
                let field_extents: HashMap<String, Extent> = fields
                    .iter()
                    .map(|(name, val)| (name.clone(), Self::extent_for_value(val)))
                    .collect();
                Extent::record(field_extents)
            }
        }
    }
}

impl Operator for Literal {
    fn extent(&self) -> &Extent {
        &self.extent
    }

    fn subscribe(
        &mut self,
        _intent_guard: Guard,
        mut consumer: Box<dyn Consumer>,
        _var_scope: Option<VarScope>,
    ) -> Box<dyn Producer> {
        consumer.notify(Guard::universal());

        Box::new(LiteralProducer {
            value: self.value.clone(),
        })
    }
}

struct LiteralProducer {
    value: Value,
}

impl Producer for LiteralProducer {
    fn get(&mut self) -> Value {
        self.value.clone()
    }

    fn release(&mut self, obsolete_guard: Guard) -> Guard {
        // Release is a no-op for literals - just return the obsolete guard unchanged
        obsolete_guard
    }
}

// ============================================================================
// Variable System
// ============================================================================

use std::cell::RefCell;
use std::rc::Rc;

/// Variable scope for looking up variables.
/// Variables are looked up by name, searching up the parent chain if not found.
pub struct VarScope {
    /// Optional parent scope (for nested scopes)
    parent: Option<Box<VarScope>>,
    /// Map of variable names to their subscriptions (shared state)
    variables: HashMap<String, Rc<RefCell<VariableSubscription>>>,
}

impl VarScope {
    /// Create a new empty scope.
    pub fn new() -> Self {
        VarScope {
            parent: None,
            variables: HashMap::new(),
        }
    }

    /// Create a new scope with a parent.
    pub fn with_parent(parent: VarScope) -> Self {
        VarScope {
            parent: Some(Box::new(parent)),
            variables: HashMap::new(),
        }
    }

    /// Add a variable to this scope.
    pub fn add_variable(&mut self, name: String, subscription: Rc<RefCell<VariableSubscription>>) {
        self.variables.insert(name, subscription);
    }

    /// Look up a variable by name, searching up the parent chain.
    /// Returns a reference to the subscription if found.
    pub fn lookup_variable(&self, name: &str) -> Option<&Rc<RefCell<VariableSubscription>>> {
        if let Some(subscription) = self.variables.get(name) {
            Some(subscription)
        } else if let Some(ref parent) = self.parent {
            parent.lookup_variable(name)
        } else {
            None
        }
    }
}

/// A Variable operator represents a variable definition.
/// It holds all statically-defined information: name, definition operator, extent, and predicate.
pub struct Variable {
    /// The name of the variable
    name: String,
    /// The operator that defines this variable
    definition: Box<dyn Operator>,
    /// The extent of this variable (may be restricted by predicates)
    extent: Extent,
    /// Predicate that restricts this variable's extent
    /// Applied to guards before propagating to the operator
    predicate: Guard,
}

impl Variable {
    /// Create a new variable operator.
    pub fn new(name: String, definition: Box<dyn Operator>) -> Self {
        let extent = definition.extent().clone();
        Variable {
            name,
            definition,
            extent,
            predicate: Guard::Universal,
        }
    }

    /// Set a predicate that restricts this variable's extent.
    /// The predicate is applied to guards before propagating to the operator.
    /// Use `Guard::Universal` to remove the predicate (no restriction).
    pub fn set_predicate(&mut self, predicate: Guard) {
        self.predicate = predicate;
    }

    /// Subscribe to this variable and return the VariableSubscription directly.
    /// This is useful when you need access to the subscription object (e.g., for adding to VarScope).
    /// The regular Operator::subscribe wraps this in Box<dyn Producer>.
    pub fn subscribe_to_var(
        &mut self,
        intent_guard: Guard,
        consumer: Box<dyn Consumer>,
        var_scope: Option<VarScope>,
    ) -> Rc<RefCell<VariableSubscription>> {
        // Apply predicate to intent guard
        let restricted_guard = intent_guard.intersect(self.predicate.clone());

        // Create VariableSubscription first (it will act as consumer for the definition)
        // TODO: does using RC for passing this struct both up and down the tree create a memory leak?
        let subscription = Rc::new(RefCell::new(VariableSubscription::new()));

        // Add the original consumer to the subscription's consumers vec
        subscription.borrow_mut().add_consumer(consumer);

        // TODO: what's the perf penalty of an RC in a box? It's not really necessary, just required by the type signature.
        let subscription_consumer: Box<dyn Consumer> = Box::new(subscription.clone());

        // Subscribe to definition, which will notify our subscription
        let definition_producer =
            self.definition
                .subscribe(restricted_guard, subscription_consumer, var_scope);

        // Store the definition producer in the subscription
        subscription
            .borrow_mut()
            .set_definition_producer(definition_producer);

        subscription
    }
}

impl Operator for Variable {
    fn extent(&self) -> &Extent {
        &self.extent
    }

    fn subscribe(
        &mut self,
        intent_guard: Guard,
        consumer: Box<dyn Consumer>,
        var_scope: Option<VarScope>,
    ) -> Box<dyn Producer> {
        // Use the specialized method and wrap the result
        let subscription = self.subscribe_to_var(intent_guard, consumer, var_scope);
        Box::new(subscription)
    }
}

/// VariableSubscription implements both Producer and Consumer.
/// It stores the yield guard (monotonically growing) and forwards notifications to all consumers.
pub struct VariableSubscription {
    /// The producer from the variable's definition
    definition_producer: Option<Box<dyn Producer>>,
    /// The current yield guard (monotonically growing)
    /// The contract of `notify` guarantees that guards are monotonically growing.
    yield_guard: Guard,
    /// All consumers that have subscribed to this variable
    consumers: Vec<Box<dyn Consumer>>,
    /// The stored release guard for use by variable references
    stored_release_guard: Guard,
}

impl VariableSubscription {
    fn new() -> Self {
        VariableSubscription {
            definition_producer: None,
            yield_guard: Guard::Empty,
            consumers: Vec::new(),
            stored_release_guard: Guard::Empty,
        }
    }

    /// Set the definition producer (called after subscribing to definition).
    fn set_definition_producer(&mut self, producer: Box<dyn Producer>) {
        self.definition_producer = Some(producer);
    }

    /// Add a consumer to be notified when yield guards arrive.
    /// If there's already a yield guard (data is ready), notify the new consumer immediately.
    fn add_consumer(&mut self, mut consumer: Box<dyn Consumer>) {
        // If data is already ready, notify the new consumer immediately
        if !self.yield_guard.is_empty() {
            consumer.notify(self.yield_guard.clone());
        }
        self.consumers.push(consumer);
    }

    /// Get the current yield guard.
    fn get_yield_guard(&self) -> Guard {
        self.yield_guard.clone()
    }

    /// Store a release guard.
    fn store_release_guard(&mut self, guard: Guard) {
        self.stored_release_guard = guard;
    }

    /// Get the stored release guard.
    fn get_stored_release_guard(&self) -> Guard {
        self.stored_release_guard.clone()
    }
}

impl Producer for VariableSubscription {
    fn get(&mut self) -> Value {
        self.definition_producer
            .as_mut()
            .expect("Definition producer should be set")
            .get()
    }

    fn release(&mut self, obsolete_guard: Guard) -> Guard {
        // Store the release guard for use by variable references
        self.store_release_guard(obsolete_guard.clone());
        // Forward release to definition
        self.definition_producer
            .as_mut()
            .expect("Definition producer should be set")
            .release(obsolete_guard)
    }
}

impl Consumer for VariableSubscription {
    /// Notify this subscription of a yield guard (called by definition).
    fn notify(&mut self, yield_guard: Guard) {
        self.yield_guard = yield_guard.clone();

        // Forward to all consumers
        let yield_guard = self.get_yield_guard();
        for consumer in self.consumers.iter_mut() {
            consumer.notify(yield_guard.clone());
        }
    }
}

/// A VariableRef operator represents a reference to a variable.
/// It holds the variable name and looks it up in the VarScope when subscribing.
pub struct VariableRef {
    /// The name of the variable being referenced
    name: String,
    /// The extent (cached from the variable when found)
    extent: Extent,
}

impl VariableRef {
    /// Create a new variable reference.
    pub fn new(name: String, extent: Extent) -> Self {
        VariableRef { name, extent }
    }
}

impl Operator for VariableRef {
    fn extent(&self) -> &Extent {
        &self.extent
    }

    fn subscribe(
        &mut self,
        intent_guard: Guard,
        consumer: Box<dyn Consumer>,
        var_scope: Option<VarScope>,
    ) -> Box<dyn Producer> {
        // Look up the variable in the scope
        let var_scope = var_scope.expect("VariableRef requires a VarScope");
        let variable_subscription = var_scope
            .lookup_variable(&self.name)
            .expect(&format!("Variable '{}' not found in scope", self.name))
            .clone();

        // Create VariableRefSubscription with the consumer stored
        let ref_subscription = Rc::new(RefCell::new(VariableRefSubscription {
            variable_subscription: variable_subscription.clone(),
            intent_guard,
            consumer,
        }));

        // Add the VariableRefSubscription as the consumer of the variable subscription
        let ref_subscription_consumer: Box<dyn Consumer> = Box::new(ref_subscription.clone());
        variable_subscription
            .borrow_mut()
            .add_consumer(ref_subscription_consumer);

        Box::new(ref_subscription) // As a producer.
    }
}

/// VariableRefSubscription implements both Producer and Consumer.
/// As a Consumer: it receives notifications from VariableSubscription, intersects
/// the yield guard with its intent guard, and forwards to the actual consumer.
/// As a Producer: it provides access to data and handles release requests.
struct VariableRefSubscription {
    /// Reference to the VariableSubscription
    variable_subscription: Rc<RefCell<VariableSubscription>>,
    /// The intent guard for this subscription
    intent_guard: Guard,
    /// The consumer of the variable ref that will receive filtered notifications
    consumer: Box<dyn Consumer>,
}

impl Consumer for VariableRefSubscription {
    /// Notify this subscription of a yield guard from the variable.
    fn notify(&mut self, yield_guard: Guard) {
        let restricted_guard = yield_guard.intersect(self.intent_guard.clone());
        self.consumer.notify(restricted_guard);
    }
}

impl Producer for VariableRefSubscription {
    fn get(&mut self) -> Value {
        // Get data from variable subscription
        let value = self.variable_subscription.borrow_mut().get();

        // TODO: Filter data based on intent guard
        // For now, return the full value
        value
    }

    fn release(&mut self, _obsolete_guard: Guard) -> Guard {
        // Return the stored release guard from the variable subscription
        self.variable_subscription
            .borrow()
            .get_stored_release_guard()
    }
}

// ============================================================================
// Lambda Operator
// ============================================================================

/// A no-op consumer that discards all notifications.
/// Used when we need to subscribe to an operator but don't care about its notifications.
struct NoOpConsumer;

impl Consumer for NoOpConsumer {
    fn notify(&mut self, _yield_guard: Guard) {
        // No-op: discard the notification
    }
}

/// A Lambda operator represents a lambda expression.
/// It has a variable and a body, and manages the variable scope.
pub struct Lambda {
    variable: Variable,
    body: Box<dyn Operator>,
    extent: Extent,
}

impl Lambda {
    pub fn new(variable: Variable, body: Box<dyn Operator>) -> Self {
        // Compute the extent: function type from domain (variable) to codomain (body)
        let domain = variable.extent().clone();
        let codomain = body.extent().clone();
        let extent = Extent::function(domain, codomain);
        Lambda {
            variable,
            body,
            extent,
        }
    }
}

impl Operator for Lambda {
    fn extent(&self) -> &Extent {
        &self.extent
    }

    fn subscribe(
        &mut self,
        intent_guard: Guard,
        consumer: Box<dyn Consumer>,
        var_scope: Option<VarScope>,
    ) -> Box<dyn Producer> {
        // Split intent guard into domain and codomain
        let (domain_guard, codomain_guard) = intent_guard
            .split_function()
            .unwrap_or((Guard::universal(), Guard::universal()));

        // Create a new VarScope first (before subscribing, since subscribe consumes var_scope)
        let mut new_scope = if let Some(parent) = var_scope {
            VarScope::with_parent(parent)
        } else {
            VarScope::new()
        };

        // Subscribe to the variable with the domain guard
        // Use a no-op consumer since the lambda doesn't do anything with notifications from its variable
        let variable_consumer: Box<dyn Consumer> = Box::new(NoOpConsumer);
        let subscription = self
            .variable
            .subscribe_to_var(domain_guard, variable_consumer, None);

        new_scope.add_variable(self.variable.name.clone(), subscription);

        // Subscribe to the body with the new scope and codomain guard
        self.body
            .subscribe(codomain_guard, consumer, Some(new_scope))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A test consumer that stores notifications in shared state.
    /// The notifications Vec is kept by the test, allowing access to notifications
    /// even after the consumer is moved into subscribe.
    /// Uses Rc<RefCell<>> for single-threaded, lock-free shared state.
    pub struct TestConsumer {
        notifications: Rc<RefCell<Vec<Guard>>>,
    }

    impl TestConsumer {
        /// Create a new TestConsumer and return both the consumer and the shared notifications Vec.
        /// The consumer can be moved into subscribe, while the notifications Vec allows
        /// reading notifications from outside.
        pub fn new() -> (Self, Rc<RefCell<Vec<Guard>>>) {
            let notifications = Rc::new(RefCell::new(Vec::new()));
            (
                TestConsumer {
                    notifications: notifications.clone(),
                },
                notifications,
            )
        }
    }

    impl Consumer for TestConsumer {
        fn notify(&mut self, yield_guard: Guard) {
            // Push the notification to the shared Vec
            self.notifications.borrow_mut().push(yield_guard);
        }
    }

    #[test]
    fn test_literal_int() {
        let mut literal = Literal::new(Value::Int(42));

        // Check extent
        assert_eq!(literal.extent(), &Extent::Base(BaseType::Int));

        // Create consumer with shared notifications Vec - keep the Vec reference
        let (consumer, notifications) = TestConsumer::new();
        let mut producer = literal.subscribe(Guard::universal(), Box::new(consumer), None);

        // The consumer should have been notified immediately
        // Now we can check the notification via the shared Vec
        let notifications_borrowed = notifications.borrow();
        assert_eq!(notifications_borrowed.len(), 1);
        assert_eq!(notifications_borrowed[0], Guard::universal());

        // Verify get returns the constant value
        let value = producer.get();
        assert_eq!(value, Value::Int(42));

        // Verify release is a no-op
        let released = producer.release(Guard::universal());
        assert_eq!(released, Guard::universal());
    }

    #[test]
    fn test_literal_string() {
        let mut literal = Literal::new(Value::String("hello".to_string()));

        assert_eq!(literal.extent(), &Extent::Base(BaseType::String));

        let (consumer, notifications) = TestConsumer::new();
        let mut producer = literal.subscribe(Guard::universal(), Box::new(consumer), None);

        // Verify we received the notification
        let notifications_borrowed = notifications.borrow();
        assert_eq!(notifications_borrowed.len(), 1);
        assert_eq!(notifications_borrowed[0], Guard::universal());

        let value = producer.get();
        assert_eq!(value, Value::String("hello".to_string()));
    }

    #[test]
    fn test_variable_proxy() {
        let literal = Literal::new(Value::Int(42));
        let mut variable = Variable::new("x".to_string(), Box::new(literal));
        let extent = variable.extent().clone();
        let mut var_ref = VariableRef::new("x".to_string(), extent);

        assert_eq!(var_ref.extent(), &Extent::Base(BaseType::Int));

        // Create a VarScope with the variable
        let mut var_scope = VarScope::new();
        // Subscribe to the variable and get its subscription directly
        let (var_consumer, _) = TestConsumer::new();
        let var_subscription =
            variable.subscribe_to_var(Guard::universal(), Box::new(var_consumer), None);
        var_scope.add_variable("x".to_string(), var_subscription);

        // Subscribe and verify it works
        let (consumer, notifications) = TestConsumer::new();
        let mut producer =
            var_ref.subscribe(Guard::universal(), Box::new(consumer), Some(var_scope));

        // Verify notification was received
        let notifications_borrowed = notifications.borrow();
        assert_eq!(notifications_borrowed.len(), 1);
        assert_eq!(notifications_borrowed[0], Guard::universal());

        // Verify get returns the value
        let value = producer.get();
        assert_eq!(value, Value::Int(42));

        // Verify release returns stored release guard (initially empty)
        let released = producer.release(Guard::universal());
        assert_eq!(released, Guard::Empty);
    }
}
