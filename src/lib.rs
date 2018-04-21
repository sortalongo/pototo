// Allows objects of the same type to be merged into a larger whole.
pub trait Merge {
    fn merge(self, other: Self) -> Self;
    fn merge_in_place(&mut self, other: Self);
}

// Allows code to compactly reason about sets of objects.
// For example, an implementation of PointSet with `Point = f64` and `Set = IntervalSet<f64>` would
// be sets of intervals of the real line.  Code using this interface can then work with infinite
// sets of f64s without representing them explicitly.
pub trait PointSet {
    // The point type.
    type Point: Copy;
    // The type of sets of Points.
    type Set;
    // The empty set.
    fn empty() -> Self::Set;
    // Tests whether an element is a member of the set.
    fn is_elem(p: &Self::Point, s: &Self::Set) -> bool;
    // Takes the union of two sets.
    fn union(s1: Self::Set, s2: Self::Set) -> Self::Set;
}

// Consumes messages, accumulating them until they are complete.
pub trait Consumer<T> {
    // The type of the points that are being consumed.
    type InP: Copy;
    type InPS: PointSet<Point = Self::InP>;
    // Put a message `t` into point k.
    fn put(&mut self, p: Self::InP, t: T);
    // Mark a region of the space `ks` as ready to be produced.
    fn advance(&mut self, s: <Self::InPS as PointSet>::Set);
}

// A bundle of produced points paired with the region to which these points correspond.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bundle<PS: PointSet, T> {
    pub punctuation: PS::Set,
    pub points: Vec<(PS::Point, T)>,
}
impl<PS: PointSet, T> Bundle<PS, T> {
    pub fn new(punc: PS::Set, points: Vec<(PS::Point, T)>) -> Bundle<PS, T> {
        Bundle {
            punctuation: punc,
            points: points,
        }
    }
}
// Produces messages on demand, forgetting them when receiving acks for them.
pub trait Producer<T> {
    // The type of points that are being produced.
    type OutP: Copy;
    type OutPS: PointSet<Point = Self::OutP>;
    // Produce a bundle of complete points. May be None even if some points are available. The
    // implementation is responsible for deciding when to produce.
    // ack_inline signals the implementation that it can immediately GC any points returned.
    fn get(&mut self, ack_inline: bool) -> Option<Bundle<Self::OutPS, T>>;
    // Acknowledge a region of points as received, allowing the producer to GC the region.
    fn ack(&mut self, acks: <Self::OutPS as PointSet>::Set);
}
pub trait Processor<I, O>: Consumer<I> + Producer<O> {}
pub trait Buffer<T: Merge>: Processor<T, T> {}

use std::fmt::Debug;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Unit();
impl PointSet for Unit {
    type Point = ();
    type Set = bool;
    fn empty() -> bool {
        false
    }
    fn is_elem(_u: &(), b: &bool) -> bool {
        *b
    }
    fn union(b1: bool, b2: bool) -> bool {
        b1 || b2
    }
}

use std::mem;

pub struct FusedBuffer<T> {
    elem: Option<T>,
    done: bool,
}
impl<T> FusedBuffer<T> {
    pub fn new(t: T) -> FusedBuffer<T> {
        FusedBuffer {
            elem: Some(t),
            done: false,
        }
    }
    pub fn empty() -> FusedBuffer<T> {
        FusedBuffer {
            elem: None,
            done: false,
        }
    }
    pub fn put(&mut self, t: T)
    where
        T: Merge,
    {
        if self.done {
            panic!("Trying to put into a finished FusedBuffer.");
        }
        let tmp = if self.elem.is_some() {
            let mut prev = None;
            mem::swap(&mut prev, &mut self.elem);
            prev.unwrap().merge(t)
        } else {
            t
        };
        mem::replace(&mut self.elem, Some(tmp));
    }
}
impl<T: Merge> Consumer<T> for FusedBuffer<T> {
    type InP = ();
    type InPS = Unit;
    fn put(&mut self, _k: (), t: T) {
        self.put(t);
    }
    fn advance(&mut self, ks: bool) {
        self.done = <Unit as PointSet>::union(self.done, ks);
    }
}
impl<T: Clone> Producer<T> for FusedBuffer<T> {
    type OutP = ();
    type OutPS = Unit;
    fn get(&mut self, ack_inline: bool) -> Option<Bundle<Unit, T>> {
        fn mk_bundle<T>(t: T) -> Bundle<Unit, T> {
            Bundle::new(true, vec![((), t)])
        }
        match (self.done, ack_inline) {
            (true, true) => mem::replace(&mut self.elem, None).map(|elem| mk_bundle(elem)),
            (true, false) => self.elem.as_ref().cloned().map(|elem| mk_bundle(elem)),
            _ => None,
        }
    }
    fn ack(&mut self, ack: bool) {
        if ack {
            mem::replace(&mut self.elem, None);
        }
    }
}
impl<T: Clone + Merge> Processor<T, T> for FusedBuffer<T> {}
impl<T: Clone + Merge> Buffer<T> for FusedBuffer<T> {}

impl Merge for i64 {
    fn merge(self, other: i64) -> i64 {
        self + other
    }
    fn merge_in_place(&mut self, other: i64) {
        *self += other;
    }
}
#[test]
fn fused_buffer_inserts_ack_inline() {
    let mut b = FusedBuffer::empty();
    assert_eq!(None, b.get(true));
    let i = 5;
    b.put(i);
    assert_eq!(None, b.get(true));
    b.advance(true);
    assert_eq!(Some(Bundle::new(true, vec![((), i)])), b.get(true));
    assert_eq!(None, b.get(true));
}
#[test]
fn fused_buffer_inserts_ack_later() {
    let mut b = FusedBuffer::empty();
    assert_eq!(None, b.get(false));
    let i = 5;
    b.put(i);
    assert_eq!(None, b.get(false));
    b.advance(true);
    assert_eq!(Some(Bundle::new(true, vec![((), i)])), b.get(false));
    assert_eq!(Some(Bundle::new(true, vec![((), i)])), b.get(false));
    b.ack(true);
    assert_eq!(None, b.get(false));
}

use std::marker::PhantomData;

pub struct PreFnProcessor<I, O, F, B>
where
    O: Merge,
    F: Fn(B::InP, I) -> O,
    B: Buffer<O>,
{
    f: F,
    buf: B,
    // Below are just markers for the compiler to track unused types.
    _in: PhantomData<I>,
    _out: PhantomData<O>,
}
impl<I, O: Merge, F, B> PreFnProcessor<I, O, F, B>
where
    F: Fn(B::InP, I) -> O,
    B: Buffer<O>,
{
    pub fn new(f: F, b: B) -> PreFnProcessor<I, O, F, B> {
        PreFnProcessor {
            _in: PhantomData,
            _out: PhantomData,
            f: f,
            buf: b,
        }
    }
}
impl<I, O: Merge, F, B> Consumer<I> for PreFnProcessor<I, O, F, B>
where
    F: Fn(B::InP, I) -> O,
    B: Buffer<O>,
{
    type InP = B::InP;
    type InPS = B::InPS;
    fn put(&mut self, p: Self::InP, input: I) {
        self.buf.put(p, (self.f)(p, input));
    }
    fn advance(&mut self, s: <Self::InPS as PointSet>::Set) {
        self.buf.advance(s);
    }
}
impl<I, O: Merge, F, B> Producer<O> for PreFnProcessor<I, O, F, B>
where
    F: Fn(B::InP, I) -> O,
    B: Buffer<O>,
{
    type OutP = B::OutP;
    type OutPS = B::OutPS;
    fn get(&mut self, ack_inline: bool) -> Option<Bundle<Self::OutPS, O>> {
        self.buf.get(ack_inline)
    }
    fn ack(&mut self, acks: <Self::OutPS as PointSet>::Set) {
        self.buf.ack(acks);
    }
}
impl<I: Merge, O: Merge, F, B> Processor<I, O> for PreFnProcessor<I, O, F, B>
where
    F: Fn(B::InP, I) -> O,
    B: Buffer<O>,
{
}

#[test]
fn prefn_processor_works_ack_inline() {
    let mut fn_b = PreFnProcessor::new(|_p, i| i * 2, FusedBuffer::empty());
    assert_eq!(None, fn_b.get(true));
    let i = 5;
    fn_b.put((), i);
    assert_eq!(None, fn_b.get(true));
    fn_b.advance(true);
    assert_eq!(Some(Bundle::new(true, vec![((), i * 2)])), fn_b.get(true));
    assert_eq!(None, fn_b.get(true));
}

#[test]
fn prefn_processor_works_ack_later() {
    let mut fn_b = PreFnProcessor::new(|_p, i| i * 2, FusedBuffer::empty());
    assert_eq!(None, fn_b.get(false));
    let i = 5;
    fn_b.put((), i);
    assert_eq!(None, fn_b.get(false));
    fn_b.advance(true);
    assert_eq!(Some(Bundle::new(true, vec![((), i * 2)])), fn_b.get(false));
    assert_eq!(Some(Bundle::new(true, vec![((), i * 2)])), fn_b.get(false));
    fn_b.ack(true);
    assert_eq!(None, fn_b.get(false));
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ZeroToN;

impl PointSet for ZeroToN {
    type Point = usize;
    type Set = usize;
    fn empty() -> usize {
        0
    }
    fn is_elem(u: &usize, us: &usize) -> bool {
        u < us
    }
    fn union(us1: usize, us2: usize) -> usize {
        std::cmp::max(us1, us2)
    }
}

use std::collections::VecDeque;

pub struct LinearBuf<T> {
    deque: VecDeque<FusedBuffer<T>>,
    // Which index of the buffer is represented by deque[0].
    buffer_min: usize,
    // Lowest index of the buffer that is incomplete.
    incomplete_min: usize,
}
impl<T> LinearBuf<T> {
    pub fn new() -> LinearBuf<T> {
        LinearBuf {
            deque: VecDeque::new(),
            buffer_min: 0,
            incomplete_min: 0,
        }
    }
    fn summary(&self) -> String {
        format!(
            "LinearBuf {{ deque (len {}), buffer_min: {}, incomplete_min: {} }}",
            self.deque.len(),
            self.buffer_min,
            self.incomplete_min
        )
    }
    fn drain(&mut self, pending: usize) -> Vec<(usize, T)> {
        assert!(
            self.buffer_min + pending <= self.incomplete_min,
            "Draining past completion boundary: {}.",
            self.summary()
        );
        let old_buffer_min = self.buffer_min;
        self.buffer_min += pending;
        self.deque
            .drain(0..pending)
            .filter(|fb| fb.elem.is_some())
            .enumerate()
            .map(|(i, fb)| (old_buffer_min + i, fb.elem.unwrap()))
            .collect()
    }
}

impl<T: Merge + Debug> Consumer<T> for LinearBuf<T> {
    type InP = usize;
    type InPS = ZeroToN;
    fn put(&mut self, k: usize, t: T) {
        assert!(k >= self.buffer_min);
        assert!(k >= self.incomplete_min);
        let idx = k - self.buffer_min;
        let len = self.deque.len();
        if len > idx {
            assert!(
                idx < self.deque.len(),
                "Failed getting index {} from linear buf {}.",
                idx,
                self.summary()
            );
            self.deque.get_mut(idx).unwrap().put(t);
        } else {
            self.deque.reserve(idx - len + 1);
            for _j in len..idx {
                self.deque.push_back(FusedBuffer::empty());
            }
            self.deque.push_back(FusedBuffer::new(t));
        }
    }
    fn advance(&mut self, ks: usize) {
        self.incomplete_min = Self::InPS::union(self.incomplete_min, ks);
    }
}
impl<T: Clone> Producer<T> for LinearBuf<T> {
    type OutP = usize;
    type OutPS = ZeroToN;
    fn get(&mut self, ack_inline: bool) -> Option<Bundle<ZeroToN, T>> {
        let pending = self.incomplete_min as i64 - self.buffer_min as i64;
        assert!(pending >= 0);
        let pending = pending as usize;
        if pending == 0 {
            return None;
        }
        let points = if ack_inline {
            self.drain(pending)
        } else {
            self.deque
                .iter()
                .take(pending)
                .filter(|&fb| fb.elem.is_some())
                .enumerate()
                .map(|(i, fb)| (self.buffer_min + i, fb.elem.as_ref().cloned().unwrap()))
                .collect()
        };
        Some(Bundle::new(self.incomplete_min, points))
    }
    fn ack(&mut self, acks: usize) {
        let to_gc = acks as i64 - self.buffer_min as i64;
        if to_gc > 0 {
            self.drain(to_gc as usize);
        }
    }
}
impl<T: Merge + Clone + Debug> Processor<T, T> for LinearBuf<T> {}
impl<T: Merge + Clone + Debug> Buffer<T> for LinearBuf<T> {}

impl Merge for String {
    fn merge(self, other: String) -> String {
        self + &other
    }
    fn merge_in_place(&mut self, other: String) {
        *self += &other;
    }
}

#[test]
fn linear_buf_works() {
    let mut buf = LinearBuf::<String>::new();
    buf.put(1, "1".to_string());
    buf.put(2, "2".to_string());
    buf.put(0, "0".to_string());
    assert_eq!(None, buf.get(true));
    buf.advance(1);
    assert_eq!(
        Some(Bundle::new(1, vec![(0, "0".to_string())])),
        buf.get(true)
    );
    assert_eq!(None, buf.get(true));
    buf.advance(3);
    assert_eq!(
        Some(Bundle::new(
            3,
            vec![(1, "1".to_string()), (2, "2".to_string())]
        )),
        buf.get(true)
    );
    assert_eq!(None, buf.get(true));
}

use std::cmp::Ord;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::Bound;
use std::hash::Hash;

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Boundary<K> {
    Open(K),
    Closed(K),
}
impl<K> Boundary<K> {
    pub fn value(&self) -> &K {
        match self {
            &Boundary::Open(ref k) => k,
            &Boundary::Closed(ref k) => k,
        }
    }
    pub fn is_open(&self) -> bool {
        match self {
            &Boundary::Open(ref _k) => true,
            &Boundary::Closed(ref _k) => false,
        }
    }
    pub fn inverted(self) -> Self {
        match self {
            Boundary::Open(k) => Boundary::Closed(k),
            Boundary::Closed(k) => Boundary::Open(k),
        }
    }
    pub fn to_bound(&self) -> Bound<&K> {
        match self {
            &Boundary::Open(ref k) => Bound::Excluded(k),
            &Boundary::Closed(ref k) => Bound::Included(k),
        }
    }
}
impl<K: PartialOrd> PartialOrd for Boundary<K> {
    fn partial_cmp(&self, other: &Boundary<K>) -> Option<Ordering> {
        let same_boundary_type = self.is_open() == other.is_open();
        self.value()
            .partial_cmp(other.value())
            .and_then(|ord| match ord {
                Ordering::Equal => if same_boundary_type {
                    Some(ord)
                } else {
                    None
                },
                o => Some(o),
            })
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Interval<K: Ord> {
    lower: Boundary<K>,
    upper: Boundary<K>,
}
impl<K> Interval<K>
where
    K: Ord + Clone + Debug,
{
    fn check_order(lower: &Boundary<K>, upper: &Boundary<K>) {
        if lower.value() > upper.value() {
            panic!("Interval: {:?} must be <= {:?}.", lower, upper);
        } else if lower.value() == upper.value() && lower.is_open() != upper.is_open() {
            panic!(
                "Interval: Constructing illegal half-open interval with equal endpoints {:?}.",
                lower.value()
            );
        }
    }

    pub fn empty() -> Interval<K>
    where
        K: Default,
    {
        Interval::open(Default::default(), Default::default())
    }
    pub fn singleton(k: K) -> Interval<K> {
        Interval::closed(k.clone(), k)
    }

    pub fn of(lower: Boundary<K>, upper: Boundary<K>) -> Interval<K> {
        Interval::check_order(&lower, &upper);
        Interval {
            lower: lower,
            upper: upper,
        }
    }

    pub fn open(lower: K, upper: K) -> Interval<K> {
        Interval::of(Boundary::Open(lower), Boundary::Open(upper))
    }
    pub fn closed(lower: K, upper: K) -> Interval<K> {
        Interval::of(Boundary::Closed(lower), Boundary::Closed(upper))
    }
    pub fn closed_open(lower: K, upper: K) -> Interval<K> {
        Interval::of(Boundary::Closed(lower), Boundary::Open(upper))
    }
    pub fn open_closed(lower: K, upper: K) -> Interval<K> {
        Interval::of(Boundary::Open(lower), Boundary::Closed(upper))
    }

    pub fn is_empty(&self) -> bool {
        self.upper.value() == self.lower.value() && self.upper.is_open() && self.lower.is_open()
    }

    pub fn to_range(&self) -> (Bound<&K>, Bound<&K>) {
        (self.lower.to_bound(), self.upper.to_bound())
    }

    pub fn overlaps_with(&self, other: &Interval<K>) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        let is_self_lower = self.lower.value() < other.lower.value();
        let (smaller, larger) = if is_self_lower {
            (self, other)
        } else {
            (other, self)
        };
        let is_range_overlap = smaller.upper.value() > larger.lower.value();
        let is_point_overlap = smaller.upper.value() == larger.lower.value()
            && (!smaller.upper.is_open() || !larger.lower.is_open());
        is_range_overlap || is_point_overlap
    }

    pub fn merge_overlapping(self, other: Interval<K>) -> Interval<K> {
        debug_assert!(self.overlaps_with(&other));
        let is_self_lower = self.lower.value() < other.lower.value()
            || (self.lower.value() == other.lower.value() && !self.lower.is_open());
        let is_self_upper = self.upper.value() > other.upper.value()
            || (self.upper.value() == other.upper.value() && !self.upper.is_open());
        match (is_self_lower, is_self_upper) {
            (true, true) => self,
            (true, false) => Interval::of(self.lower, other.upper),
            (false, false) => other,
            (false, true) => Interval::of(other.lower, self.upper),
        }
    }
}
impl<K: Ord + Clone + Debug> PartialOrd for Interval<K> {
    fn partial_cmp(&self, other: &Interval<K>) -> Option<Ordering> {
        if self.overlaps_with(other) {
            None
        } else {
            self.lower.value().partial_cmp(other.lower.value())
        }
    }
}

#[test]
fn overlaps_with_works() {
    assert!(Interval::closed(0, 2).overlaps_with(&Interval::closed(1, 4)));
    assert!(Interval::closed(0, 2).overlaps_with(&Interval::closed(2, 4)));
    assert!(!Interval::closed(0, 2).overlaps_with(&Interval::closed(4, 6)));
    assert!(Interval::closed_open(0, 2).overlaps_with(&Interval::closed_open(2, 4)));
    assert!(Interval::open_closed(0, 2).overlaps_with(&Interval::open_closed(2, 4)));
    assert!(!Interval::open(0, 2).overlaps_with(&Interval::open(2, 4)));
    assert!(!Interval::closed_open(0, 2).overlaps_with(&Interval::open_closed(2, 4)));
    assert!(Interval::open_closed(0, 2).overlaps_with(&Interval::closed_open(2, 4)));
    assert!(Interval::open(0, 2).overlaps_with(&Interval::open(0, 4)));
}

use std::collections::HashSet;

// A private interval type that implements a total ordering by ordering by lower bound first,
// with Closed < Open, then by upper bound, with Open < Closed.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct _Interval<K: Ord + Clone + Debug>(Interval<K>);
impl<K: Ord + Clone + Debug> PartialOrd for _Interval<K> {
    fn partial_cmp(&self, other: &_Interval<K>) -> Option<Ordering> {
        fn boundary_cmp<K>(l: &Boundary<K>, r: &Boundary<K>) -> Ordering {
            match (l, r) {
                (&Boundary::Open(_), &Boundary::Closed(_)) => Ordering::Greater,
                (&Boundary::Closed(_), &Boundary::Open(_)) => Ordering::Less,
                _ => Ordering::Equal,
            }
        }
        let left_ord = self.0
            .lower
            .partial_cmp(&other.0.lower)
            .unwrap_or(Ordering::Equal)
            .then(boundary_cmp(&self.0.lower, &other.0.lower));
        let right_ord = self.0
            .upper
            .partial_cmp(&other.0.upper)
            .unwrap_or(Ordering::Equal)
            .then(boundary_cmp(&self.0.upper, &other.0.upper).reverse());
        Some(left_ord.then(right_ord))
    }
}
impl<K: Ord + Clone + Debug> Ord for _Interval<K> {
    fn cmp(&self, other: &_Interval<K>) -> Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntervalSet<K>
where
    K: Clone + Debug + Ord,
{
    // IntervalSet doesn't allow overlapping intervals, so we don't need an interval tree for
    // efficiency.
    intervals: BTreeSet<_Interval<K>>,
}
impl<K> IntervalSet<K>
where
    K: Clone + Debug + Hash + Ord,
{
    pub fn empty() -> IntervalSet<K> {
        IntervalSet {
            intervals: BTreeSet::new(),
        }
    }
    pub fn singleton(i: Interval<K>) -> IntervalSet<K> {
        let mut s = BTreeSet::new();
        s.insert(_Interval(i));
        IntervalSet { intervals: s }
    }
    pub fn of(mut intvls: Vec<Interval<K>>) -> IntervalSet<K> {
        intvls
            .drain(..)
            .fold(IntervalSet::empty(), |mut iset, intvl| {
                iset.insert(intvl);
                iset
            })
    }

    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Interval<K>> {
        self.intervals.iter().map(|i| &i.0)
    }

    pub fn contains_point(&self, k: K) -> bool {
        let k_intvl = Interval::singleton(k);
        self.intervals
            .range(.._Interval(k_intvl.clone()))
            .next_back()
            .map(|i| i.0.overlaps_with(&k_intvl))
            .unwrap_or(false)
    }

    pub fn contains_interval(&self, interval: Interval<K>) -> bool {
        self.intervals.contains(&_Interval(interval))
    }

    pub fn copy_to_vec(&self) -> Vec<Interval<K>> {
        self.intervals.iter().map(|i| &i.0).cloned().collect()
    }

    // Inserts an interval, merging with any overlapping it.
    pub fn insert(&mut self, intvl: Interval<K>) {
        let mut to_remove: HashSet<_Interval<K>> = {
            let lower_singleton = _Interval(Interval::singleton(intvl.lower.value().clone()));
            let upper_singleton = _Interval(Interval::singleton(intvl.upper.value().clone()));
            let prev_intvls = self.intervals
                .range(..upper_singleton)
                .rev()
                .take_while(|prev| intvl.overlaps_with(&prev.0))
                .cloned();
            let next_intvls = self.intervals
                .range(lower_singleton..)
                .take_while(|next| intvl.overlaps_with(&next.0))
                .cloned();
            prev_intvls.chain(next_intvls).collect()
        };

        println!("Insert: to_remove={:?}", to_remove);
        for overlapping_int in to_remove.iter() {
            assert!(self.intervals.remove(overlapping_int));
        }
        let all_merged = to_remove
            .drain()
            .fold(intvl, |merged_intvl, overlapping_intvl| {
                merged_intvl.merge_overlapping(overlapping_intvl.0)
            });
        println!("Insert: inserting={:?}", &all_merged);
        self.intervals.insert(_Interval(all_merged));
    }

    pub fn remove(&mut self, interval: &Interval<K>) -> bool {
        self.intervals.remove(&_Interval(interval.clone()))
    }

    // Returns the union of two IntervalSets. The returned IntervalSet:
    // 1. Contains no overlapping intervals.
    // 2. Contains only intervals overlapping with at least one interval in the inputs.
    // 3. Contains no intervals containing regions not overlapping any intervals in the inputs.
    pub fn union(self, other: IntervalSet<K>) -> IntervalSet<K> {
        let (smaller, mut larger) = if self.intervals.len() > other.intervals.len() {
            (other, self)
        } else {
            (self, other)
        };
        for int in smaller.intervals {
            larger.insert(int.0);
        }
        larger
    }
}

#[test]
fn interval_set_inserts() {
    let mut intvs = IntervalSet::empty();
    let intv1 = Interval::open(0, 2);
    intvs.insert(intv1);
    let intv2 = Interval::open(5, 7);
    intvs.insert(intv2);
    assert_eq!(intvs.intervals.len(), 2, "Set: {:?}", intvs);

    let intv3 = Interval::closed(2, 4);
    intvs.insert(intv3);
    assert_eq!(intvs.intervals.len(), 2, "Set: {:?}", intvs);

    let intv4 = Interval::closed(2, 5);
    intvs.insert(intv4);
    assert_eq!(intvs.intervals.len(), 1, "Set: {:?}", intvs);

    let mut ordered_intvs = vec![intv3, intv4, intv2];
    let expected = IntervalSet::singleton(
        ordered_intvs
            .drain(..)
            .fold(intv1, |merged, next| merged.merge_overlapping(next)),
    );
    assert_eq!(intvs, expected);
}

#[test]
fn interval_set_unions_merges() {
    let iset1 = IntervalSet::of(vec![Interval::closed_open(0, 2)]);
    let iset2 = IntervalSet::of(vec![Interval::closed_open(2, 4)]);
    let union = iset1.union(iset2);
    assert_eq!(union.copy_to_vec(), vec![Interval::closed_open(0, 4)]);
}

#[test]
fn interval_set_unions_merges_surrounded() {
    let iset1 = IntervalSet::of(vec![Interval::open(0, 2), Interval::open(4, 6)]);
    let iset2 = IntervalSet::of(vec![Interval::closed(2, 4)]);
    let union = iset1.union(iset2);
    assert_eq!(union.copy_to_vec(), vec![Interval::open(0, 6)]);
}

#[test]
fn interval_set_unions_enclosed() {
    let iset1 = IntervalSet::of(vec![Interval::open(0, 2), Interval::open(4, 6)]);
    let iset2 = IntervalSet::of(vec![Interval::closed(-1, 7)]);
    let union = iset1.union(iset2);
    assert_eq!(union.copy_to_vec(), vec![Interval::closed(-1, 7)]);
}
#[test]
fn interval_set_unions_squashed() {
    let iset1 = IntervalSet::of(vec![Interval::closed(0, 2)]);
    let iset2 = IntervalSet::of(vec![Interval::open(2, 4)]);
    let union = iset1.union(iset2);
    assert_eq!(union.copy_to_vec(), vec![Interval::closed_open(0, 4)]);
}
#[test]
fn interval_set_unions_singleton() {
    let neg = _Interval(Interval::singleton(-1));
    let pos = _Interval(Interval::open(2, 4));
    let mut iset1 = IntervalSet::of(vec![Interval::open(2, 4)]);
    assert_eq!(neg.cmp(&pos), Ordering::Less);
    assert_ne!(neg, pos);
    println!("iset1: {:?}", &iset1);
    iset1.insert(Interval::singleton(-1));
    assert_eq!(
        iset1.copy_to_vec(),
        vec![Interval::singleton(-1), Interval::open(2, 4)]
    );
}

#[test]
fn interval_set_unions_assorted() {
    let iset1 = IntervalSet::of(vec![
        Interval::open(2, 4),
        Interval::singleton(-1),
        Interval::open(2, 5),
    ]);
    let iset2 = IntervalSet::of(vec![
        Interval::closed(0, 2),
        Interval::open(5, 7),
        Interval::closed_open(-5, -1),
    ]);
    let union = iset1.union(iset2);
    assert_eq!(
        union.copy_to_vec(),
        vec![
            Interval::closed(-5, -1),
            Interval::closed_open(0, 5),
            Interval::open(5, 7),
        ]
    );
}

// The carrier type for intervals for any totally-orderd P.
// Implements a PointSet with Interval<P> as the Point type and IntervalSet<P> as the Set type.
#[derive(Debug, Eq, PartialEq)]
pub struct Intervals<P: Ord> {
    _k: PhantomData<P>,
}

impl<P> PointSet for Intervals<P>
where
    P: Ord + Clone + Copy + Debug + Default + Hash,
{
    type Point = P;
    type Set = IntervalSet<P>;
    fn empty() -> IntervalSet<P> {
        IntervalSet::empty()
    }
    fn is_elem(p: &P, intvl_set: &IntervalSet<P>) -> bool {
        intvl_set.contains_point(p.clone())
    }
    fn union(ints1: IntervalSet<P>, ints2: IntervalSet<P>) -> IntervalSet<P> {
        ints1.union(ints2)
    }
}

pub struct ParBuf<P, V>
where
    P: Clone + Debug + Hash + Ord,
    Intervals<P>: PointSet,
{
    points: BTreeMap<P, V>,
    completed: IntervalSet<P>,
    acked: IntervalSet<P>,
}

impl<P, V> ParBuf<P, V>
where
    P: Clone + Debug + Hash + Ord,
    Intervals<P>: PointSet,
{
    pub fn empty() -> ParBuf<P, V> {
        ParBuf {
            points: BTreeMap::new(),
            completed: IntervalSet::empty(),
            acked: IntervalSet::empty(),
        }
    }
}

impl<P, V> Consumer<V> for ParBuf<P, V>
where
    P: Copy + Clone + Debug + Default + Hash + Ord,
    V: Debug + Merge,
{
    type InP = P;
    type InPS = Intervals<P>;

    fn put(&mut self, p: P, v: V) {
        assert!(!self.completed.contains_point(p));
        assert!(!self.acked.contains_point(p));
        println!("putting: {:?}", &v);
        let v_prev_opt = self.points.insert(p, v);
        println!("put, prev: {:?}", &v_prev_opt);
        match v_prev_opt {
            Some(v_prev) => self.points.get_mut(&p).unwrap().merge_in_place(v_prev),
            _ => (),
        }
    }

    fn advance(&mut self, s: IntervalSet<P>) {
        let moved = mem::replace(&mut self.completed, Intervals::empty());
        self.completed = Intervals::union(moved, s);
    }
}

impl<P, V> Producer<V> for ParBuf<P, V>
where
    P: Copy + Clone + Debug + Default + Hash + Ord,
    V: Clone + Merge,
{
    type OutP = P;
    type OutPS = Intervals<P>;

    fn get(&mut self, ack_inline: bool) -> Option<Bundle<Self::OutPS, V>> {
        if self.completed.len() == 0 {
            return None;
        }
        let intervals = if ack_inline {
            mem::replace(&mut self.completed, Intervals::empty())
        } else {
            self.completed.clone()
        };
        let pts = intervals
            .iter()
            .cloned()
            .flat_map(|interval| {
                let complete_points: Vec<(P, V)> = self.points
                    .range(interval.to_range())
                    .map(|(p, v)| (*p, v.clone()))
                    .collect();
                if ack_inline {
                    self.acked.insert(interval);
                }
                for &(ref pt, _) in complete_points.iter() {
                    if ack_inline {
                        self.points.remove(pt);
                    }
                }
                complete_points
            })
            .collect();
        Some(Bundle::new(intervals, pts))
    }

    fn ack(&mut self, acks: <Self::OutPS as PointSet>::Set) {
        for ack in acks.iter() {
            assert!(self.completed.contains_interval(ack.clone()));
            assert!(self.completed.remove(&ack));
            let acked_points: Vec<P> = self.points.range(ack.to_range()).map(|(p, _)| *p).collect();
            for pt in acked_points {
                assert!(self.points.remove(&pt).is_some());
            }
            self.acked.insert(ack.clone());
        }
    }
}

impl<P, V> Processor<V, V> for ParBuf<P, V>
where
    P: Copy + Clone + Debug + Default + Hash + Ord,
    V: Merge + Clone + Debug,
{
}

impl<P, V> Buffer<V> for ParBuf<P, V>
where
    P: Copy + Clone + Debug + Default + Hash + Ord,
    V: Merge + Clone + Debug,
{
}

fn make_par_buf<P, V>(pairs: Vec<(P, V)>) -> ParBuf<P, V>
where
    P: Copy + Clone + Debug + Default + Hash + Ord,
    V: Debug + Merge,
{
    let mut buf = ParBuf::empty();
    for pair in pairs {
        buf.put(pair.0, pair.1);
    }
    buf
}
#[test]
fn par_buf_puts() {
    let buf = make_par_buf(vec![("a", 1), ("a", 2), ("b", 3)]);
    assert_eq!(buf.points.len(), 2);
}

#[test]
fn par_buf_advances() {
    let mut buf = make_par_buf(vec![("a", 1), ("a", 2), ("b", 3)]);
    let intervals = IntervalSet::singleton(Interval::closed_open("a", "b"));
    buf.advance(intervals.clone());
    assert_eq!(
        buf.get(true),
        Some(Bundle::new(intervals.clone(), vec![("a", 3)]))
    );
    assert_eq!(buf.get(true), None);

    let intervals = IntervalSet::singleton(Interval::closed_open("b", "c"));
    buf.advance(intervals.clone());
    assert_eq!(
        buf.get(true),
        Some(Bundle::new(intervals.clone(), vec![("b", 3)]))
    );
    assert_eq!(buf.get(true), None);

    assert_eq!(buf.completed.len(), 0);
    assert_eq!(buf.acked.len(), 1);
}

#[test]
fn par_buf_acks() {
    let mut buf = make_par_buf(vec![("a", 1), ("a", 2), ("b", 3)]);
    let intervals = IntervalSet::singleton(Interval::closed("a", "b"));
    buf.advance(intervals.clone());
    assert_eq!(
        buf.get(false),
        Some(Bundle::new(intervals.clone(), vec![("a", 3), ("b", 3)]))
    );

    buf.ack(intervals.clone());
    assert_eq!(buf.completed.len(), 0);
    assert_eq!(buf.acked.len(), 1);
}

struct Transform<I, O, InB, OutB, Map, Image>
where
    I: Merge,
    O: Merge,
    InB: Buffer<I>,
    OutB: Buffer<O>,
    Map: Fn(<InB as Producer<I>>::OutP, I) -> (<OutB as Consumer<O>>::InP, O),
    Image: Fn(<<InB as Producer<I>>::OutPS as PointSet>::Set)
        -> <<OutB as Consumer<O>>::InPS as PointSet>::Set,
{
    in_buf: InB,
    out_buf: OutB,
    map: Map,
    image: Image,
    _i: PhantomData<I>,
    _o: PhantomData<O>,
}

impl<I, O, InB, OutB, Map, Image> Producer<O> for Transform<I, O, InB, OutB, Map, Image>
where
    I: Merge,
    O: Merge,
    InB: Buffer<I>,
    OutB: Buffer<O>,
    Map: Fn(<InB as Producer<I>>::OutP, I) -> (<OutB as Consumer<O>>::InP, O),
    Image: Fn(<<InB as Producer<I>>::OutPS as PointSet>::Set)
        -> <<OutB as Consumer<O>>::InPS as PointSet>::Set,
{
    type OutP = <<OutB as Producer<O>>::OutPS as PointSet>::Point;
    type OutPS = <OutB as Producer<O>>::OutPS;
    fn get(&mut self, ack_inline: bool) -> Option<Bundle<Self::OutPS, O>> {
        (&mut self.out_buf).get(ack_inline)
    }
    fn ack(&mut self, acks: <Self::OutPS as PointSet>::Set) {
        (&mut self.out_buf).ack(acks);
    }
}

impl<I, O, InB, OutB, Map, Image> Consumer<I> for Transform<I, O, InB, OutB, Map, Image>
where
    I: Merge,
    O: Merge,
    InB: Buffer<I>,
    OutB: Buffer<O>,
    Map: Fn(<InB as Producer<I>>::OutP, I) -> (<OutB as Consumer<O>>::InP, O),
    Image: Fn(<<InB as Producer<I>>::OutPS as PointSet>::Set)
        -> <<OutB as Consumer<O>>::InPS as PointSet>::Set,
{
    type InP = <<InB as Consumer<I>>::InPS as PointSet>::Point;
    type InPS = <InB as Consumer<I>>::InPS;

    fn put(&mut self, p: Self::InP, i: I) {
        (&mut self.in_buf).put(p, i);
    }

    fn advance(&mut self, s: <Self::InPS as PointSet>::Set) {
        let in_buf = &mut self.in_buf;
        let image = &self.image;
        let map = &self.map;
        (in_buf).advance(s);
        in_buf.get(true).map(|mut bundle| {
            let image = image(bundle.punctuation);
            let points = bundle
                .points
                .drain(..)
                .map(|(pt, val)| map(pt, val))
                .collect();
            Bundle::<<OutB as Consumer<O>>::InPS, O>::new(image, points)
        });
    }
}

impl<I, O, InB, OutB, Map, Image> Processor<I, O> for Transform<I, O, InB, OutB, Map, Image>
where
    I: Merge,
    O: Merge,
    InB: Buffer<I>,
    OutB: Buffer<O>,
    Map: Fn(<InB as Producer<I>>::OutP, I) -> (<OutB as Consumer<O>>::InP, O),
    Image: Fn(<<InB as Producer<I>>::OutPS as PointSet>::Set)
        -> <<OutB as Consumer<O>>::InPS as PointSet>::Set,
{
}

pub fn transform<I, O, InB, OutB, Map, Image>(
    in_buf: InB,
    out_buf: OutB,
    map: Map,
    image: Image,
) -> impl Processor<I, O>
where
    I: Merge,
    O: Merge,
    InB: Buffer<I>,
    OutB: Buffer<O>,
    Map: Fn(<InB as Producer<I>>::OutP, I) -> (<OutB as Consumer<O>>::InP, O),
    Image: Fn(<<InB as Producer<I>>::OutPS as PointSet>::Set)
        -> <<OutB as Consumer<O>>::InPS as PointSet>::Set,
{
    Transform {
        in_buf: in_buf,
        out_buf: out_buf,
        map: map,
        image: image,
        _i: PhantomData,
        _o: PhantomData,
    }
}

#[test]
fn transform_puts() {}
