---
title: "The Fragmentation Problem"
date: 2026-01-19
draft: true
---

## 1. Hook (with personal experience)

I've spent over a decade building data infrastructure—streaming systems at Google, observability at Twitter, transformation pipelines at Snowflake. Throughout, I've felt like the systems we build are brittle: hard to change, and easy to break.

There's a strange incongruency between the elegance of programming languages and databases, and the reality of developing and operating real systems atop them. That reality is filled with tedium and stress. It always felt like something was missing—like there must be a way to carry that elegance and simplicity into the real world.


## 2. The observable problem: Brittleness


*Contract mismatches (no shared source of truth)*
- Rename an API field, downstream service still expects the old name—runtime error
- Microservice A deploys v2, Microservice B still expects v1—runtime error
- None of these are caught at compile time because the structure of the overall system isn't represented anywhere but runtime 

*Optimizations that require cross-system coordination*
- "Push a filter down"—you want to fetch less data, but it requires changing the API contract at every layer between UI and database
- "Reorder a join"-changing the order in which lookups are done can massively reduce processing, but might require moving logic between components in a very awkward way.
- Move a computation from app to database (or vice versa)—rewrite in a different language, re-test, hope semantics match
- Add an index to speed up a query—but first trace through app code to understand access patterns

*Ceremony and risk around changes*
- Database migrations: write SQL, write rollback SQL, coordinate deploy order, handle partial failures
- Changing a shared data model: update schema, update every service, deploy in the right order and pray, or spend weeks testing with staging environments

*The "two worlds" problem*
- The type systems of DBs and PLs are often incompatible, leading to subtle edge cases that are hard to test because they depend on the data actually stored in the DB. Logic tests and data tests live in separate worlds even though they're fundamentally specifying requirements on the same program.
- Your ORM makes relationships easy to traverse, but generates N+1 queries because it doesn't understand the database

## 3. The root cause: Fragmentation

Modern systems are assembled from pieces: databases, caches, queues, services, frontends. Each has its own programming model, semantics, and failure modes. Developers manually translate intent across these boundaries with limited help from tooling.


- Historical reasons for the split (durability, query optimization, shared access)
- Consequence: two incompatible worldviews reconciled by hand
- Every interface is an opportunity for bugs and missed optimizations

## 4. The opportunity: What's possible if we fixed this?

Held back on several fronts:
- **Verification**: Can't reason about correctness across boundaries
- **Optimization**: Can't move computation to data or vice versa, can't prune unnecessary computation, 
- **Evolution**: Changing one piece ripples unpredictably to others

Think about how much productivity could be improved if these limitations were lifted. The difference becomes even more significant in a world where agents are accelerating us: agents are really good a churning out straightforward code and iterating against an oracle. They're really bad at reasoning broadly, connecting disparate pieces, and building out necessary infrastructure.

Look at examples of where agentic coding is most powerful. It's when working within existing, self-consistent frameworks:
- stateless, single-page javascript apps
- boring, standard-architecture 3-tier apps
- writing SQL _inside_ of a data warehouse


## 5. Properties of a solution

**Metallurgy metaphor to weave in:**
- Our systems are like brittle metal—rigid crystalline structures with grain boundaries (the seams between services, databases, APIs)
- Stress concentrates at these boundaries: that's where bugs happen, where changes break things, where optimization stops
- The structure looks solid but fractures under pressure
- What we need is a forge—a process that transforms brittle material by unifying its internal structure
- In annealed metal, the grain boundaries dissolve; stress flows through instead of concentrating

**Properties (framed as what the "forge" would produce):**
- Unified model for state and computation (no grain boundaries)
- Semantics that enable automated reasoning (stress can flow—verification and optimization work across the whole)
- Flexibility to map to different physical implementations (the unified structure can be shaped into different forms without fracturing)
