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

Modern systems are assembled from components: databases, caches, queues, services, frontends.
On one hand, these components are tremendously empowering for developers. 
They make it possible to quickly assemble systems with capabilities that would be impossible if every developer had to implement their own.
In principle, all you have to do is take these components off the shelf and assemble them into a coherent system with a bit of glue code. Then, voila, you have a sophisticated, robust system.

Unfortunately, when you try to follow this process, you realize a few things:
1. It's extremely tedious. The job of so many software developers in the last decade has come to involve a tremendous amount of configuration management and quality assurance, at the cost of the creativity and ingenuity that attracted us to the field.
2. It's highly error prone. Since there's no single, coherent programming model spanning all of these components, ensuring that they're assembled together correctly is purely the developer's responsibility, with only limited tooling available to assist.
3. It's unperformant. Many architectural decisions are (rightly) driven by the need to mitigate development cost and deployment risk. As a result, performance is often left as an afterthought, resulting in a poor user experience and wasted computing resources compared to what a system could be.

We'd like to have automated tooling to help us reason about these systems to ensure correctness, to optimize the code we write, and to help us evolve our systems over time. However, automation relies on having a clear framework within which to operate. 
There is no such framework because each component has its own programming model, semantics, and failure modes.
Every time these components are combined in a novel way, new behavior can emerge.
We call this problem the *fragmentation of abstractions*.

Without such a framework, the potential impact of automation is very limited, which is why systems tend to be brittle, and developing them is tedious and stressful.


- Historical reasons for the split (durability, query optimization, shared access)
- Consequence: two incompatible worldviews reconciled by hand

## 4. The opportunity: What's possible if we fixed this?

Held back on several fronts:
- **Verification**: Can't reason about correctness across boundaries
- **Optimization**: Can't move computation to data or vice versa, can't prune unnecessary computation, 
- **Evolution**: Changing one piece ripples unpredictably to others

Think about how much productivity could be improved if these limitations were lifted. The difference becomes even more significant in a world where agents are accelerating us: agents are really good a churning out straightforward code and iterating against an oracle. They're really bad at reasoning broadly, connecting disparate pieces, and building out necessary infrastructure.

Look at examples of where agentic coding is most powerful. It's when working within narrow environments with clear rules:
- stateless, single-page javascript apps
- Standard, boilerplate 3-tier archictures
- Isolated SQL tasks within a single, well-documented data warehouse


## 5. Properties of a solution

<!-- **Metallurgy metaphor to weave in:**
- Our systems are like brittle metal—rigid crystalline structures with grain boundaries (the seams between services, databases, APIs)
- Stress concentrates at these boundaries: that's where bugs happen, where changes break things, where optimization stops
- The structure looks solid but fractures under pressure
- What we need is a forge—a process that transforms brittle material by unifying its internal structure
- In annealed metal, the grain boundaries dissolve; stress flows through instead of concentrating -->

**Properties:**
- Unified model for state and computation (no grain boundaries)
- Semantics that enable automated reasoning—verification and optimization work across the whole
- Flexibility to map to different physical implementations 

TODO: Comparison to existing solutions

We're building a solution with these properties. Excited to talk more about what it'll look like!
