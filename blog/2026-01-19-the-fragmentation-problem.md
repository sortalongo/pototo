---
title: "The Fragmentation Problem"
date: 2026-01-19
draft: true
---

## 1. Hook

I've spent over a decade building data infrastructure—streaming systems at Google, observability at Twitter, transformation pipelines at Snowflake. Throughout, I've felt like the systems we build are brittle: hard to change, and easy to break.

There's a strange incongruency between the elegance of programming languages and databases, and the reality of developing and operating real systems atop them. That reality is filled with tedium and stress. It always felt like something was missing—like there must be a way to carry that elegance and simplicity into the real world.


## 2. Fragmented Systems

Modern systems are assembled from components: databases, caches, queues, services, frontends.
On one hand, these components are tremendously empowering for developers. 
They make it possible to quickly assemble systems with capabilities that would be impossible if every developer had to implement their own.
In principle, all you have to do is take these components off the shelf and assemble them into a coherent system with a bit of glue code. Then, voila, you have a sophisticated, robust system.

Unfortunately, when you try to follow this process, you realize a few things:
1. It's extremely tedious. The job of so many software developers in the last decade has come to involve an inordinate amount of configuration management and quality assurance, at the cost of the creativity and ingenuity that attracted us to the field.
2. It's highly error prone. Since there's no single, coherent programming model spanning all of these components, ensuring that they're assembled together correctly is purely the developer's responsibility, with only limited tooling available to assist.
3. It's unperformant. Priorities are (rightly) driven by the need to mitigate development cost and deployment risk. As a result, performance is often left as an afterthought, resulting in a poor user experience and wasted computing resources compared to what a system could be.

So, in practice, we rarely end up with that coherent, robust system we hoped for. Instead we end up with a **fragmented** system.
Fragmented systems are often, but not necessarily, distributed.
Their distinguishing characteristic is that they are assembled out of numerous components with mismatched programming models.
As a result of this fragmentation, they are brittle.

In practice, that brittleness manifests in many ways. 

*Contract Mismatches*
- Rename an API field, downstream service still expects the old name—runtime error
- Microservice A deploys v2, Microservice B still expects v1—runtime error
- None of these are caught at compile time because the structure of the overall system isn't represented anywhere but runtime 

*Cross-component Optimizations*
- "Push a filter down"—you want to fetch less data, but it requires changing the API contract at every layer between UI and database
- "Reorder a join"-changing the order in which lookups are done can massively reduce processing, but might require moving logic between components in a very awkward way.
- Move a computation from app to database (or vice versa)—rewrite in a different language, re-test, hope semantics match
- Add an index to speed up a query—but first trace through app code to understand access patterns

*Ceremony and risk around changes*
- Database migrations: write SQL, write rollback SQL, coordinate deploy order, handle partial failures
- Changing a shared data model: update schema, update every service, deploy in the right order and pray, or spend weeks testing with staging environments

*Impedance Mismatches*
- The type systems of DBs and PLs are often incompatible, leading to subtle edge cases that are hard to test because they depend on the data actually stored in the DB. Logic tests and data tests live in separate worlds even though they're fundamentally specifying requirements on the same program.
- Your ORM makes relationships easy to traverse, but generates N+1 queries because it doesn't understand the database


We'd like to have automated tooling to help us to ensure correctness, improve performance, and evolve our systems over time. 
However, automation relies on having a clear conceptual framework within which to operate. 
By definition, fragmented systems don't have such a framework.
Every time the components comprising them are combined in a novel way, new behavior can emerge.
As a result, the potential impact of automation is limited, which means fragmented systems tend to be brittle and developing them is tedious and stressful.

## 4. Coherent Systems

So fragmented systems are bad. What's the alternative?
To build systems within a single conceptual framework-- what we call **coherent systems**.
Coherent systems limit themselves to working within a specific framework. 
In exchange for that limitation, they can take advantage of much better automation (depending on the quality of that framework).


There are many examples of such frameworks that allow one to build coherent systems within specific domains: 
- Web frameworks like Ruby on Rails and Backend-as-a-Services like Firebase, Supabase, and Convex eliminate a great deal of the toil and errors of building web services.
- actor systems like Erlang or Ray make it dramatically easier to build distributed systems.
- "durable execution" systems like Temporal make fault tolerance easy to achieve.
- type systems in programming languages catch many logic errors and interface misuses
- the relational model in databases enable programmers to access incredible scale and performance with minimal effort.

When working entirely within frameworks like these, the leverage of automation is much greater.
Thus, programmers often get big boosts in their productivity, and the coherent systems they build often have better correctness and performance than comparable fragmented systems.

I've been speaking of automation in the abstract, and using examples of traditional automation.
But the most powerful form of automation emerged only recently in the form of agentic AI.
Agents are much more flexible than traditional forms of automation, and they have tremendous potential for improving the productivity of developers.
However, they too are sensitive to the distinction between fragmented and coherent systems.
This is clear when you look at examples of where agentic coding is most powerful. 
It's when working within narrow environments with clear rules, such as:
- stateless, single-page javascript apps
- Standard, boilerplate 3-tier archictures
- Isolated SQL tasks within a single, well-documented data warehouse

So it sounds like I'm arguing that coherent systems are always the way to go, and we should all just buy into whatever framework will let us get our job done. Right?
The problem is that existing coherent systems are domain-specific.
Their conceptual frameworks don't generalize to other contexts.
And that's a big problem because most modern systems are not domain-specific.
Modern apps generally span a wide variety of domains, including web and API serving, transaction processing, background processing, and analytical processing.
That means that trying to build a coherent system means severely limiting what your system will ultimately do.
And that just won't do.


## 5. Universality

You probably already guessed where I'm going with this.
What we need is a universal conceptual framework that can span domains.
Atop such a framework, coherent systems could be built without fencing themselves in to a single narrow domain.

Held back on several fronts:
- **Verification**: Can't reason about correctness across boundaries
- **Optimization**: Can't move computation to data or vice versa, can't prune unnecessary computation, 
- **Evolution**: Changing one piece ripples unpredictably to others

Think about how much productivity could be improved if these limitations were lifted. The difference becomes even more significant in a world where agents are accelerating us: agents are really good a churning out straightforward code and iterating against an oracle. They're really bad at reasoning broadly, connecting disparate pieces, and building out necessary infrastructure.

**Properties:**
- Unified model for state and computation (no grain boundaries)
- Semantics that enable automated reasoning—verification and optimization work across the whole
- Flexibility to map to different physical implementations 

TODO: Comparison to existing solutions

We're building a solution with these properties. Excited to talk more about what it'll look like!
