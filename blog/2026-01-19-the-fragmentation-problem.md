---
title: "The Fragmentation Problem"
date: 2026-01-19
draft: true
---

## Hook

I've spent over a decade building data infrastructure: observability at Twitter, streaming systems at Google, transformation pipelines at Snowflake. Throughout, I've felt like the systems we build are brittle: hard to change, and easy to break.

There's a strange incongruency between the conceptual elegance of programming languages and databases, and the reality of developing and operating real systems using them. That reality is filled with tedium and stress. It has always felt like something was missing—like there must be a way to carry that elegance and simplicity into the real world.


## Models

Computers are magic.
They let abstract concepts manifest in and affect the real world.
(TODO: add examples)

Computers work by modeling some aspect of the real world, representing a simplified version of that world internally, and manipulating it to achieve some goal.
A model is a correspondence between concepts and objects in the real world.
It is a set of concepts, rules for how to combine them, rules for how those combinations map to real-world behavior.

In a sense, all computer programs work in terms of the same base model: bits stored in memory, instructions to manipulate them, arranged in a dizzying variety of configurations depending on the hardware they're instantiated in. 
But this model is so low-level that it's hard to map its concepts to the familiar, high-level concepts we typically care about.
That is, given a program written in terms of bits and intructions, it's very difficult to understand its purpose.
Conversely, given a high-level specification of a program's effects on the real world, it's very difficult to create a "bits and instructions" program that will satisfy that specification.

To make this mapping easier, we build higher-level models atop this base: programming languages, operating systems, databases, etc.
Using a high-level model gives up control over the way the program is "lowered" into a lower-level model.
When writing a program in Java, you lose the ability to manually manage memory.
But with that loss of control comes a reduction in complexity, which is often a favorable trade.
When using Java, you generally don't have to worry about memory management.
But there are better models and worse models. Better models work in terms of intuitive, high-level concepts.
They also give you clear, useful rules about how to create programs and reason about their behavior.
Great models are magical: using them makes it easier to think about programs, easier to write and read programs, and easier to create tooling to manipulate (i.e. verify, optimize, refactor, etc) those programs.
The best models are such that it rarely makes sense to work directly in terms of the lower-level model they abstract over.
It's extremely uncommon to want to write programs directly in assembly, that don't run on an operating system, or that don't store state (if they have any) in a database.


## Fragmented Systems

Modern systems (i.e. complicated programs) are assembled from components (i.e. less-complicated pieces of programs that are intended to be combined). Examples include databases, caches, queues, services, frontends.
On one hand, the existence of components is tremendously empowering for developers. 
Components make it possible to quickly assemble systems with capabilities that would be impossible if every developer had to implement their own.
In principle, all you have to do is take these components off the shelf and assemble them into a coherent system with a bit of glue code. Then, voila, you have a sophisticated, robust system.

Unfortunately, when you try to follow this process, you realize a few things:
1. It's extremely tedious. The job of so many software developers in the last decade has come to involve an inordinate amount of configuration management and quality assurance, at the cost of the creativity and ingenuity that attracted us to the field.
2. It's highly error prone. Ensuring that they're assembled together correctly is the developer's responsibility, with only limited tooling available to assist.
3. It's unperformant. Priorities are (rightly) driven by the need to mitigate development cost and deployment risk. As a result, performance rarely receives much attention, and often degrades over the lifetime of a system.

So, in practice, we rarely end up with that coherent, robust system we hoped for. Instead we end up with a **fragmented** system.

Components have some internal model that determines how they work, and they interact with each other in terms of another, often lower-level, model.
When we build systems out of components, the amount of leverage (TODO: better word) we get over the system is determined by the interaction model, which is typically lower-level than the internal model.
The overwhelmingly dominant interaction model today is what I'd call the "networks and operating systems" model, and it is quite low level.
In this model, we have computers, processes, memory, network ports, and the like.
But the actual concepts we had in mind when writing the programs are entirely implicit in the behavior of the bytes comprising the system.
This means that, for example, if we write a program in a nice programming language and connect it to a relational database, then, even though the internal models of the program and database have clean, well-defined semantics, the behavior of our system is not easily constrained by those semantics.
Instead, we have to think in terms of networks and operating systems to understand them (e.g. "my server process crashed", "my data encoding is corrupted", "my connection was dropped").

The distinguishing characteristic of a fragmented system is that it is assembled out of numerous components with incompatible internal models, which forces the developer to rely on the interaction model to reason about the system's behavior. 
These mismatched concepts mean that the components comprising these systems are not really composable.
Every time a component is added to a fragmented system, the implications of that change on other parts of the system are not constrained.
The developer is responsible for carefully thinking through those implications and doing comprehensive testing to add confidence.
Good architecture can limit the scope that small, localized changes can have on a system.
However, large scale changes or even nuanced changes in API contracts can have unexpected consequences in seemingly unrelated parts of the system.
As a result, fragmented systems are brittle.

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


High-level models make it possible to dramatically reduce that brittleness through automation.
Automation can help us to ensure correctness, improve performance, and evolve our systems over time. 
Unfortunately, fragmented systems by necessity work in terms of the low-level interaction models of their components.
That puts severe limits on the kinds of automation that are possible.
For example, consider what an OS-level tool like `top` can tell you about your program: resource consumption, uptime, network throughput, etc.
It cannot do the things that are possible for a language-level tool like `gdb`, which works in terms of C's programming model.

As a result, fragmented systems are fundamentally limited in the degree to which their development and operation can be automated.

## 4. Coherent Systems

So fragmented systems are bad. What's the alternative?
To build systems within a single conceptual model-- what we call **coherent systems**.
Coherent systems limit themselves to working within a specific model. 
In exchange for that limitation, they can take advantage of much better automation (depending on the quality of that model).


There are many examples of such models that allow one to build coherent systems within specific domains: 
- Web models like Ruby on Rails and Backend-as-a-Services like Firebase, Supabase, and Convex eliminate a great deal of the toil and errors of building web services.
- actor systems like Erlang or Ray make it dramatically easier to build distributed systems.
- "durable execution" systems like Temporal make fault tolerance easy to achieve.
- type systems in programming languages catch many logic errors and interface misuses
- the relational model in databases enable programmers to access incredible scale and performance with minimal effort.

When working entirely within models like these, the leverage of automation is much greater.
Thus, programmers often get big boosts in their productivity, and the coherent systems they build often have better correctness and performance than comparable fragmented systems.


<!--TODO: crisper terminology around coherent systems and the models that support coherent systems. 
What about a model allows it to yield a coherent system?
It's something about it having concepts that map cleanly to the domain in question.
In a sense, the very idea of a "domain" implies some kind of programming model to which automated reasoning can be applied.
When this is the case, it's much easier to reason about systems built out of that programming model, which facilitates verification and optimization.-->

I've been speaking of automation in the abstract, and using examples of traditional automation.
But the most powerful form of automation emerged only recently in the form of agentic AI.
Agents are much more flexible than traditional forms of automation, and they have tremendous potential for improving the productivity of developers.
However, they too are sensitive to the distinction between fragmented and coherent systems.
This is clear when you look at examples of where agentic coding is most powerful. 
It's when working within narrow environments with clear rules, such as:
- stateless, single-page javascript apps
- Standard, boilerplate 3-tier archictures
- Isolated SQL tasks within a single, well-documented data warehouse

So it sounds like I'm arguing that coherent systems are always the way to go, and we should all just buy into whatever model will let us get our job done. Right?
The problem is that existing coherent systems are domain-specific.
Their conceptual models don't generalize to other contexts.
And that's a big problem because most modern systems are not domain-specific.
Modern apps generally span a wide variety of domains, including web and API serving, transaction processing, background processing, and analytical processing.
That means that trying to build a coherent system means severely limiting what your system will ultimately do.
And that just won't do.


## General-Purpose Conceptual Models

Might think that there's an inherent tradeoff between domain-specificity and automatability (is this the right term?).
Historically, there is. But this doesn't seem strictly necessary. We have historical examples of good abstractions being invented that are both general-purpose *and* easier (C, relational DBs).
These innovations are rare, but they are possible. 

What we need is a model that can span domains.
Atop such a model, coherent systems could be built without fencing themselves in to a single narrow domain.

Held back on several fronts:
- **Verification**: Can't reason about correctness across boundaries
- **Optimization**: Can't move computation to data or vice versa, can't prune unnecessary computation, 
- **Evolution**: Changing one piece ripples unpredictably to others

Think about how much productivity could be improved if these limitations were lifted. The difference becomes even more significant in a world where agents are accelerating us: agents are really good a churning out straightforward code and iterating against an oracle. They're really bad at reasoning broadly, connecting disparate pieces, and building out necessary infrastructure.

**Properties:**
- Unified model for state and computation (no grain boundaries)
- Semantics that enable automated reasoning—verification and optimization work across the whole
- Flexibility to map to different physical implementations 

<!-- TODO: Comparison to existing solutions -->

We're building a solution with these properties. Excited to talk more about what it'll look like!
