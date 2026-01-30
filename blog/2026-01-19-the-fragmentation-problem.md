---
title: "The Fragmentation Problem"
date: 2026-01-19
draft: true
---

## Hook

I've spent over a decade building data infrastructure: observability at Twitter, streaming systems at Google, transformation pipelines at Snowflake. Throughout, I've felt like the systems we build are brittle: hard to change, and easy to break.

There's a strange incongruency between the conceptual elegance of programming languages and databases, and the reality of developing and operating real systems using them. That reality is filled with tedium and stress. It has always felt like something was missing—like there must be a way to carry that elegance and simplicity into the real world.

We want to build the thing that's missing. 
But before we can explain what that thing is, we need to step back and talk about why things are the way they are. 
Decades of effort by thousands of brilliant minds have gone into field of computing.
That foundation deserves its due:
before proposing a radical idea, it's important to understand what you're departing from, and why.
And we're going to propose something a bit radical... but something that makes a lot of sense too.

## Models

Computers are magic.
They let abstract concepts manifest in and affect the real world.
(TODO: add examples)

Every computer program works in terms of a model, which is an abstract way to represent the world in simplified terms.
A model allows the program to ignore the overwhelming complexity of reality, and instead focus on the parts of the world that are essential to the programmer's goal.
At its most reductive, a program is a loop that receives inputs, updates its representation of the world, computes consequences to those updates, and sends output to effect some change in the world.
However, that reductive perspective leaves out a crucial fact: the choice of model that the program uses for its representations has a huge impact on which programs are feasible to develop and maintain.
In other words, there are better models and worse models.
Better models work in terms of intuitive, well-behaved concepts.
They also give you clear, useful rules about how to create programs and reason about their behavior.
Great models give you superpowers: using them makes it easier to think about programs, easier to write and read programs, and easier to create tooling to manipulate (i.e. verify, optimize, refactor, etc) those programs.

In a sense, all modern computer programs work in terms of the same foundational model: bits stored in memory and instructions to manipulate them, arranged in a dizzying variety of configurations depending on the hardware they're instantiated in. 
But this model is so low-level that it's hard to map its concepts to the familiar, high-level concepts we typically care about.
In other words, given a program written in terms of bits and intructions, it's very difficult to infer its purpose.
Conversely, given a high-level specification of a program's effects on the real world, it's very difficult to create a "bits and instructions" program that will satisfy that specification.

To make this mapping easier, we build higher-level models atop this foundation: programming languages, operating systems, databases, etc.
This concept, also known as a layer of abstraction, is familiar to most programmers.
Programming in terms of a high-level model comes with a sacrifice: you give up control over the way the program is "lowered" into a lower-level model.
When writing a program in Java, you lose the ability to manually manage memory.
But with that loss of control comes a reduction in complexity, which is often a favorable trade.
When using Java, you generally don't have to worry about memory management, and so may be more productive than in a lower-level language.

The best higher-level models are such that it rarely makes sense to work directly in terms of the lower-level model they abstract over.
We think of these models as **sealed**: they provide an abstraction that doesn't leak its internal details often.
The modern world has many examples of ubiquitous, sealed models.
It's very uncommon to want to write programs directly in assembly rather than a programming language, that don't run on an operating system, or that don't store state (if they have any) in a database.
Once a model in some domain becomes sealed, we see a shift in the kinds of programs that are developed in that domain.
Efforts bifurcate into developing programs in terms of that model, and developing programs that implement that model.
For example, most of the knowledge and effort that goes into understanding the specific instructions of various hardware architectures lies with compiler developers, not general-purpose programmers.

The magic of high-level models comes from automation.
Automation can help us to ensure correctness, improve performance, and evolve our systems over time. 
But automation works in terms of a specific model, and only has leverage over the concepts in that model.
For example, consider what an OS-level tool like `top` can tell you about your program: resource consumption, uptime, network throughput, etc.
It cannot do the things that are possible for a language-level tool like `gdb`, which works in terms of C's programming model.
In general, automation is useful at every level, as there are usually things that can only be done at that level.

## Components & Systems

Modern software systems (i.e. complicated programs) are assembled from software components (i.e. less-complicated programs that are intended to be combined with others). Examples include databases, caches, queues, services, frontends.
On one hand, the existence of components is tremendously empowering for developers. 
Components make it possible to quickly assemble systems with capabilities that would be impossible if every developer had to implement these capabilities themselves.
In principle, all a developer has to do is take these components off the shelf, wire them together with a bit of glue code, and voila, you have a sophisticated, robust system.

Unfortunately, too often, when you try to follow this process, you realize a few things:
1. It's extremely tedious. The job of so many software developers in the last decade has come to involve an inordinate amount of configuration management and quality assurance, at the cost of the creativity and ingenuity that attracted us to the field.
2. It's inflexible. Once you've chosen some components and wired them together, changing the capabilities of your system is quite difficult, since you usually can't modify the components and swapping out components is very hard.
2. It's highly error prone. Ensuring that they're wired together correctly is the developer's responsibility, with only limited tooling available to assist.
3. It's unperformant. Priorities are (rightly) driven by the need to minimize development cost and mitigate deployment risk. As a result, performance rarely receives much attention, and often degrades over the lifetime of a system.

So, in practice, the systems we build often end up brittle and we end up unsatisfied. But why is this the case? Is it a necessary consequence of building complex systems? We don't think so. We think it happens for a specific reason.

The components that comprise systems are themselves programmed in terms of some internal model.
Most components also interact with other components. 
Sometimes, they are designed to interact with other components with the same model, like a library in a programming language.
But sometimes, they are designed to interact using a lower-level model, like a microservice exposing an API.
When we build a system out of components, the model we use to reason about the system is determined by the interaction models, not the internal models, of the components.
So, when components use a lower-level model to interact, systems built using those components are forced to also use that lower-level model.
In the world of internet software, systems are overwhelmingly forced into what I'd call the "networks and operating systems" model.
In this model, we have computers, processes, memory, network addresses, packets, and the like.
Those are fantastically powerful abstractions, but they're far removed from the concepts we have in mind when writing our programs.
They work in terms of bytes and addresses, not objects, people, places, and actions.
Those high-level concepts are entirely implicit in the behavior of the bytes comprising the system, from the perspective of a network or operating system.

For example, say we write a program in a good programming language and connect it to a good relational database.
The internal models of the program and database have clean, well-defined semantics, and they allow us to model our domain reasonably well.
But the behavior of the system is not easily constrained by the semantics of either model.
Instead, we have to think in terms of networks and operating systems to understand any problem that is not entirely contained to one of the components (e.g. "the server process crashed", "the data encoding is corrupted", "the connection was dropped").

There's a good reason that so many components use a different interaction model than their internal model: interoperability.
The truth is that there are lots of models out there, with many valuable components built using them.
Unfortunately, most of those models are incompatible with each other in some way.
Components with incompatible internal models cannot interact directly. 
They must instead "drop down" to a lower-level, common model to mediate their interactions.
This is why the "networks and OSes" model is ubiquitous: it's very powerful, having stood the test of time, and is sufficiently low-level that most components can build atop it.
But achieving interoperability sacrifices the system-level benefits of working within a high-level model.

## Fragmented Systems

Let's call this kind of system a **fragmented system**.
The distinguishing characteristic of a fragmented system is that it is assembled out of numerous components with incompatible internal models.
Fragmented systems are brittle: they are hard to change and easy to break.
In practice, that brittleness manifests in many ways. 

*Contract Mismatches*
- Rename an API field, downstream service still expects the old name—runtime error
- Microservice A deploys v2, Microservice B still expects v1—runtime error
- None of these are caught at compile time because the structure of the overall system isn't represented anywhere but runtime 

*Cross-component Optimizations*
- "Push a filter down"—you want to fetch less data, but it requires changing the API contract at every layer between UI and database
- "Reorder a join"-changing the order in which lookups are done can massively reduce processing, but might require moving logic between components in a very awkward way.
- Move some logic from app to database (or vice versa)—rewrite in a different language, re-test, hope semantics match

*Ceremony and risk around changes*
- Database migrations: write SQL, write rollback SQL, coordinate deploy order, handle partial failures
- Changing a shared data model: update schema, update every service, deploy in the right order and pray, or spend weeks testing with staging environments

*Impedance Mismatches*
- The type systems of DBs and PLs are often incompatible, leading to subtle edge cases that are hard to test because they depend on the data actually stored in the DB. Logic tests and data tests live in separate worlds even though they're fundamentally specifying requirements on the same program.
- Your ORM makes relationships easy to traverse, but generates N+1 queries because it doesn't understand the database
 
What is the cause of this brittleness?
The nature of a fragmented system forces the developer to rely on a low-level model to reason about the system's behavior. 
These mismatched concepts mean that the components comprising fragmented systems are not trivially composable.
Every time a component is added or modified in a fragmented system, the implications of that change on other parts of the system are not constrained by that component's internal model.
They are only constrained by the low-level interaction model, constraints which are difficult to match to the requirements of the system.
The developer is responsible for carefully thinking through the implications of each change in terms of the interaction model.
Reaching confidence that the system meets its requirements typically requires extensive validation.

Good architecture can limit the scope that some changes can have on a system.
A well-architected system is divided into independent components, and most changes to such a system only require changes within a given component.
That means the implications of those changes can be considered only in terms of the component's internal model.
We call such changes **localized**.
However, changes to the ways components interact can have unexpected consequences in seemingly unrelated parts of the system.
This is because a change in the behavior of one component can cascade into other components in ways that are hard to predict.
These changes are **nonlocalized**.
It's not always easy to tell if a change is localized or nonlocalized.

So, fragmented systems are brittle by necessity. This brittleness can be mitigated somewhat by architecture and effort.
But without the ability to combine components within a single high-level interaction model, the kind of automation available to the system is fundamentally limited.
So, the effort required to build a fragmented system scales unfavorably with its complexity.

## 4. Coherent Systems

So fragmented systems are bad. What's the alternative?
To build systems within a single, high-level model-- what we call a **coherent system**.
Coherent systems limit themselves to working within a specific high-level model. 
In exchange for that limitation, they can take advantage of much better automation (depending on the quality of that model).

There are many examples of such models that allow one to build coherent systems within specific domains: 
- Type systems in programming languages catch many logic errors and interface misuses
- The relational model in databases enable programmers to access incredible scale and performance with minimal effort.
- Web frameworks like Ruby on Rails and Backend-as-a-Services like Firebase, Supabase, and Convex eliminate a great deal of the toil and errors of building web services.
- Actor systems like Erlang or Ray make it easier to build certain kinds of distributed systems.
- "durable execution" systems like Temporal make fault tolerance more approachable.

When working entirely within models like these, the leverage of automation is much greater.
Thus, programmers often get big boosts in their productivity, and the coherent systems they build often have better correctness and performance than comparable fragmented systems.

So it sounds like I'm arguing that coherent systems are always the way to go, and we should all just buy into whatever model will let us get our job done. Right?
The problem is that all of the high-level models I listed above are domain-specific.
They don't generalize to other contexts.
And that's a big problem because most modern systems are not domain-specific.
Modern apps typically span a wide variety of domains, including web and API serving, transaction processing, background processing, and analytical processing.
That means that trying to build a coherent system implies severely limiting what your system will ultimately do.
So, even when we begin by trying to build a coherent system, application requirements often push us outside of the domain of whatever model we chose.
Then, our hand forced, we introduce a component with another model, and our coherent system fragments.


## General-Purpose Models & Coherence

Might think that there's an inherent tradeoff between domain-specificity and high-level-ness (is this the right term?).
Historically, there is. But this doesn't seem strictly necessary. We have historical examples of good abstractions being invented that are both general-purpose *and* high-level (C, relational DBs).
These innovations are rare, but they are possible. 

What we need is a high-level model that can span all of the domains typically required to build internet software.
Atop such a model, coherent systems could be built without fencing oneself in to a single narrow domain.

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

**AI**
I've been speaking of automation in the abstract, and using examples of traditional automation.
But the most powerful form of automation emerged only recently in the form of agentic AI.
Agents are much more flexible than traditional forms of automation, and they have tremendous potential for improving the productivity of developers.
However, they too are sensitive to the distinction between fragmented and coherent systems.
This is clear when you look at examples of where agentic coding is most powerful. 
It's when working within narrow environments with clear rules, such as:
- stateless, single-page javascript apps
- Standard, boilerplate 3-tier archictures
- Isolated SQL tasks within a single, well-documented data warehouse

We're building a solution with these properties. Excited to talk more about what it'll look like!
