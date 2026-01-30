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
Decades of effort by thousands of brilliant minds have gone into the field of computing.
That foundation deserves its due: before proposing a departure from it, it's important to understand what you're departing from, and why.
In this post, we're going to argue that the orthodoxy creates a forced choice between powerful tooling and general-purpose software.
But this is a false dichotomy—we can have both at once.

## Models

Computers are magic.
They let abstract concepts manifest in and affect the real world.
A spreadsheet formula updates a budget, and you decide whether you can afford rent.
A routing algorithm computes the shortest path, and you arrive at your destination.
A database records a transaction, and money moves between bank accounts.

Every computer program works in terms of a model, which is an abstract way to represent the world in simplified terms.
A model allows the program to ignore the overwhelming complexity of reality, and instead focus on the parts of the world that are essential to the programmer's goal.
At its most reductive, a program is a loop that receives inputs, updates its representation of the world, computes consequences to those updates, and sends output to effect some change in the world.
However, that reductive perspective leaves out a crucial fact: the choice of model that the program uses for its representations has a huge impact on which programs are feasible to develop and maintain.
In other words, there are better models and worse models.
Better models work in terms of intuitive, well-behaved concepts.
They also give you clear, useful rules about how to create programs and reason about their behavior.
Great models give you superpowers: using them makes it easier to think about programs, easier to write and read programs, and easier to create tooling to manipulate (i.e. verify, optimize, refactor, etc) those programs.

So why don't we just use great models all the time? To answer that, we need to start at the bottom. In a sense, all modern computer programs work in terms of the same foundational model: bits stored in memory and instructions to manipulate them, arranged in a dizzying variety of configurations depending on the hardware they're instantiated in. 
But this model is so low-level that it's hard to map its concepts to the familiar, high-level concepts we typically care about.
In other words, given a program written in terms of bits and instructions, it's very difficult to infer its purpose.
Conversely, given a high-level specification of a program's effects on the real world, it's very difficult to create a "bits and instructions" program that will satisfy that specification.

To make this mapping easier, we build higher-level models atop this foundation: programming languages, operating systems, databases, etc.
This concept, also known as a layer of abstraction, is familiar to most programmers.
Programming in terms of a high-level model comes with a sacrifice: you give up control over the way the program is "lowered" into a lower-level model.
When writing a program in Java, you lose the ability to manually manage memory.
But with that loss of control comes a reduction in complexity, which is often a favorable trade.
When using Java, you generally don't have to worry about memory management, and so may be more productive than in a lower-level language.

Much of the value of high-level models comes from tooling.
Tooling can help us ensure correctness, improve performance, and evolve our systems over time.
But tooling works in terms of a specific model, and only has leverage over the concepts in that model.
For example, consider what an OS-level tool like `top` can tell you about your program: resource consumption, uptime, network throughput, etc.
It cannot do the things that are possible for a language-level tool like `gdb`, which works in terms of C's programming model.

But tooling only helps when you're working within its model. If you frequently need to "escape" to a lower level, you lose those benefits.
The best higher-level models are ones where you rarely need to escape.
We call these models **sealed**: they provide an abstraction that doesn't leak its internal details often.
The modern world has many examples of ubiquitous, sealed models.
It's rare to find programs written directly in assembly, or that bypass the operating system, or that manage state without a database.
Once a model becomes sealed, efforts bifurcate: some people develop programs in terms of that model, others develop programs that implement it.

This is the ideal: work within a sealed, high-level model, and let tooling handle the rest. But what happens when the system you're building doesn't fit within a single model?

## Components & Systems

Modern software systems (i.e. complicated programs) are assembled from software components (i.e. less-complicated programs that are intended to be combined with others). Examples include databases, caches, queues, services, frontends.
On one hand, the existence of components is tremendously empowering for developers. 
Components make it possible to quickly assemble systems with capabilities that would be impossible if every developer had to implement these capabilities themselves.
In principle, all a developer has to do is take these components off the shelf, wire them together with a bit of glue code, and voila, you have a sophisticated, robust system.

Unfortunately, too often, when you try to follow this process, you realize a few things:
1. It's extremely tedious. The job of so many software developers in the last decade has come to involve an inordinate amount of configuration management and quality assurance, at the cost of the creativity and ingenuity that attracted us to the field.
2. It's inflexible. Once you've chosen some components and wired them together, changing the capabilities of your system is quite difficult, since you usually can't modify the components and swapping out components is very hard.
3. It's highly error prone. Ensuring that they're wired together correctly is the developer's responsibility, with only limited tooling available to assist.
4. It's unperformant. Priorities are (rightly) driven by the need to minimize development cost and mitigate deployment risk. As a result, performance rarely receives much attention, and often degrades over the lifetime of a system.

So, in practice, the systems we build often end up brittle and we end up unsatisfied. But why is this the case? Is it a necessary consequence of building complex systems? We don't think so. We think it happens for a specific reason.

The components that comprise systems are themselves programmed in terms of some internal model.
Most components also interact with other components. 
Sometimes, they are designed to interact with other components with the same model, like a library in a programming language.
But sometimes, they are designed to interact using a lower-level model, like a microservice exposing an API.
When we build a system out of components, the model we use to reason about the system is determined by the interaction models, not the internal models, of the components.
So, when components use a lower-level model to interact, systems built using those components are forced to also use that lower-level model.
In the world of internet software, systems are overwhelmingly forced into what we call the "networks and operating systems" model.
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
- "Reorder a join"—changing the order in which lookups are done can massively reduce processing, but might require moving logic between components in a very awkward way.
- Move some logic from app to database (or vice versa)—rewrite in a different language, re-test, hope semantics match

*Ceremony and risk around changes*
- Database migrations: write SQL, write rollback SQL, coordinate deploy order, handle partial failures
- Changing a shared data model: update schema, update every service, deploy in the right order and pray, or spend weeks testing with staging environments

*Impedance Mismatches*
- The type systems of DBs and PLs are often incompatible, leading to subtle edge cases that are hard to test because they depend on the data actually stored in the DB. Logic tests and data tests live in separate worlds even though they're fundamentally specifying requirements on the same program.
- Your ORM makes relationships easy to traverse, but generates N+1 queries because it doesn't understand the database

These are symptoms. What is the underlying cause of this brittleness?
The nature of a fragmented system forces the developer to rely on a low-level model to reason about the system's behavior. 
These mismatched concepts mean that the components comprising fragmented systems are not trivially composable.
Every time a component is added or modified in a fragmented system, the implications of that change on other parts of the system are not constrained by that component's internal model.
They are only constrained by the low-level interaction model, constraints which are difficult to match to the requirements of the system.
The developer is responsible for carefully thinking through the implications of each change in terms of the interaction model.
Reaching confidence that the system meets its requirements typically requires extensive validation.

Can good architecture mitigate this? Somewhat. Good architecture can limit the scope that some changes can have on a system.
A well-architected system is divided into independent components, and most changes to such a system only require changes within a given component.
That means the implications of those changes can be considered only in terms of the component's internal model.
We call such changes **localized**.
However, changes to the ways components interact can have unexpected consequences in seemingly unrelated parts of the system.
This is because a change in the behavior of one component can cascade into other components in ways that are hard to predict.
These changes are **nonlocalized**.
It's not always easy to tell if a change is localized or nonlocalized.

So, fragmented systems are brittle by necessity. This brittleness can be mitigated somewhat by architecture and effort.
But without the ability to combine components within a single high-level interaction model, the kind of tooling available to the system is fundamentally limited.
And the effort required to build a fragmented system scales unfavorably with its complexity.

## Coherent Systems

So fragmented systems are bad. What's the alternative?
We call it a **coherent system**.
Coherent systems limit themselves to working within a specific high-level model.
In exchange for that limitation, tooling can operate at the level of the model across the whole system.
This creates major opportunities for verification, optimization, and automation.

There are many examples of such models that allow one to build coherent systems within specific domains: 
- Type systems in programming languages catch many logic errors and interface misuses
- The relational model in databases enables programmers to access incredible scale and performance with minimal effort.
- Web frameworks like Rails, Express, and Django and Backends-as-a-Service like Firebase, Supabase, and Convex eliminate a great deal of the toil and errors of building web services.
- Actor systems like Erlang or Ray make it easier to build certain kinds of distributed systems.
- "durable execution" systems like Temporal and ReState make fault tolerance more approachable.

When working entirely within models like these, the leverage of tooling is much greater.
Programmers often get big boosts in their productivity, and the coherent systems they build often have better correctness and performance than comparable fragmented systems.

So it sounds like we're arguing that coherent systems are always the way to go, and everyone should just buy into whatever model will let them get their job done. Right?
The problem is that all of the high-level models I listed above are domain-specific.
They don't generalize to other contexts.
And that's a big problem because most modern internet software systems are not domain-specific.
Modern applications typically span a wide variety of domains, including web and API serving, transaction processing, background processing, analytical processing, and telemetry.
That means that trying to keep a system coherent limits what your system can ultimately do.
Even if we begin by trying to build a coherent system, application requirements don't care.
Those requirements push us outside of a single domain, forcing us to reach for components with a different internal model.
So, step by step, our system fragments.

The industry's response to this situation has been to accept fragmentation as inevitable.
"Use the right tool for the job," we say. Each domain gets its own specialized component, and we'll wire them together.
This is pragmatic advice—it reflects reality. But it also encodes a hidden assumption: that fragmentation is an acceptable cost, that we can't do better.
We reject that assumption.

## General-Purpose Models & Coherence

But rejecting an assumption isn't the same as proving it wrong. Is a general-purpose, high-level model actually possible?
If it were possible, wouldn't one already exist?
You might speculate that there's an inherent tradeoff between generality and how high-level a model can be.
Looking at the current population of models, we do observe such a trend empirically.
But it doesn't seem strictly necessary.
In this post, many of the examples we've referred to are both general-purpose and sealed models.
The C language exists in the stack of nearly all modern software.
Linux is ubiquitous, only inappropriate in rare circumstances like hard-real-time and safety-critical systems.
Relational databases are nearly as ubiquitous, with NoSQL DBs comprising only a small proportion of usage and applications only rarely needing to reach for models below that level.
Furthermore, there are examples of the general-purpose vs high-level tradeoff being pushed out.
The Rust language is at once more general-purpose than C, higher-level, and has better tooling.
These innovations are rare, but they are possible.

So let's imagine one more such innovation. If we could build a high-level, sealed model that were general-purpose across the domains typically required to build internet software, coherent systems could be built atop it without being fenced in to a single narrow domain.
This would create tremendous opportunities for tooling:
- Development would be accelerated dramatically by enabling components to interact directly in terms of a high-level model.
- System-wide verification would become tractable for most applications.
- Performance tooling could profile applications and apply optimizations automatically across the whole system.
- Operational tooling could instrument, monitor, and orchestrate services with minimal setup and oversight. 

If realized, these opportunities have the potential to revolutionize the development of internet software.

This is what we're building: a high-level, general-purpose, sealed model for internet software.
It's a bet against the conventional wisdom that says "use the right tool for the job" and accept the resulting fragmentation.
We believe coherence doesn't have to be sacrificed for generality—and that the payoff for achieving both is immense.

Of course, we're not the first to attempt this. Many have tried to build general-purpose, high-level models for software development. Most either sacrificed generality to achieve coherence, or sacrificed high-level semantics to achieve generality. The ones that achieved both often failed to become sealed—they leaked too often to displace lower-level alternatives.
So why do we think we can succeed? We'll share more about our approach in the future. For now, we'll just say: we believe recent advances in programming language theory and database systems have opened a path that wasn't available before.

---

## Postscript: What About AI?

That's the core argument. But there's a question we expect many readers are already asking: doesn't AI change everything? Why worry about models and coherence when agents can just handle the complexity for us?

We've been speaking of tooling in the abstract, and using examples of traditional tooling.
But the most powerful form of tooling emerged only recently in the form of agentic AI.
Agents are much more flexible than traditional forms of tooling, and they have tremendous potential for improving the productivity of developers.
They already represent a revolution in software development, and in many fields beyond.

Given this, there's a popular narrative that AI itself is sufficient to realize the opportunities referenced above.
In this narrative, code is merely a by-product, an intermediate representation of the programmer's intent, the truth of which is captured by the prose prompts and documents that are provided to the agents.
The agents will handle all of the complexity of code. 
We don't need to worry about abstractions or maintainability.
Those are vestigial concerns from an era that will soon be past.
AI is the future, and traditional software engineering will soon be obsolete.

We believe this narrative builds on several fundamental misunderstandings of software (and of epistemology more generally, but let's not get sidetracked).

The first misunderstanding is the conflation of ambiguity and abstractness.
Code is often "low level", in the sense of "it doesn't map well to the concepts in my domain". 
But this is not an inherent property of all code. It's a property of the model being used.
Some models are low-level, some are high-level. But code can be either.
What code is truly about is precision. Code has semantics—it's unambiguous. 
It's easy to conflate ambiguity and abstraction—both involve "a single statement that could refer to multiple meanings." 
But the meanings of an ambiguous statement are unconstrained. 
The meanings of an abstract statement are tightly constrained by the semantics of the model. 
Those semantics are designed to yield useful groupings, so you don't get a combinatorial explosion when you compose statements.
Prose is ambiguous, and always will be. Fluidity of meaning is core to its utility.
Code is precise. Precision is core to its utility.
Whether the one reading and writing that code is a human or AI may shift over time.
But code will never be obviated by prose because precision will always be important when designing a complex system.

The second misunderstanding relates to levels of abstraction.
Even when communicating in prose, programmers and AI still need shared concepts to reason about.
Models will always be important for that.
Without layers of abstraction, complexity grows combinatorially. No matter how fast AI gets smarter, combinatorial explosions can always grow faster.
The only way to make progress is to find ways to reduce the exponential into polynomial amounts of work.
That's what the difference between fragmented and coherent systems is all about.
The distinction will never become irrelevant, no matter who is building the system.
We will just keep building better and better models, extending the reach of coherence to ever greater heights.

There is substantial evidence for this view: agentic coding is most powerful when working within narrow environments with clear rules, such as:
- Stateless, single-page javascript apps
- Standard 3-tier architectures using a good web framework
- Clearly-articulated analytics tasks within a single, well-documented data warehouse

Naturally, the capabilities of AI systems will continue growing.
Consequently, so will the breadth of the domains they can excel within.
But there will always be room for innovations that make AI more productive.
Those innovations will amplify the gains that increased AI model capabilities bring, not compete with them.
AI is not an excuse to stop innovating.
