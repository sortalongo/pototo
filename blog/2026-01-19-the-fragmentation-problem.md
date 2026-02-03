---
title: "Composition Shouldn’t be this Hard"
date: 2026-01-19
draft: true
---

I've spent over a decade building data infrastructure: observability systems at Twitter, streaming data processing systems at Google, declarative data processing systems at Snowflake. From the beginning, I noticed a strange gap between the conceptual elegance of programming languages and databases, and the reality of developing and operating real systems using them. That reality is filled with tedium and stress. All of the systems I’ve ever worked on have felt brittle in one way or another: hard to change, and easy to break.

Infrastructure engineers develop paranoia around change. We invest more effort testing and deploying changes than making them. We call it maturity, but I've never stopped questioning it. There must be a way to delegate the tedium to our tools and focus on what attracted us to this field: brainstorming ideas, trying them out, and seeing their effects.

But what’s missing, exactly? Decades of effort by thousands of brilliant minds have gone into the field of computing, much of it directed at closing the gap between [accidental and essential complexity](https://en.wikipedia.org/wiki/No_Silver_Bullet). Surely some major innovation in the foundation isn't just waiting to be discovered—wouldn't someone have found it already?

Maybe. But maybe not. The structure of modern abstractions points to a specific opportunity: the status quo forces a choice between powerful tooling and general-purpose software. This feels like a false dichotomy. There's no reason we can't have both—if we can find the right model.

After years of searching, I think I’ve found one. Implementing it requires more than I can do alone, which is why I'm starting Cambra with Daniel Mills and Skylar Cook. We are developing a new kind of programming system that rethinks the traditional software stack. Our goal: make developing internet software feel like working on a single, coherent system, not wiring together a fragmented mess of components. In what follows, I will explain why models matter, how fragmentation undermines them, and why building multi-domain coherent systems is both possible and necessary.

## **Models Give You Superpowers**

Computers are magic. They let abstract concepts manifest in and affect the real world. A spreadsheet formula updates a budget, and you decide whether you can afford that shiny new thing. A routing algorithm computes the shortest path, and you arrive at your destination. A database records a transaction, and money moves between bank accounts.

Every computer program works in terms of a **model***:* an abstract way to represent the world in simplified terms. Models allow programs to ignore the overwhelming complexity of reality, and instead focus on the parts of the world that are essential to the programmer's goal. At its most reductive, a program is a loop that receives input, updates internal state, computes consequences, and sends output. However, that oversimplification masks a deep truth: the choice of model has a huge impact on which programs are feasible to develop and maintain. In other words, there are better models and worse models. Better models rely on intuitive, well-behaved concepts and give you useful rules about how to create programs and reason about their behavior. Great models give you superpowers. They don’t just make programs easier to read and write. They make them easier to reason about. They make it possible to create tooling that can verify, optimize, and refactor programs automatically.

So why don't we just use great models all the time? To answer that, we need to start at the bottom. All modern computer programs ultimately work in terms of the same foundational model: bits stored in memory and instructions to manipulate them. But this model is so low-level that it's hard to map its concepts to the familiar concepts we typically care about. In other words, given a program written in terms of bits and instructions, it's very difficult to infer its purpose. Conversely, given an intuitive specification of a program's effects on the real world, it's very difficult to map this specification to a "bits and instructions" program.

To make this mapping easier, we build higher-level models atop this foundation: programming languages, operating systems, databases. Programming in terms of a higher-level model comes with a sacrifice: you give up control over how the program is "lowered" into lower-level terms. But with that loss of control comes a reduction in complexity, which is often a favorable trade. For example, garbage collection allows a programmer to not worry about deallocation, in exchange for giving up control over memory management.

Models form a partially-ordered hierarchy, with a model being *higher* than those it builds upon and *lower* than those that build upon it. But higher level models are not necessarily better suited for implementing a particular program. Ideally, a model's concepts correspond cleanly to those of the problem domain. Working within a **domain-aligned** model makes it easier to convert back and forth between requirements and implementation.

Much of the value of models comes from tooling. Tooling can help us ensure correctness, improve performance, and evolve our systems over time. But tooling works in terms of a specific model, and only has leverage over the concepts in that model. For example, consider what an OS-level tool like top can tell you about your program: resource consumption, uptime, network throughput, etc. It cannot do the things that are possible for a language-level tool like gdb, which works in terms of C's programming model.

But since tooling only helps within its model, if you frequently need to "drop down" to a lower level, you lose those benefits. The best higher-level models are ones where you rarely need to drop down. We call these models **sealed**: they provide an abstraction that [doesn't leak](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/) its internal details often. The modern world has many examples of ubiquitous, sealed models: it's rare to find programs written directly in assembly, that implement their own operating system, or that manage state without a database. Once a model becomes sealed, efforts bifurcate: some people develop programs in terms of that model, others develop programs that implement it.

This is the ideal: work within a sealed, domain-aligned model, and let tooling handle the boring stuff. But what happens when the system you're building doesn't fit within a single model?

## **Interoperability Causes Fragmentation**

Modern software systems are assembled from components: databases, caches, queues, services, frontends. In principle, this is empowering—you take components off the shelf, wire them together, and have a sophisticated system.

In practice, the process is often frustrating:

1. It's tedious. The job of so many software developers in the last decade has come to involve an inordinate amount of configuration management and quality assurance, at the cost of the creativity and ingenuity that attracted us to the field.  
2. It's inflexible. Once you've chosen some components and wired them together, changing the capabilities of your system is quite difficult, as modifying or swapping components is often very hard.  
3. It's error prone. Ensuring that they're wired together correctly is the developer's responsibility, with only limited tooling available to assist. Bespoke testing frameworks abound, but they invariably fall short.  
4. It's unperformant. Priorities are (rightly) driven by the need to minimize development cost and mitigate deployment risk. As a result, performance rarely receives much attention, and often degrades over the lifetime of a system.

So, the systems we build often end up brittle. But why? Is it a necessary consequence of building complex systems? We don't think so. We think it happens for a specific reason.

Each component has an *internal model*—the concepts it uses internally. But components also need to interact with each other, and often use a different, lower-level model for those interactions. A library interacts in terms of the same model as your code. A microservice exposing an API does not.

When we build a system out of components, the model we use to reason about the system is determined by these *interaction models*, not the internal models. When components use a lower-level model to interact, the whole system is forced down to that level. In internet software, systems are overwhelmingly forced into what we call the "networks and operating systems" model: computers, processes, memory, network addresses, packets. These are powerful abstractions, but they're far removed from what we actually care about. They work in terms of bytes and addresses, not objects, people, places, and actions.

For example, say we write a program and connect it to a relational database. The internal models of the program and database have clean, well-defined semantics, and they allow us to model our domain reasonably well. But the behavior of the system is not easily constrained by the semantics of either model. Instead, we have to think in terms of networks and operating systems to understand any problem that is not entirely contained to one of the components (e.g. "the server process crashed", "the data encoding is corrupted", "the connection was dropped").

There's a good reason so many components use a different interaction model than their internal model: interoperability. There are lots of models out there, with valuable components built using them. But most of those models are incompatible with each other. Components with incompatible internal models cannot interact directly—they must drop down to a lower-level, common model. This is why the "networks and OSes" model is ubiquitous: it's powerful, battle-tested, and sufficiently low-level that most components can build atop it. But achieving interoperability this way sacrifices the system-level benefits of working within a domain-aligned model.

## **The Costs of Fragmentation**

Let's call this kind of system a **fragmented** system. The distinguishing characteristic of a fragmented system is that it is assembled out of numerous components with incompatible internal models. Fragmented systems are brittle: they are hard to change and easy to break. In practice, that brittleness manifests in many ways.

*Contract Mismatches*

* Tweak the semantic meaning of an API field, downstream service still expects the old meaning—runtime error  
* Microservice A deploys v2, Microservice B still expects v1—runtime error  
* None of these are caught at compile time because the structure of the overall system isn't represented anywhere but runtime

*Cross-component Optimizations*

* "Push a filter down"—you want to fetch less data, but it requires changing the API contract at every layer between UI and database  
* "Reorder a join"—changing the order in which lookups are done can massively reduce processing, but might require moving logic between components in a very awkward way.  
* Move some logic from app to database (or vice versa)—rewrite in a different language, re-test, hope semantics match

*Ceremony and risk around changes*

* Database migrations: write SQL, write rollback SQL, coordinate deploy order, handle partial failures  
* Changing a shared data model: update schema, update every service, deploy in the right order and pray, or spend weeks testing with staging environments

*[Impedance Mismatches](https://en.wikipedia.org/wiki/Object%E2%80%93relational_impedance_mismatch)*

* The type systems of databases and programming languages are often incompatible, leading to subtle edge cases that are hard to test because they depend on the data actually stored in the database. Logic tests and data tests live in separate worlds even though they're fundamentally specifying requirements on the same program.  
* Your ORM makes relationships easy to traverse, but generates N+1 queries because it doesn't understand the database

These are symptoms. What is the underlying cause? In a fragmented system, the developer must reason about behavior in terms of a low-level interaction model. Components are not trivially composable—every time one is added or modified, the implications of that change on other parts of the system are not constrained by that component's internal model. They are only constrained by the interaction model, which is domain-misaligned, and therefore difficult to match to the system's requirements. The developer must carefully think through every change in those terms. Confidence that the system meets its requirements typically requires extensive validation.

Fragmented systems are intrinsically brittle. Good architecture and careful engineering can mitigate this somewhat, but without the ability to combine components within a single domain-aligned model, the tooling available to the system is fundamentally limited. The effort required to build and maintain a fragmented system scales unfavorably with its complexity.

## **Coherence Has Limits**

So fragmented systems are bad. What's the alternative? We call it a coherent system. A coherent system works entirely within a single, domain-aligned model. This constraint allows tooling to operate within that model across the whole system, creating major opportunities for verification, optimization, and automation.

There are many examples of models that enable coherent systems within specific domains:

* Type systems in programming languages catch many logic errors and interface misuses  
* The relational model in databases enables programmers to access incredible scale and performance with minimal effort.  
* Web frameworks like Rails, Express, and Django and Backends-as-a-Service like Firebase, Supabase, and Convex eliminate a great deal of the toil and errors of building web services.  
* Actor systems like Erlang or Ray make it easier to build certain kinds of distributed systems.  
* Durable execution systems like Temporal and ReState make fault tolerance more approachable.

When working entirely within models like these, the leverage of tooling is much greater. Programmers often get big boosts in their productivity, and the coherent systems they build often have better correctness and performance than comparable fragmented systems.

So coherent systems are great: everyone should just buy into whatever model will most effectively do the job. Right? Unfortunately, the listed models are all domain-specific–they don't generalize to other contexts. And most modern internet software is not domain-specific. Modern applications typically span a wide variety of domains, including web and API serving, transaction processing, background processing, analytical processing, and telemetry. That means that trying to keep a system coherent limits what your system can ultimately do. Even as we try to keep our system coherent, application requirements push us outside of a single domain, forcing us to reach for components with a different internal model. So, step by step, our system fragments.

The industry's response to this situation has been to accept fragmentation as inevitable. "Use the right tool for the job," they say. Each domain gets its own specialized component, and we'll wire them together. This is pragmatic advice—it reflects reality. But it also encodes a hidden assumption: that fragmentation is an acceptable cost, that we can't do better. We reject that assumption.

## **Generality Without Fragmentation**

But rejecting an assumption isn't the same as proving it wrong. Is a general-purpose model, aligned with the domains of internet software, actually possible? If it were, wouldn't one already exist? You might speculate that there's an inherent tradeoff between generality and how domain-aligned a model can be. “Domain-aligned” and “domain-specific” sound suspiciously similar, don’t they? Empirically, we do observe such a trend. But it doesn't seem strictly necessary. Many of the models we've discussed are both general-purpose and sealed. The C language exists in the stack of nearly all modern software. Linux is ubiquitous, only inappropriate in rare circumstances like hard-real-time and safety-critical systems. Relational databases are nearly as ubiquitous, with NoSQL DBs comprising only a small proportion of usage and applications only rarely needing to reach for models below that level. And sometimes the tradeoff gets pushed out entirely. Rust is more general-purpose than C, aligns with more domains, and has better tooling—without giving up performance. These innovations are rare, but they are possible.

So let's imagine one more such innovation. If we could build a sealed model that were general-purpose across *and* aligned with the domains typically required to build internet software, coherent systems could be built atop it without being fenced in to a single narrow domain. This would create tremendous opportunities for tooling:

* Development would be accelerated dramatically by enabling components to interact directly in terms of a domain-aligned model.  
* System-wide verification would become tractable for most applications.  
* Performance tooling could profile applications and apply optimizations automatically across the whole system.  
* Operational tooling could instrument, monitor, and orchestrate services with minimal setup and oversight.

If realized, these opportunities have the potential to revolutionize the development of internet software.

This is what we're trying to implement: a general-purpose, domain-aligned, sealed model for internet software. It's a bet against the conventional wisdom that says "use the right tool for the job" and simply accepts the resulting fragmentation. We believe coherence doesn't have to be sacrificed for generality—and that the payoff for achieving both is immense.

Of course, we're not the first to attempt this. Many have tried to build general-purpose, domain-aligned models for software development. Most sacrifice domain alignment to achieve generality, which is the tradeoff we've focused on in this essay. Some sacrifice generality to achieve coherence, such as Erlang, Smalltalk, and Prolog. The ones that achieved both failed to become sealed—they leaked too often to displace lower-level alternatives. Memorable attempts include object-oriented databases and J2EE. So why do we think we can succeed? We'll share more about our approach in the future. For now, we'll just say: we believe advances in programming language theory and database systems have opened a path that wasn't available before.

---

## **Postscript: What About AI?**

That's the core argument. But there's a question we expect many readers are already asking: doesn't AI change everything? Why worry about models and coherence when agents can just handle the complexity for us?

We've been speaking of tooling in the abstract, and using examples of traditional tooling. But the most powerful form of tooling emerged only recently in the form of agentic AI. Agents are much more flexible than previous forms of tooling, and they have tremendous potential for improving the productivity of developers. They already represent a revolution in software development.

Given this, there's a popular narrative that AI itself is sufficient to realize the opportunities referenced above. In this narrative, code is merely a by-product, an intermediate representation of the programmer's intent, the truth of which is captured by the prose prompts and documents that are provided to the agents. The agents will handle all of the complexity of code. We don't need to worry about abstractions or maintainability. Those are vestigial concerns from an era that will soon be past. AI is the future, and traditional software engineering will soon be obsolete.

We believe this narrative builds on several fundamental misconceptions of software.

The first misconception is the conflation of ambiguity and abstractness. People often think of code as "low level", when what they really mean is that it’s domain-misaligned. But this is not an inherent property of all code. It's a property of the model being used, relative to the problem being solved. Some models are lower-level, some are higher-level. Code can be either. What code is truly about is precision: code has unambiguous semantics. It's easy to conflate ambiguity and abstraction—both involve "a single statement that could refer to multiple meanings." But the meanings of an ambiguous statement are unconstrained. The meanings of an abstract statement are tightly constrained by the semantics of the model. Prose is ambiguous, and always will be. Fluidity of meaning is core to its utility. Code is precise. Precision is core to its utility. Whether the one reading and writing that code is a human or AI may shift over time, but code will never be obviated by prose because precision will always be important when designing a complex system.

The second misconception relates to the alignment between models and domains. Even when communicating in prose, programmers and AI still need shared concepts to reason about. Models will always be important for that—they give concepts meaning. Those models must align with the domains of the problems we seek to solve. If they don’t, problems quickly become intractable. That's what the difference between fragmented and coherent systems is all about: using a single, domain-aligned model to make problems tractable. That distinction will remain relevant, no matter who is building the system.

The third misconception relates to the value and rarity of good models. Some would argue that AI will get good enough to shield programmers from the problem of domain misalignment. Once that happens, it seems that the value of domain-aligned models disappears. The problem with this view is that it doesn’t engage with what must be happening *inside* the AI for this to happen. Shielding a programmer from domain misalignment is a very hard problem, and the best way to do it is to invent a sealed, domain-aligned model, and present that new model to the programmer. An AI armed with such a model will outperform an AI without one. Inventing such models is hard, and those models are valuable once found. So the pursuit and usage of sealed, domain-aligned models will continue, regardless of whether humans or AIs are the ones pursuing them.

With a more nuanced understanding than the popular narrative, we are led to very different conclusions. AI is powerful, yes, and it will likely keep getting more powerful. But while AI may be able to reason faster or remember more than a human, some constraints apply universally: the problem space of the universe is too large to explore by brute force. Intelligence consists in finding good models with which to understand and manipulate the universe. In software, we have access to many powerful models, but there are obvious opportunities for improvement. The discovery and implementation of useful new models will give AI superpowers, just like it does us.

There is already substantial evidence in support of this view. Look no further than agentic coding. If you unleash even our most powerful AI agents in a complex, poorly-structured codebase, it tends to make a mess. Agents are most powerful when working within narrow environments with clear rules, such as:

* Stateless, single-page javascript apps  
* Standard 3-tier architectures using a good web framework  
* Clearly-articulated analytics tasks within a single, well-documented data warehouse

Naturally, the capabilities of AI systems will continue growing. Consequently, so will the extent of the domains they can excel within. But there will always be room for innovations that make AI more productive. AI makes it easier to work within models, but it is not an excuse to stop innovating on the models themselves. Right now, we badly need a programming model that will enable the development of coherent internet applications. For now, AI can help, but it won’t do it for us. Let’s go build it.

