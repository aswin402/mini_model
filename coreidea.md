# LITTLE — Core Idea

**Version:** 0.1

---

## 1. The central idea

LITTLE starts from one question:

> What if an AI did not need to contain all of its knowledge inside one enormous neural network?

Instead, we want a system with:

- a relatively small computational core;
- persistent memory;
- explicit concepts;
- relationships between concepts;
- experiences;
- procedures;
- uncertainty;
- continual learning.

The system should grow its knowledge through experience.

---

## 2. LLM versus LITTLE

A simplified LLM-centric approach is:

```text
huge dataset
     ↓
large training process
     ↓
large parameter set
     ↓
prompt
     ↓
generated response
```

LITTLE aims for:

```text
experience
    ↓
observation
    ↓
concept formation
    ↓
memory
    ↓
reasoning
    ↓
action/answer
    ↓
feedback
    ↓
new experience
```

The comparison is not meant to claim that all LLMs work identically. It is a conceptual distinction between a large pretrained knowledge system and a continually learning cognitive architecture.

---

## 3. The unit of learning

The primary unit is the **concept**.

Example:

```text
APPLE
```

is not just the string "apple."

It may contain:

```text
APPLE
├── category -> FRUIT
├── edible -> true
├── colors -> {RED, GREEN, YELLOW}
├── has_part -> SEED
├── has_part -> STEM
└── examples -> [...]
```

The concept evolves as evidence accumulates.

---

## 4. Experience versus knowledge

Suppose a user says:

> This apple is green.

LITTLE should remember the event:

```text
Experience:
    apple_instance_12
    color = green
```

It may then update generalized knowledge:

```text
APPLE can_have_color GREEN
```

Those are different things.

The first is an experience.

The second is an abstraction.

---

## 5. Generalization

Generalization should come from identifying invariant structure.

Example:

```text
red apple
green apple
sliced apple
small apple
large apple
```

Some properties vary:

```text
color
size
state
orientation
```

while others may remain more stable.

The system should learn which properties are relevant to identity instead of assuming every observed feature defines the concept.

This is a research problem, not something we should hard-code as a universal rule.

---

## 6. Unknown is a valid state

One of LITTLE's strongest requirements is:

> Not knowing is better than inventing evidence.

The system should distinguish:

```text
I know.
I have evidence.
I am uncertain.
I have conflicting evidence.
I don't know.
```

Example:

```text
Question:
Is a kiwi an apple?

Knowledge:
APPLE -> FRUIT
KIWI -> FRUIT

No identity relationship.

Answer:
UNKNOWN / NOT SUPPORTED
```

If later taught:

```text
KIWI is not APPLE
```

that becomes explicit negative evidence.

---

## 7. Memory should be inspectable

We should be able to inspect:

```text
What does LITTLE know about DOG?
```

and see:

```text
DOG

is_a:
    ANIMAL

properties:
    four_legged
    domestic

evidence:
    experience_12
    experience_37

confidence:
    0.94
```

This makes the architecture scientifically useful.

---

## 8. Learning should be local when possible

If the system learns:

```text
BICYCLE is a VEHICLE
```

we should update the relevant knowledge structures.

We should not automatically retrain every learned component.

This is intended to reduce:

- compute;
- forgetting;
- training cycles;
- opacity.

It may not always be possible. Some neural representations will eventually need adaptation. That should be measured rather than assumed.

---

## 9. Concepts form a world model

As concepts accumulate:

```text
ANIMAL
├── DOG
├── CAT
└── HORSE

FRUIT
├── APPLE
├── BANANA
└── KIWI

VEHICLE
├── CAR
├── BICYCLE
└── BUS
```

The system begins constructing a structured model of its domain.

Relationships provide more information than isolated facts.

---

## 10. Skills are learned procedures

A fact:

```text
2 + 3 = 5
```

is different from a skill:

```text
ADD(a, b)
```

A skill describes how to transform inputs into outputs.

Programming will later exploit this distinction:

```text
problem
 ↓
plan
 ↓
program
 ↓
execute
 ↓
observe
 ↓
correct
```

---

## 11. Self-directed learning

The eventual learning loop:

```text
I observe something.
        ↓
I compare it with what I know.
        ↓
I find uncertainty.
        ↓
I decide what information would reduce uncertainty.
        ↓
I obtain evidence.
        ↓
I update my model.
        ↓
I remember what happened.
```

This is the long-term direction.

---

## 12. Why start with English?

English gives us an easy-to-inspect environment.

We can write:

```text
A dog is an animal.
```

and directly inspect:

```text
DOG --is_a--> ANIMAL
```

That makes it ideal for testing:

- memory;
- concepts;
- relationships;
- reasoning;
- uncertainty;
- continual learning.

Once the mechanism works, we can apply the same architecture to mathematics, programming, and vision.

---

## 13. Mathematics as procedural reasoning

Mathematics will test whether the architecture can learn transformations rather than memorize answers.

Example:

```text
2 + 3 = 5
```

should eventually produce a reusable operation.

Then:

```text
17 + 29
```

can be solved even though that exact pair was never observed.

---

## 14. Programming as action + feedback

Programming provides a natural learning environment.

The system can:

```text
write code
   ↓
execute code
   ↓
observe result
   ↓
compare expected/actual
   ↓
debug
   ↓
store successful strategy
```

This gives LITTLE a rich feedback loop.

---

## 15. Vision as concept invariance

The apple example becomes:

```text
Learn:
red apple

Observe:
green apple

Question:
same concept?
```

The architecture should compare representations and concept structures.

Later:

```text
whole apple
half apple
sliced apple
apple under different lighting
```

should test invariance.

---

## 16. What LITTLE is not

LITTLE is not initially:

- a general chatbot;
- a replacement for all foundation models;
- an attempt to train a giant model from scratch;
- a database with a chat interface;
- a collection of prompts around an LLM.

It is an experiment in **continual concept-centered learning**.

---

## 17. The ultimate hypothesis

Our hypothesis is:

> A system with explicit concepts, persistent episodic/semantic/procedural memory, uncertainty-aware inference, and continual learning may acquire useful domain competence more efficiently and transparently than repeatedly retraining a monolithic model for every new piece of knowledge.

This is a hypothesis.

We must test it.

---

## 18. Scientific attitude

LITTLE should be willing to prove itself wrong.

If an experiment shows that a simpler neural model performs better, record it.

If an LLM component solves a problem better, record it.

If explicit symbolic memory becomes a bottleneck, change it.

The goal is not to defend an architecture.

The goal is to discover an architecture that works.
