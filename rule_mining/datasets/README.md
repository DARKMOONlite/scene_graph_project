# Rule Learning

current systems for rule learning often take the form of rule mining where they "mine" patterns from knowledgebases like knowledge graphs. These systems however create relational atoms only, and cannot do arithmetic comparisons which I believe is necessary for the system to correctly determine facts about people.

The following example determines a relationship between two people that a human can easily create, however First Order Logic cannot create as it cannot arithmetically compare `X` & `Y`.
```prolog
    Taller(Mark,Steph) <= Height(Mark,X) Height(Steph,Y), X>Y
```
instead these FOL systems can only solve holes in existing data for example
```prolog
Taller(Alice, Bob) & Taller(Bob, Charlie) => Taller(Alice, Charlie)
```
### Abstraction Alternative (Grounding of Concepts)
An alternative is instead of passing in positional/numeric data into scallop we instead abstract away the numbers to basic predicates. 
```prolog
distance < 1.0 -> Close(X,Y)
1.0 - 3.0 -> Near(X,Y)
>3.0 -> Far(X,Y)
```
This is what we're going with for the first component of the project. 