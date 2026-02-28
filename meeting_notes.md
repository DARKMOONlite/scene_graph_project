# Meeting Notes:

## 14/2/26
### Things I've done:
- Tested both anyBURL2023  and AMIE2020 on the visual genome database as well as a subset of the Freebase Dataset.
	- AMIE seemed to do a better job at generating useful rules
	- a lot of anyBURL rules were defining simple linquistical relationships like Next to is related to Near
	- AMIE did better but I did let it run longer, as the anyBURL java script stops after 100s.
	- A lot of overlap, if there are 100 object classes then it needed 100 lines for each basic common relationship
		- 2 Kinds of rules being made really:
			- Loop Resolution basically Bigger(A,B), Bigger (B,C) -> Bigger(A,C)
			- Generalisation if all instances of musicians have made music then maybe all musicians make music. 
- Tried testing AMIE and anyBURL on the CLEVR database, but they don't really work as they cannot do numerical comparisons, thus most of the benefits of this architecture would be mute
	- ClEVR has position data and relates those to predicates like Near/Next-to Behind
	- for now we can just hard code and abstract away to these predicates, but in the future it would probably be better for the system to be able to intake numerical numbers

- There are major differences between scallop rules and FOL rules
	- Numerical Comparisons
		- Bigger(A,B), Bigger (B,C) -> Bigger(A,C) vs Height(A,h1) and Height(B,h2) and h1>h2 => Bigger(A,B)
	- FOL can be collated into a graph
	- you technically can translate FOL into Scallop, but you lose a lot of the benefits of scallop. 
TODO:
- Apply the concept behind AMIE and/or AnyBURL to scallop rules, find a way to incorporate numerical comparisons into the test scope
- We need the system to focus on Generisations, to do that we either need to teach it how it can generalise, or find a big dataset of objects and how they're related:
	- e.g. Present -> Box -> Container, Bird -> Animal	

### Meeting Notes
Domain Specific Logic is based on some small commonsense knowledge


`"Grounding of Concepts"` : Standardise some basic concepts from which all the commonsense Knowledge is based off of.
- relationship between the logical world and the geometrical world
	- "A drone is “near” the landing point when it is within about 1–3 meters horizontally and 1–5 meters vertically of the touchdown location."
	- grounding must be probabilistic, and cannot have hard cut offs. 
"Knowledge of Concepts": Not grounded. 

Focus on the taxonomy of rules based on how specific they are or where they come from. - Ivan
#### TODO:
1. Focus on creating the commonsense scallop rules
2. create LLM based Human in the loop Domain Specific scallop rules
3. create Grounding of Concepts

## 28/2/26
### Things I've done
- Created a basic predicate grounding system to take images and create simple grounded predicates based on that. e.g. `Near to`, `Beneath`, `inside` etc.
  - Currently this is not the best solution as its hard coded based on the centroids and edges of the generated Segment Anything masks.
  <img width="1232" height="905" alt="Screenshot from 2026-02-23 13-39-40" src="https://github.com/user-attachments/assets/8339c9ae-1992-4eca-8eaf-e610127d59af" />
- Predicate Enrichment:
	**Predicate enrichment** is the act of taking simple incomplex predicates and using additional context to convert them into more interesting predicates. The goal should be to take our concept-grounded predicates which are defined by experts in the field, and convert them using aditional information from the scene into more complex and useful predicates. 
  - **Predicate Complexity**: we want to create rules that are more complex than grounded rules, each predicate should be given a complexity value based on either how difficult it is to discern or how much and what type of contextual information is needed to create it
  - **Generalisability**: we want to create rules that are abstract enough and cover as much of the search space as possible, we should focus at the start on creating highly generalised rules and then slowly increase the specificity of the generated rules. 

### Meetings Notes
Contextual awareness being added to promote low level predicates into high level predicates.
Low (Abstract) High Level (Task/Scene Specific)
- we need aditional information
We need to get this information.
- How can we get this additional information or Entropy.

- Write down what roadblocks ive met and how ive moved around or overcome them.
- Predicates are more textual and rich than just the word itself, we need to capture these ideas.
- What levels NEED to be symbolic
	- Can we obtain the rules using 


We need to define the meaning of predicates, then we can define the rules
- basic rules can be easily defined but complex prediates
- `Using the Computer` is a composite rule with lots of conditions, 
- Using Neural networks to learn rules
- 
Domain Context is always needed.
- Different Learning Strategies
- Top down AND bottom up
  - Bottum up ones can be symbolic,
- PredicateConcept Allignment
  - `Looking at screen`, `Touching Keyboard`, `touching mouse` all conceptually alligned, then combined symbolically to create `Person using Computer(x,y)`
    - with this someone would be able to look at `Person using a computer` and see why the system thought that, if the system is wrong in some of the conceptually alligned predicates: `looking at screen` or `touching mouse` the human should be able to look at 
-  Sandwich method Symbolic -> Neural -> Symbolic
-   Neural Tightening / Neural Loosening, where is the Neural Crux

```
 Using_computer(x,y)
```
  
