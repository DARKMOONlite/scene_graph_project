# scene_graph_project
A project for creating useful spatial scene graphs from images. 




## Data sets

- [**Visual Genome**](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html): an ongoing effort to connect structured image concepts to language
- [**ConceptNET**](https://conceptnet.io/): semantic network, designed to help computers understand the meanings of words that people use.
- [**Atomic20/20**](https://github.com/allenai/comet-atomic-2020):
- [**CLEVR**](https://cs.stanford.edu/people/jcjohns/clevr/): a diagnostic dataset that tests a range of visual reasoning abilities
## Rule Mining Links

- [**SAFRAN**](https://github.com/OpenBioLink/SAFRAN): Scalable and fast non-redundant rule application
- [**AnyBURL**](https://web.informatik.uni-mannheim.de/AnyBURL/): rule learner AnyBURL (Anytime Bottom Up Rule Learning). AnyBURL has been designed for the use case of knowledge base completion, however, it can also be applied to any other use case where rules are helpful.
- [**pyClause**](https://github.com/symbolic-kg/PyClause): a library for easy and efficient usage and learning of symbolic knowledge graph rules
```bash
pip install git+https://github.com/symbolic-kg/PyClause.git
```


## Setting up python

```bash
#create virtual environment
python3 -m venv .venv
. .venv/bin/activate
#install dependencies
pip install -e . 
```
