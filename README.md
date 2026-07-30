# scene_graph_project
A project for creating useful spatial scene graphs from images. 


## Data sets
### Tested:
- [**NuScenes**](https://www.nuscenes.org/): `Tested`: A driving centric dataset containing multiple camera perspectives and highly annotated images.
- [**Visual Genome**](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html):`Tested` an ongoing effort to connect structured image concepts to language

### Untested:
- [**ConceptNET**](https://conceptnet.io/): semantic network, designed to help computers understand the meanings of words that people use.
- [**CLEVR**](https://cs.stanford.edu/people/jcjohns/clevr/): a diagnostic dataset that tests a range of visual reasoning abilities
- [**Ego4D**](https://ego4d-data.org/): Driving dataset
- [**BDD-X**](https://github.com/JinkyuKimUCB/BDD-X-dataset): Driving Dataset



## Installation

```bash

pip install .

```

## Usage

### Fusion

 ```bash
python3 scene_graph_project/scene_graph_fusion/fuse_scene_graphs.py \
 {any number of folders}\
 -o {output folder}\

 ```

### Stabalise Nuscenes 
> [!NOTE]
> this requires the nuscenes dataset loaded into a sqlite3 db. I've added the databasemanager class and some utility to make it easier to use.

```bash
python3 scene_graph_project/scene_graph_fusion//stabalise_scene_graphs.py \
--save -o /mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs/stabalised_graphs_action2 \
 --samples-folder /mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs/samples/merged/ \
  --sweeps-folder /mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs/sweeps/merged/
```

### View Scene Graph

```bash
python3 scene_graph_project/scene_graph_fusion/view_scene_graphs.py {scene graph path}
```

## Interesting Links
[**Scallop Interpreting over Graphs**](https://www.scallop-lang.org/log22/slides/log-2022-tutorial-scallop-part.pdf)


## TODO

#### - Conversion of Ego-centric, Allo-centric and Exo-centric scene graphs.
   - [VLM-Grounder](https://github.com/InternRobotics/VLM-Grounder)
   - [VIZOR](https://vivekmadhavaram.github.io/vizor/)
   - [Vil3DRel](https://arxiv.org/pdf/2211.09646)

#### - Scene Graph specific MOT (Multi Object Tracking)




