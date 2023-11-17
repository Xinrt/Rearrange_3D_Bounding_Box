# Rearrange_3D_Bounding_Box

## Task
### 11.14 - 11.22
For all :
- [ ] Install the AI2THOR rearrangement and mass Github repo.
        https://github.com/allenai/ai2thor-rearrangement/
        https://github.com/brandontrabucco/mass/
- [ ] Find potential 3D object bounding box code candidates.

Name 1: 
- [ ] Learn how to use the dataset: train, val, test. https://github.com/allenai/ai2thor-rearrangement/tree/main/data/2023

### 11.22 - 12.3
Name 2,3: 
- [ ] Determine a working SOTA 3D object bounding box codebase.
- [ ] Integrate the 3D object bounding box code to our project.

Name 1:
- [ ] Prepare the code structure for our project.

### 12.3 - 12.16
- [ ] Write report.
- [ ] Run the experiment tests.
- [ ] (If possible) Add 3D object bounding box code to MaSS.

## Run
```
export PYTHONPATH=$PYTHONPATH::/path/to/ai2thor-rearrangement
export PYTHONPATH=$PYTHONPATH::/path/to/Rearrange_3D_Bounding_Box
```


```
python -u agent.py \
--logdir ./testing-the-agent --stage val \
--semantic-search-walkthrough \
--semantic-search-unshuffle \
--use-feature-matching \
--start-task 0 --total-tasks 20
```




## References
### 3D object Oriented Bounding Box
1. [Canonical Voting: Towards Robust Oriented Bounding Box Detection
in 3D Scenes](https://arxiv.org/pdf/2011.12001.pdf): https://github.com/qq456cvb/CanonicalVoting

