# MisActBench

The benchmark data (including trajectory screenshots and annotations) is hosted on HuggingFace:

👉 **[Download from HuggingFace](https://huggingface.co/datasets/osunlp/MisActBench)**

After downloading, place the files under this directory so that the structure looks like:

```
MisActBench/
├── README.md
├── misactbench.json
└── trajectories/
    ├── <trajectory_id>/
    │   ├── step_0_*.png
    │   ├── step_1_*.png
    │   └── ...
    └── ...
```

Then you can run DeAction following the instructions in the [main README](../README.md).
