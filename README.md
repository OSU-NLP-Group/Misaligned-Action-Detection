# Misaligned Action Detection in CUAs

Code and data for the paper "When Actions Go Off-Task: Detecting and Correcting Misaligned Actions in Computer-Use Agents".

<p align="center">
[<a href="https://osu-nlp-group.github.io/Misaligned-Action-Detection">Website</a>] •
[<a href="https://arxiv.org/abs/2602.08995">Paper</a>] •
[<a href="https://huggingface.co/datasets/osunlp/MisActBench">Data</a>]
</p>

## 🗂️ MisActBench

### Download

The benchmark data is hosted on HuggingFace. Please download it and place the files under `MisActBench/`:

```bash
# Make sure you have git-lfs installed
git lfs install
git clone https://huggingface.co/datasets/osunlp/MisActBench MisActBench
```

After downloading, the directory structure should look like:

```
MisActBench/
├── misactbench.json
└── trajectories/
    ├── <trajectory_id>/
    │   ├── step_0_*.png
    │   ├── step_1_*.png
    │   └── ...
    └── ...
```

### Data Format

`MisActBench/misactbench.json` is a JSON object keyed by `trajectory_id` (UUID). Each entry has the following fields:

- `trajectory_id`: UUID for the trajectory.
- `trajectory_path`: Relative path to the trajectory folder (e.g. `MisActBench/trajectories/<uuid>`).
- `instruction`: User instruction for the trajectory.
- `total_steps`: Total number of steps.
- `steps`: A dict keyed by step number. Each step contains:
  - `step_idx`: Step index (integer).
  - `label`: `true` (misaligned), `false` (aligned), or `null` (not annotated).
  - `category`: Misalignment category (`"Malicious instruction following"`, `"Unintended harmful behavior"`, `"Other task-irrelevant behavior"`, or `null`).
  - `agent_output`: The agent's generated code/action for this step.
  - `screenshot_path`: Relative path to the screenshot before this step's action.
- `metadata`: `{ "source": ..., "agent": ... }`.

## 🛡️ DeAction

### Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Then set the environment variables for your API provider:

- Azure OpenAI: `AZURE_API_KEY`, `AZURE_API_VERSION`, `AZURE_ENDPOINT`
- OpenAI: `OPENAI_API_KEY`
- AWS Bedrock (Claude): `AWS_REGION`, `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`
- Local server: optional `LOCAL_OPENAI_BASE_URL` (defaults to `http://localhost:8000/v1`)

### Run

Basic command:

```bash
python DeAction/run.py \
  --benchmark_file MisActBench/misactbench.json \
  --result_file output/deaction_results.json \
  --systematic_analysis_model <MODEL> \
  --fast_check_model <MODEL> \
  --narrative_summary_model <MODEL> \
  --annotate_actions
```

Models are specified as `provider|model` (e.g., `openai|gpt-4o`, `azure|gpt-4o`, `aws|claude-3-7-sonnet`, `local|<model>`). The provider is inferred from the prefix before `|`.

Common optional flags (see `DeAction/run.py` for full list):
- `--model_reasoning_effort <LEVEL>`: reasoning effort for the systematic analysis model (if supported).
- `--fast_check_reasoning_effort <LEVEL>`: reasoning effort for the fast-check model (if supported).
- `--max_workers <N>`: number of worker threads to run trajectories in parallel (default: 4).
- `--narrative_summary_cache_file <PATH>`: JSON cache for narrative summaries to reuse across runs.
- `--no_narrative_summary`: disable narrative summary generation (enabled by default).

## 📮 Contacts
[Yuting Ning](mailto:ning.151@osu.edu), [Huan Sun](mailto:sun.397@osu.edu)

## 📝 Citation Information
If you find this work useful, please consider citing our paper:
```
@misc{ning2026actionsofftaskdetectingcorrecting,
      title={When Actions Go Off-Task: Detecting and Correcting Misaligned Actions in Computer-Use Agents}, 
      author={Yuting Ning and Jaylen Jones and Zhehao Zhang and Chentao Ye and Weitong Ruan and Junyi Li and Rahul Gupta and Huan Sun},
      year={2026},
      eprint={2602.08995},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.08995}, 
}
```
