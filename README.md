# Uncertainty-Aware Inference
### How Quantization Affects LLM Confidence Calibration

**Columbia University EECS 6895 — Advanced Big Data and AI**  
**Mentor: Ruchi Muhindru, Distinguished Engineer, IBM Research**

---

## Overview

Quantization dramatically reduces LLM inference cost — but does it corrupt
the model's confidence? A well-quantized 7B model that preserves top-1 accuracy
may still produce overconfident wrong answers and underconfident correct ones,
because quantization distorts the full output probability distribution in ways
that accuracy alone fails to detect.

This project systematically studies **how post-training quantization affects LLM
confidence calibration** across three model families and five precision configs,
and whether **knowledge distillation** from an FP16 teacher can recover
lost calibration quality.

**Teams and models:**

| Team | Model        | Specialty                          |
|------|--------------|------------------------------------|
| A    | Llama-2 7B   | Calibration pipeline + KD          |
| B    | Mistral 7B   | GPU profiling + vLLM serving       |
| C    | Llama-2 13B  | Pareto analysis + routing          |

---

## Research Questions

1. Does PTQ degrade calibration beyond what accuracy metrics detect?
2. Is calibration degradation consistent across model families and methods?
3. Can knowledge distillation from an FP16 teacher recover calibration in an INT4 student?
4. What is the Pareto-optimal configuration balancing cost, accuracy, and calibration?
5. Can uncertainty-based routing cheaply serve most queries while preserving calibration?

---

## Repository Structure

```
uncertainty-aware-inference/
│
├── src/
│   ├── calibration/
│   │   ├── metrics.py              # ECE, MCE, Brier, LLM log-likelihood scoring
│   │   ├── plots.py                # Reliability diagrams, entropy plots, dashboards
│   │   ├── datasets.py             # ARC-Challenge, HellaSwag, TriviaQA loaders
│   │   └── temperature_scaling.py  # Post-hoc recalibration baseline (Guo et al.)
│   │
│   ├── quantization/
│   │   └── loaders.py              # Unified loader: FP16, GPTQ_INT8/4, AWQ_INT4, NF4
│   │
│   ├── distillation/
│   │   └── trainer.py              # KD loss (T² scaling), training loop, recovery metrics
│   │
│   ├── profiling/
│   │   └── harness.py              # Latency, memory, PyTorch Profiler, Roofline model
│   │
│   ├── analysis/
│   │   └── pareto.py               # Pareto dominance, routing simulation, cross-model plots
│   │
│   └── utils/
│       └── logging.py              # Logging + W&B helpers
│
├── scripts/
│   ├── run_ptq_sweep.py            # PTQ calibration sweep (all teams)
│   ├── run_kd.py                   # Knowledge distillation (Team A)
│   ├── run_profiling.py            # GPU profiling + Roofline (Team B)
│   ├── run_vllm_serving.py         # vLLM throughput benchmark (Team B)
│   ├── run_analysis.py             # Pareto + routing simulation (Team C)
│   └── run_qat.py                  # QAT stretch goal (Team A, optional)
│
├── configs/
│   └── models.yaml                 # All model IDs, dataset configs, hyperparameters
│
├── tests/
│   ├── test_calibration.py         # Unit tests for ECE, Brier, temperature scaling
│   └── test_analysis.py            # Unit tests for Pareto dominance, routing simulation
│
├── Makefile                        # Convenience targets for the full pipeline
├── requirements.txt
├── setup.py
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-org/uncertainty-aware-inference.git
cd uncertainty-aware-inference
pip install -r requirements.txt
```

### 2. HuggingFace authentication (required for Llama-2)

```bash
huggingface-cli login
# Accept the Llama-2 license at https://huggingface.co/meta-llama/Llama-2-7b-hf
```

### 3. W&B setup (optional but recommended)

```bash
wandb login
# Update configs/models.yaml → project.wandb_entity with your team name
```

### 4. Verify installation

```bash
pytest tests/ -v
```

---

## Usage

### Quick start with Makefile

```bash
make install          # install dependencies
make test             # run unit tests

make sweep-a          # Team A: Llama-2 7B calibration sweep
make sweep-b          # Team B: Mistral 7B calibration sweep (parallel)
make sweep-c          # Team C: Llama-2 13B calibration sweep (parallel)

make profile          # Team B: GPU profiling + Roofline
make kd               # Team A: knowledge distillation
make analysis         # Team C: Pareto + routing analysis

make all              # full pipeline in sequence
```

---

### Phase 1–2: PTQ Calibration Sweep

Each team runs this for their assigned model:

```bash
# Team A: Llama-2 7B
python scripts/run_ptq_sweep.py \
    --model_id   meta-llama/Llama-2-7b-hf \
    --model_name Llama-2-7B \
    --output_dir results/llama2_7b \
    --n_samples  500 \
    --precisions FP16 GPTQ_INT8 GPTQ_INT4 AWQ_INT4 NF4 \
    --datasets   arc_challenge hellaswag triviaqa

# Team B: Mistral 7B
python scripts/run_ptq_sweep.py \
    --model_id   mistralai/Mistral-7B-v0.1 \
    --model_name Mistral-7B \
    --output_dir results/mistral_7b \
    --n_samples  500

# Team C: Llama-2 13B
python scripts/run_ptq_sweep.py \
    --model_id   meta-llama/Llama-2-13b-hf \
    --model_name Llama-2-13B \
    --output_dir results/llama2_13b \
    --n_samples  500
```

**Outputs per config:**
- `{ModelName}_{Precision}_{Dataset}_calibration.json` — ECE, MCE, Brier, accuracy, bin data
- `plots/{tag}_reliability.png` — reliability diagram
- `plots/{tag}_dashboard.png` — 4-panel calibration dashboard

---

### Phase 2: GPU Profiling (Team B)

```bash
# Profile all three models
python scripts/run_profiling.py \
    --model_id   meta-llama/Llama-2-7b-hf \
    --model_name Llama-2-7B \
    --output_dir results/profiling \
    --precisions FP16 GPTQ_INT4 AWQ_INT4 NF4
```

**vLLM Serving Benchmark:**

```bash
# Step 1: Print launch commands
python scripts/run_vllm_serving.py \
    --model_id meta-llama/Llama-2-7b-hf \
    --model_name Llama-2-7B \
    --print_commands

# Step 2: Start each server in a separate terminal (copy commands from above)
# Step 3: Run the benchmark
python scripts/run_vllm_serving.py \
    --model_id   meta-llama/Llama-2-7b-hf \
    --model_name Llama-2-7B \
    --output_dir results/vllm
```

**Outputs:**
- `{ModelName}_profiling.json` per config — latency, memory, Roofline metrics
- `{ModelName}_roofline.png` — Roofline plot
- Chrome profiler trace JSON (open in `chrome://tracing`)
- `{ModelName}_vllm_results.json` — throughput sweep data

---

### Phase 3: Knowledge Distillation (Team A)

Run *after* the PTQ sweep has produced FP16 and GPTQ_INT4 calibration JSONs.

```bash
python scripts/run_kd.py \
    --model_id          meta-llama/Llama-2-7b-hf \
    --model_name        Llama-2-7B \
    --student_precision GPTQ_INT4 \
    --results_dir       results/llama2_7b \
    --output_dir        results/llama2_7b/kd \
    --n_kd_samples      2000 \
    --n_epochs          3
```

**Outputs:**
- `{ModelName}_{Precision}_KD_{Dataset}_calibration.json` — post-KD calibration metrics
- `plots/kd_recovery.png` — FP16 vs pre-KD vs post-KD bar chart
- `kd_recovery.json` — fraction of calibration quality recovered per metric

---

### Phase 3: Cross-Model Pareto + Routing (Team C)

Run *after* all three PTQ sweeps and profiling are complete.

```bash
python scripts/run_analysis.py \
    --calibration_dirs results/llama2_7b results/mistral_7b results/llama2_13b \
    --profiling_dir    results/profiling \
    --output_dir       results/analysis \
    --dataset          arc_challenge
```

**Outputs:**
- `pareto_front.json` — Pareto-optimal configurations
- `plots/pareto_*.png` — 2D Pareto plots (latency×acc, ECE×acc, memory×ECE, tps×ECE)
- `plots/pareto_3d.png` — 3D Pareto frontier (cost × accuracy × calibration)
- `plots/heatmap_*.png` — cross-model metric heatmaps
- `plots/routing_*.png` — routing simulation per precision
- `routing_summary.json` — optimal thresholds and projected cost savings

---

### Stretch Goal: QAT (Team A, optional)

Requires ~20 GB VRAM and 8–12 GPU hours.

```bash
python scripts/run_qat.py \
    --model_id        meta-llama/Llama-2-7b-hf \
    --model_name      Llama-2-7B \
    --results_dir     results/llama2_7b \
    --output_dir      results/llama2_7b/qat \
    --n_train_samples 2000 \
    --n_epochs        3
```

**Outputs:**
- `QAT_NF4_LoRA_{Dataset}_calibration.json`
- `plots/qat_vs_ptq.png` — three-way comparison (FP16 / PTQ INT4 / QAT INT4)
- `qat_vs_ptq_comparison.json`

---

## Execution Timeline

```
Week 1    All teams:  make install && pytest tests/
          Team A:     run_ptq_sweep.py (FP16 only → validate calibration pipeline)
          Team B:     run_profiling.py (FP16 baseline)
          Team C:     set up W&B, repo, experiment tracking

Week 2    All teams:  run_ptq_sweep.py (FP16 baseline, full datasets)

Week 3    All teams:  run_ptq_sweep.py (GPTQ_INT8, GPTQ_INT4, AWQ_INT4, NF4)
          Team B:     run_profiling.py (all precisions) + Nsight Systems

Week 4    Team A:     entropy distribution analysis, prepare KD scripts
          Team B:     run_vllm_serving.py (concurrency sweep)
          Team C:     begin aggregating calibration results

          ── CHECKPOINT: merge all results into shared repo ──

Week 5    Team A:     run_kd.py (all 3 models)
          Team B:     complete Roofline for Teams A & C models
          Team C:     run_analysis.py (Pareto + routing)

Week 6    Team A:     run_qat.py (stretch goal, if GPU budget permits)
          Team B:     finalize Roofline figures + throughput tables
          Team C:     finalize Pareto visualizations

Weeks 7–8  All teams: write report sections, prepare presentation
```

---

## Calibration Methodology

### Why log-likelihood scoring?

We evaluate calibration on **multiple-choice tasks** (ARC-Challenge, HellaSwag, TriviaQA).
For each sample we score every answer option by its average token log-probability
conditioned on the question:

```python
score(choice) = mean[ log P(token_i | question, choice_tokens_0..i-1) ]
probs = softmax(scores over all choices)
```

This gives a proper probability distribution over 3–5 choices, making ECE and
Brier score well-defined. The alternative — raw softmax over 32K vocab tokens — is
not meaningful for calibration because it mixes irrelevant tokens into the probability mass.

### Calibration Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ECE** (15 bins) | Σ\|acc(b)−conf(b)\|×n(b)/N | Average miscalibration across confidence bins |
| **MCE** | max\|acc(b)−conf(b)\| | Worst-case bin miscalibration |
| **Brier** | mean Σ(p_c − y_c)² | Proper scoring rule combining accuracy and calibration |
| **Brier-Reliability** | Murphy decomposition | How well predicted probs match observed frequencies |
| **Brier-Resolution** | Murphy decomposition | How much forecasts vary from base rate |

### Temperature Scaling Baseline

As a post-hoc recalibration baseline we fit a single temperature scalar T on a
held-out validation split. If T > 1 the model is overconfident; T < 1 it is
underconfident. We compare temperature-scaled calibration against KD-recovered
calibration to determine whether KD provides calibration recovery *beyond* simple
rescaling.

---

## Knowledge Distillation Details

We use Hinton (2015) soft-label distillation with T² gradient rescaling:

```
Loss = α × KL(student_soft || teacher_soft) × T²
     + (1−α) × CrossEntropy(student, hard_labels)
```

- **Teacher**: frozen FP16 model
- **Student**: INT4 quantized model
- **Trainable**: `lm_head` + LoRA adapters on `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Default hyperparameters**: T=4, α=0.7, lr=2e-5, 3 epochs on 2K WikiText-2 samples

The T² term (Hinton 2015) ensures the KD gradient has the same scale as the CE
gradient, regardless of temperature. This is critical for stable training.

---

## Pareto Analysis

We identify **Pareto-optimal** configurations: no other configuration is simultaneously
cheaper, more accurate, and better calibrated.

Objectives (all normalized to "higher = better"):
- Accuracy ↑
- −ECE ↑ (lower ECE is better)
- −Brier ↑
- Throughput (tok/s) ↑
- −GPU memory ↓

Configuration A **dominates** B if A is at least as good on all objectives
and strictly better on at least one.

---

## Routing Simulation

We simulate a confidence-threshold routing strategy:

```
if quantized_model.max_prob(query) >= θ:
    serve cheaply with quantized model
else:
    escalate to FP16
```

For each threshold θ ∈ [0, 1] we report:
- Fraction of queries served cheaply
- Effective accuracy of the mixed system
- Effective ECE of the mixed system  
- Cost saving vs uniform FP16 serving

We find the **optimal threshold** θ*: the highest cost saving that keeps
accuracy drop < 1% and ECE increase < 0.005 vs FP16 baseline.

---

## Expected Results

Based on Proskurina et al. (NAACL 2024), the closest prior work:

| Finding | Expected |
|---------|----------|
| INT4 ECE increase | 0.01–0.05 |
| AWQ vs GPTQ calibration | AWQ generally better |
| Scale effect (7B vs 13B) | 13B more robust |
| KD recovery of ECE | 40–70% |
| Optimal routing cost saving | 30–60% at <1% accuracy drop |

---

## Reading List

### Calibration Foundations
- Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017)
- Kadavath et al., *Language Models (Mostly) Know What They Know* (arXiv 2022)
- Naeini et al., *Obtaining Well-Calibrated Probabilities Using BBQ* (AAAI 2015)

### PTQ Methods
- Frantar et al., *GPTQ* (ICLR 2023)
- Lin et al., *AWQ* (MLSys 2024, Best Paper)
- Dettmers et al., *QLoRA* (NeurIPS 2023)
- Dettmers et al., *LLM.int8()* (NeurIPS 2022)

### Quantization × Calibration *(core project literature)*
- **Proskurina et al., *When Quantization Affects Confidence of LLMs* (NAACL 2024)** — closest prior work
- Williams & Aletras, *Impact of Calibration Data in PTQ and Pruning* (ACL 2024)
- Xia et al., *Confidence–Calibration Dilemma in Quantized NNs* (arXiv 2021)
- Akgül et al., *Interpreting the Effects of Quantization on LLMs* (arXiv 2025)

### Knowledge Distillation
- Hinton et al., *Distilling the Knowledge in a Neural Network* (NeurIPS 2015)
- Kim et al., *The Role of Teacher Calibration in KD* (arXiv 2025)
- Gu et al., *Knowledge Distillation of LLMs* (ICLR 2024)

### Efficient LLM Serving
- Kwon et al., *PagedAttention / vLLM* (SOSP 2023)
- Dao et al., *FlashAttention-2* (ICLR 2024)

---

## Deliverables Checklist

- [ ] Calibration evaluation framework (ECE, MCE, Brier, reliability diagrams)
- [ ] FP16 baseline results for all 3 models × 3 datasets
- [ ] PTQ calibration sweep: 15 model–config pairs
- [ ] Temperature scaling post-hoc recalibration baseline
- [ ] CUDA profiling: latency, memory, Roofline per config
- [ ] vLLM throughput benchmark with concurrency sweep
- [ ] KD calibration recovery analysis
- [ ] Cross-model Pareto frontiers (2D + 3D)
- [ ] Cross-model heatmaps
- [ ] Uncertainty-based routing simulation
- [ ] Technical report (8–10 pages, workshop-ready)
- [ ] Final presentation + live demo
- [ ] [Stretch] QAT vs PTQ calibration comparison

---

## License

MIT
