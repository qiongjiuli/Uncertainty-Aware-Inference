# Makefile
# Convenience targets for running the full project pipeline.
# Usage: make <target> MODEL=Llama-2-7B MODEL_ID=meta-llama/Llama-2-7b-hf

.PHONY: help install test sweep-a sweep-b sweep-c profile kd analysis vllm qat all

# ── Defaults (override on command line) ────────────────────────────────────────
MODEL_A     ?= Llama-2-7B
MODEL_A_ID  ?= meta-llama/Llama-2-7b-hf

MODEL_B     ?= Mistral-7B
MODEL_B_ID  ?= mistralai/Mistral-7B-v0.1

MODEL_C     ?= Llama-2-13B
MODEL_C_ID  ?= meta-llama/Llama-2-13b-hf

N_SAMPLES   ?= 500
DEVICE      ?= cuda

# ── Help ───────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Uncertainty-Aware Inference — Project Makefile"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install all dependencies"
	@echo "  make test            Run unit tests"
	@echo ""
	@echo "Phase 1-2: PTQ Calibration Sweep (run all three in parallel):"
	@echo "  make sweep-a         Team A: Llama-2 7B calibration sweep"
	@echo "  make sweep-b         Team B: Mistral 7B calibration sweep"
	@echo "  make sweep-c         Team C: Llama-2 13B calibration sweep"
	@echo ""
	@echo "Phase 2: GPU Profiling:"
	@echo "  make profile         Team B: profile all models"
	@echo "  make vllm            Team B: vLLM serving benchmark"
	@echo "  make vllm-commands   Print vLLM server launch commands"
	@echo ""
	@echo "Phase 3: Distillation + Analysis:"
	@echo "  make kd              Team A: knowledge distillation experiment"
	@echo "  make analysis        Team C: Pareto + routing simulation"
	@echo "  make qat             Team A: QAT stretch goal (8-12 GPU hrs)"
	@echo ""
	@echo "  make all             Run full pipeline in sequence"
	@echo ""
	@echo "Override defaults:"
	@echo "  make sweep-a N_SAMPLES=1000 DEVICE=cuda"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	@echo "Installation complete."

test:
	pytest tests/ -v --tb=short

# ── PTQ Sweeps ─────────────────────────────────────────────────────────────────
sweep-a:
	python scripts/run_ptq_sweep.py \
		--model_id   $(MODEL_A_ID) \
		--model_name $(MODEL_A) \
		--output_dir results/$(MODEL_A) \
		--n_samples  $(N_SAMPLES) \
		--device     $(DEVICE)

sweep-b:
	python scripts/run_ptq_sweep.py \
		--model_id   $(MODEL_B_ID) \
		--model_name $(MODEL_B) \
		--output_dir results/$(MODEL_B) \
		--n_samples  $(N_SAMPLES) \
		--device     $(DEVICE)

sweep-c:
	python scripts/run_ptq_sweep.py \
		--model_id   $(MODEL_C_ID) \
		--model_name $(MODEL_C) \
		--output_dir results/$(MODEL_C) \
		--n_samples  $(N_SAMPLES) \
		--device     $(DEVICE)

# ── Profiling ──────────────────────────────────────────────────────────────────
profile:
	@for model_id in "$(MODEL_A_ID)" "$(MODEL_B_ID)" "$(MODEL_C_ID)"; do \
		name=$$(echo $$model_id | sed 's|.*/||'); \
		python scripts/run_profiling.py \
			--model_id   $$model_id \
			--model_name $$name \
			--output_dir results/profiling \
			--device     $(DEVICE); \
	done

vllm-commands:
	python scripts/run_vllm_serving.py \
		--model_id $(MODEL_A_ID) --model_name $(MODEL_A) \
		--print_commands

vllm:
	python scripts/run_vllm_serving.py \
		--model_id   $(MODEL_A_ID) \
		--model_name $(MODEL_A) \
		--output_dir results/vllm \
		--precisions FP16 GPTQ_INT4 AWQ_INT4 NF4

# ── Distillation ───────────────────────────────────────────────────────────────
kd:
	python scripts/run_kd.py \
		--model_id          $(MODEL_A_ID) \
		--model_name        $(MODEL_A) \
		--student_precision GPTQ_INT4 \
		--results_dir       results/$(MODEL_A) \
		--output_dir        results/$(MODEL_A)/kd \
		--n_kd_samples      2000 \
		--n_epochs          3 \
		--device            $(DEVICE)

# ── Cross-model Analysis ───────────────────────────────────────────────────────
analysis:
	python scripts/run_analysis.py \
		--calibration_dirs results/$(MODEL_A) results/$(MODEL_B) results/$(MODEL_C) \
		--profiling_dir    results/profiling \
		--output_dir       results/analysis

# ── QAT Stretch Goal ───────────────────────────────────────────────────────────
qat:
	@echo "WARNING: QAT requires ~20GB VRAM and 8-12 GPU hours."
	@read -p "Continue? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		python scripts/run_qat.py \
			--model_id        $(MODEL_A_ID) \
			--model_name      $(MODEL_A) \
			--results_dir     results/$(MODEL_A) \
			--output_dir      results/$(MODEL_A)/qat \
			--n_train_samples 2000 \
			--n_epochs        3 \
			--device          $(DEVICE); \
	fi

# ── Full Pipeline ──────────────────────────────────────────────────────────────
all: sweep-a sweep-b sweep-c profile kd analysis
	@echo ""
	@echo "Full pipeline complete. Results in results/"
	@echo "Run 'make analysis' again if new profiling data was added."
