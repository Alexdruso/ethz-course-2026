PYTHON_VERSION ?= 3.12
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
UV ?= uv
RUFF ?= uvx ruff
TY ?= uvx ty
PY_SRC ?= env exercises scripts

STAMP := $(VENV)/.deps-stamp

# Scripts import both `env`/`exercises` (repo root) and `__init__`/`utils` (scripts/).
RUN := PYTHONPATH=. $(PYTHON)

# TensorBoard
LOGDIR ?= logs
PORT ?= 6006

# Checkpoint selection for the evaluation scripts
LOAD_RUN ?= 1
CHECKPOINT ?= 500

.DEFAULT_GOAL := help
.PHONY: help venv install clean-venv py-fmt py-lint py-static \
        interactive ik spline pid train eval eval-traj tensorboard

help:  ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

venv: $(VENV)/pyvenv.cfg  ## Create the project venv (Python 3.12, override with PYTHON_VERSION=)
$(VENV)/pyvenv.cfg:
	$(UV) venv --python $(PYTHON_VERSION) $(VENV)

install: $(STAMP)  ## Install pinned requirements + so101_gym (editable) into the venv
$(STAMP): requirements.txt pyproject.toml | $(VENV)/pyvenv.cfg
	$(UV) pip install --python $(PYTHON) -r requirements.txt
	$(UV) pip install --python $(PYTHON) -e .
	@touch $@

clean-venv:  ## Remove the venv
	rm -rf $(VENV)

# F401 is never auto-fixed here: the exercise stubs import helpers that only get
# used once the TODOs are filled in, and ruff would happily delete them.
py-fmt:  ## Auto-fix lint and formatting issues with ruff (check --fix, then format)
	-$(RUFF) check --fix --ignore F401 $(PY_SRC)
	$(RUFF) format $(PY_SRC)

py-lint:  ## Report lint issues without changing any files
	$(RUFF) check $(PY_SRC)

py-static:  ## Run astral's ty type checker
	$(TY) check --python $(PYTHON) $(PY_SRC)

interactive: $(STAMP)  ## Smoke test: open the MuJoCo viewer with the SO-100 arm
	$(RUN) scripts/interactive.py

ik: $(STAMP)  ## Ex1: track the Lemniscate keypoints with inverse kinematics
	$(RUN) scripts/inverse_kinematics.py

spline: $(STAMP)  ## Ex2: preview the quintic-spline waypoints (no control law yet)
	$(RUN) scripts/quintic_splines.py

pid: $(STAMP)  ## Ex2: run the full waypoint -> IK -> PID control pipeline
	$(RUN) scripts/pid_control.py

train: $(STAMP)  ## Ex3: train the PPO policy (pass extra flags via ARGS="--max_iterations 100")
	$(RUN) scripts/train.py $(ARGS)

eval: $(STAMP)  ## Ex3: evaluate the trained policy on random targets
	$(RUN) scripts/evaluate_rand_targets.py --load_run=$(LOAD_RUN) --checkpoint=$(CHECKPOINT) $(ARGS)

eval-traj: $(STAMP)  ## Ex3 bonus: evaluate the policy on the Lemniscate trajectory
	$(RUN) scripts/evaluate_trajectory.py --load_run=$(LOAD_RUN) --checkpoint=$(CHECKPOINT) $(ARGS)

tensorboard: $(STAMP)  ## Serve TensorBoard on http://localhost:6006 (override with PORT=)
	$(VENV)/bin/tensorboard --logdir=$(LOGDIR) --port=$(PORT)
