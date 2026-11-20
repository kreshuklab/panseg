.PHONY: test run

## Run tests headless under Xvfb (works inside the agent sandbox). Pass ARGS="tests/widgets -q" to scope.
test:
	env -u QT_QPA_PLATFORM LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=$(CURDIR) xvfb-run -a python -m pytest $(ARGS)

## Run the panseg CLI from this worktree/checkout instead of the editable install, which points at the main checkout.
run:
	env -u QT_QPA_PLATFORM PYTHONPATH=$(CURDIR) python -m panseg.run_panseg -n $(ARGS)
