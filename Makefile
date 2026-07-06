# Sgr A* Scientific Simulator — top-level automation.
#
#   make figures    — regenerate every paper figure from simulation output
#   make paper      — figures + compile paper/main.pdf (pdflatex + bibtex)
#   make reproduce  — re-run the full science chain that produces every number
#                     cited in the paper (tests, fits, metrology, calibration)
#
# Windows note: run under Git Bash / MSYS make with the MSVC toolchain already
# built (scripts/vcenv.cmd wraps the environment for direct cmake calls).

PY      ?= py -3.11
BUILD   ?= build/sci-rel

.PHONY: paper thesis figures reproduce clean-paper

figures:
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_diagrams.py

paper: figures
	$(MAKE) -C paper

thesis: figures
	$(MAKE) -C paper thesis

reproduce:
	cmd //c "scripts\\vcenv.cmd ctest --preset sci-rel"
	$(BUILD)/validation/render_reference.exe schwarzschild validation/reference_images/schwarzschild_cpu.png 800
	$(BUILD)/validation/render_reference.exe kerr94 validation/reference_images/kerr_cpu_a94.png 800
	$(BUILD)/science/subring_tool.exe validation/phase3 0.94 80 800
	$(BUILD)/science/orbit_fitter/fit_s2_tool.exe data/gillessen_2017 validation/phase4 kerr
	$(BUILD)/science/orbit_fitter/fit_s2_tool.exe data/gillessen_2017 validation/phase5 nohair
	$(BUILD)/validation/sweep_tool.exe calibrate validation/phase10
	$(BUILD)/validation/sweep_tool.exe sweep validation/phase10
	$(BUILD)/validation/physics_tables.exe validation/thesis
	$(BUILD)/validation/convergence_tool.exe validation/thesis
	$(BUILD)/validation/sweep_tool.exe photonorbits validation/thesis
	$(BUILD)/science/subring_tool.exe --sweep validation/thesis 80 560
	$(PY) scripts/run_mcmc.py both
	$(BUILD)/science/orbit_fitter/fit_cluster.exe data/gillessen_2017 validation/phase13
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_diagrams.py
	$(PY) scripts/make_numbers.py

clean-paper:
	$(MAKE) -C paper clean
