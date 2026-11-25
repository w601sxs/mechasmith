# Build the paper. `make results` regenerates the tables from fresh builds (needs an X server for
# renders: run under `xvfb-run -a -s "-screen 0 1280x1024x24"` with MUJOCO_GL=glfw when headless).

main.pdf: main.tex refs.bib sections/*.tex generated/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

results:
	cd .. && uv run python paper/scripts/collect_results.py --rebuild

# Fails while any planned experiment or draft note is still in the text.
check:
	@! grep -n '\\planned{\|\\note{' sections/*.tex || (echo "unfinished items above"; exit 1)

clean:
	latexmk -C main.tex

.PHONY: results check clean
