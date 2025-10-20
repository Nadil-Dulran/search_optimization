# Optimization & Search (Programming Assignment)

This repo contains my solutions (**BFS, Heuristics, A*, IDS, Simulated Annealing(SA), LP, DP**) plus a small runner and a browser summary page.

The runner generates `problem.json` and `results.json`. The `index.html` page loads those files to render grids/plots and lets you type justifications (**saved locally in your browser**).

---

## Folder layout

```
Assignment_/
├── index.html
├── runner.py
├── heuristics.py
├── student_bfs.py
├── student_astar.py
├── student_ids.py
├── student_sa.py
├── student_lp_dp.py
└── (generated at runtime)
    ├── problem.json
    └── results.json
```

---

## Prerequisites

- Python **3.9+**
- A terminal (macOS: **Terminal/zsh**; Windows: **PowerShell**)
- A modern browser (**Chrome/Edge/Firefox**)

> No external services required. Everything runs locally.

---

## Quick Start — macOS (zsh)

```bash
# 1) Go to the assignment folder
cd /path/to/Assignment_IT23161788

# 2) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install deps (only a couple needed)
pip install numpy matplotlib

# 4) Generate results (you can change seed/rows/cols/density)
python3 runner.py --student_id IT23161788 --seed 42 --rows 6 --cols 6 --density 0.22

# 5) Start a local web server from this exact folder
python3 -m http.server 8000

# 6) Open the summary page
# (copy/paste in your browser)
http://localhost:8000/index.html

```

---

## Quick Start — Windows (PowerShell)

```powershell
# 1) Go to the assignment folder
cd C:\path\to\Assignment_IT23161788

# 2) Create & activate a virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install deps
pip install numpy matplotlib

# 4) Generate results
py -3 .\runner.py --student_id IT23161788 --seed 42 --rows 6 --cols 6 --density 0.22

# 5) Start a local web server
py -3 -m http.server 8000

# 6) Open the summary page in your browser
http://localhost:8000/index.html

```

---

## Tips & Notes

- **Justifications are saved locally** in your browser’s `localStorage`.
  - Use the **same URL/port** (e.g., `http://localhost:8000`) when you come back, or the notes won’t appear.
- If you only see the **table header** and no content, make sure you’re serving via `http://` (not `file://`) and that `problem.json` and `results.json` exist in the same folder as `index.html`.
- You can tweak the problem via CLI flags:
  ```
  --seed <int>  --rows <int>  --cols <int>  --density <0..1>  --weighted
  ```

### Debug prints

BFS/A*/IDS can print debug info if you enable the flag:

- **macOS/Linux:**
  ```bash
  ASSIGN_DEBUG=1 python3 runner.py --student_id IT23161788
  ```

- **Windows PowerShell:**
  ```powershell
  $env:ASSIGN_DEBUG=1; py -3 runner.py --student_id IT23161788
  ```

---

## Troubleshooting

**Page shows only “Task | Status | Score | Details”**  
Start a local server in the *same* folder as the JSONs:
```bash
python3 -m http.server 8000
```
Then open `http://localhost:8000/index.html`. In DevTools → **Network**, click `results.json` and confirm it loads (status **200**) and contains fresh data.

**Stale results**  
Regenerate and hard-reload:
```bash
# macOS/Linux
rm -f problem.json results.json
python3 runner.py --student_id IT23161788 --seed 42

# Windows (PowerShell)
Remove-Item problem.json, results.json -ErrorAction SilentlyContinue
py -3 runner.py --student_id IT23161788 --seed 42
```
Open `http://localhost:8000/index.html?ts=1` and hard refresh (**Cmd/Ctrl+Shift+R**). In DevTools, tick **Disable cache**.

**Runner errors / NotImplementedError**  
Implement the remaining student files (**BFS, A\*, IDS, SA, LP, DP**). The runner writes JSONs only when it completes.

**Chart not visible offline**  
The SA chart may use a CDN when offline; that’s okay per the brief—the grading still uses the generated history.

---
