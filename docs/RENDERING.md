# Reading and rendering notebooks

The notebooks include saved text and figure outputs. Standalone reading versions are provided in `notebooks/html/`.

To update the HTML files from saved outputs:

```bash
python scripts/render_notebooks.py
```

With Jupyter and nbconvert installed, standard HTML export is also available:

```bash
jupyter nbconvert --to html notebooks/00_archive_overview.ipynb
```
