"""Helper untuk generate notebook eksperimen secara programatis."""
import nbformat as nbf


def make_notebook(filename, title, md_sections, code_cells):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3 (ipykernel)",
                              "language": "python", "name": "python3"}
    cells = [nbf.v4.new_markdown_cell(title)]
    for md, code in zip(md_sections, code_cells):
        if md:
            cells.append(nbf.v4.new_markdown_cell(md))
        cells.append(nbf.v4.new_code_cell(code))
    nb.cells = cells
    with open(filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("written:", filename)
