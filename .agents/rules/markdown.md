# Markdown Rendering Rules (research notes & docs)

The site is built with **MkDocs + Python-Markdown** (`pymdownx.arithmatex` in `generic` mode for
math). Python-Markdown is **stricter than GitHub-flavoured Markdown**: several things that look fine
on GitHub render wrong on the built site. Follow these rules so `mkdocs build --strict` output matches
what you intend.

## 1. A blank line before every list (the #1 offender)

Python-Markdown only starts a list when it is **separated from the preceding paragraph by a blank
line**. Without it, the list is absorbed into the paragraph and renders as inline text — no bullets,
no numbers.

```markdown
<!-- WRONG: renders as one paragraph, "- a - b" inline -->
Các bước:
- a
- b

<!-- RIGHT -->
Các bước:

- a
- b
```

This applies to `-`, `*`, `+` bullets **and** `1.` / `2.` ordered lists, and to lists that follow a
**bold label** (`**Ưu điểm:**`), a sentence ending in `:`, a heading, or a `$$…$$` block. When in
doubt, put a blank line before the first item.

## 2. Display math `$$…$$` needs blank lines around it

A `$$…$$` block must have a blank line **before and after**, otherwise it glues to the surrounding
text or breaks a list. Inside a list item, **indent the `$$` block by 4 spaces** and surround it with
blank lines so it stays part of the item (and the list keeps numbering):

```markdown
1. **Tính trung bình**: cho từng chiều

    $$ \mu_i = \frac{1}{n}\sum_j x_{ij} $$

2. **Định tâm dữ liệu**: ...
```

## 3. Math in notebook (`.ipynb`) markdown cells: single backslash

Notebook markdown is rendered by MathJax directly (it does **not** go through arithmatex). Write LaTeX
with a **single** backslash — `$\Sigma$`, `$\mathbf{v}$`, `$\top$`. A **double** backslash `$\\Sigma$`
is read by MathJax as a line break, so the symbol renders as broken text. (Code cells are different —
matplotlib mathtext titles legitimately need the Python-string `\\`.)

## 4. Other Python-Markdown gotchas

- **Tables** also need a blank line before them.
- A list item's continuation / nested block must be **indented 4 spaces** to stay inside the item.
- Don't rely on GitHub's "lazy" list parsing; be explicit with blank lines.

## How to check before committing

`mkdocs build --strict` does **not** catch glued lists (they are valid Markdown, just wrong output).
Eyeball the rendered page, or render a file through Python-Markdown and confirm lists become `<ul>` /
`<ol>` rather than ending up inside a `<p>`.
