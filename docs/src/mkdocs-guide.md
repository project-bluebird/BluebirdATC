# How to change these docs

## Intro

This repository uses [MkDocs](https://www.mkdocs.org/) with Material theme and a monorepo plugin to publish documentation for all BluebirdATC packages from a single docs site.

At the docs project level, `docs/mkdocs.yml` defines:

- site-wide theme and plugin settings
- top-level navigation (Home, package docs, and this guide)
- inclusion of package-level docs nav files

Each package then provides its own local `mkdocs.yml` to define that package's section structure.

## Serve the docs locally

From the repository root (`BluebirdATC`), run:

```bash
./scripts/docs-serve
```

Then open:

- `http://127.0.0.1:8010` or `http://localhost:8010`

MkDocs watches files and refreshes the site as you edit documentation.

## Build the docs for production

From the repository root (`BluebirdATC`), run:

```bash
./scripts/docs-build
```

This generates a static site in the `site/` directory, ready to publish to static hosting (for example, GitHub Pages).

## File hierarchy

The docs setup is split across root docs files and package docs files.

Main locations:

- `docs/pyproject.toml`: docs-specific uv project and dependencies
- `uv.lock`: workspace lockfile
- `docs/mkdocs.yml`: docs project MkDocs config and top-level site nav
- `docs/src/`: root docs content and shared assets
- `docs/src/index.md`: homepage content
- `docs/src/stylesheets/nav.css`: small nav style overrides
- `bluebird-dt/mkdocs.yml`: nav config for the `bluebird-dt` section
- `bluebird-api/mkdocs.yml`: nav config for the `bluebird-api` section
- `bluebird-gymnasium/mkdocs.yml`: nav config for the `bluebird-gymnasium` section
- `bluebird-hmi/mkdocs.yml`: nav config for the `bluebird-hmi` section
- `*/docs/*.md`: package-specific markdown pages
- `*/docs/source.md`: API/source reference pages used by `mkdocstrings`

## Example edit-serve-build workflow

* Start local docs server from repo root:

```bash
./scripts/docs-serve
```

* Edit a docs file, for example: `bluebird-hmi/docs/index.md`

* Check the change in browser at: `http://localhost:8010`.

* If navigation changed, also update the corresponding `mkdocs.yml` file. 

- For a top-level change: `docs/mkdocs.yml`.
- For a package-level change: e.g. `bluebird-hmi/mkdocs.yml`.  

* Run a production build check:

```bash
./scripts/docs-build
```

* Commit your markdown and `mkdocs.yml` changes together.
