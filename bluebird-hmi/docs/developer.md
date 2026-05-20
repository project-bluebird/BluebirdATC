# Welcome to bluebird-hmi

The `bluebird-hmi` package is a frontend page for visualising simulated scenarios running in the `bluebird-dt` digital twin (HMI stands for "Human Machine Interface").
It is written in the [React](https://react.dev/) framework.

## Terminology

In this context, **HMI** means the web-based **GUI** ("Human Machine Interface") used to view and interact with a running Bluebird simulation.

## Using the pre-built HMI

The built (minified) HMI code is distributed in the BluebirdATC repository, so users can run it without installing Node.js or frontend dependencies.

1. Start the `bluebird-api` service (see documentation [here](../bluebird-api/index.md)).
2. Confirm the API is running locally on port `8000`.
3. Open the HMI in your browser at [http://localhost:8000/hmi](http://localhost:8000/hmi).

## Developer Guide

### Installing dependencies

For source development, install Node.js `^20.19.0 || >=22.12.0` if needed. The Node.js installer includes `npm`; then install package dependencies in `bluebird-hmi`:

```bash
npm install
```

### Running locally

To work on the frontend locally from source, use the `bluebird-hmi` directory:

```bash
cd bluebird-hmi
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

### Building for production

To build the minified frontend bundle:

```bash
cd bluebird-hmi
npm run build
```

### Running tests and checks

There is currently no dedicated JavaScript test runner configured in `package.json` for `bluebird-hmi`.

For validation checks, run:

```bash
npm run lint
npm run build
```

### Source README

For full package-level developer details, see the source README:

[bluebird-hmi README](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-hmi/README.md){ target="_blank" rel="noopener" }
