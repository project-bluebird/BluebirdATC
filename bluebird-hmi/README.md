# Bluebird-HMI

Bluebird-HMI is an optional package for visualizing simulations in the BluebirdATC digital twin.  It is designed to be viewed as a webpage, served from the user's local machine (`localhost`).  It is written in typescript in the React framework, and both the source code and the built (minified) application are available here.

## Quickstart

* Ensure that the BluebirdATC REST API is running, following the instructions in the [README](../bluebird-api/README.md).
* You should be able to view the frontend application at the URL [http://localhost:8000/hmi](http://localhost:8000/hmi).

### Note: running on a remote machine/cloud

Currently, the built version of the app is configured to look for the API running on `localhost`.  For deploying on remote machines, or a cloud service, it will be necessary to modify `src/api/config.ts` accordingly, and rebuild via `npm run build`.

## Developer instructions/details

The source code for the application is provided in the `src/` directory. Use Node.js `^20.19.0 || >=22.12.0` with `npm` if you need to build a modified version of the code, or run the dev server to test updates/fixes.

### Overview of the code

The application uses the redux/rtk toolkit to maintain a data store, which is continually refreshed by polling the backend API.  The "slices" of data can be seen in `src/slices/` and broadly represent:
* *ScenarioData* containing the state of the simulation - if a scenario is currently loaded, and if so, what is its name and category, and current simulation time.
* *StaticData* containing data that doesn't typically change over time, such as the geometry of the airspace (sector boundaries, fixes, etc.)
* *DynamicData* containing data that changes tick-by-tick, such as aircraft positions and speeds.

It is essentially a single page application, with the top-level page being defined in `src/pages/RadarPage.tsx`. This page in turn contains two main React *components* - the *Radar* (a map view with the airspace and aircraft positions marked), and a *RadarDrawer* to the left of the screen, with options and controls.

### Running the dev server

* Install Node.js `^20.19.0 || >=22.12.0` if you don't already have it. The Node.js installer includes `npm`.
* From the directory `BluebirdATC/bluebird-hmi` type the commands:

 ```shell
 npm install
 npm run dev
 ```

 then navigate your browser to `http://localhost:5173`.

### Rebuilding

To rebuild the minified version of the code in the `bluebird-api/bluebird_api/hmi/` directory, run the command:

```shell
npm run build
```

from the `BluebirdATC/bluebird-hmi` directory.

If you then want to update the built application in Github, first see the CONTRIBUTING.md guidelines if you are unsure about how to work in branches.  If you are confident you are on a working branch (or your own fork of the repository), and are happy to proceed, you should then run `git status`.   You will see something like:

```
Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    ../bluebird-api/bluebird_api/hmi/assets/index-<some-random-string>.js
	modified:   ../bluebird-api/bluebird_api/hmi/index.html

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	bluebird-api/bluebird_api/hmi/assets/index-<another-random-string>.js
```

You can then do:

```shell
git add ../bluebird-api/bluebird_api/hmi/*
git commit -m "update HMI build"
git push
```
You can then make a Pull Request into the `dev` branch, following the CONTRIBUTING.md guidelines.
