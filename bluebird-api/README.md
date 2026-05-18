## The REST API for BluebirdATC

It is possible to run the BluebirdATC digital twin in a server process, such that the simulation will evolve at regular time intervals, and Agents and/or frontend visualization software can interact with it via HTTP requests.
In particular, users can:
* Query available scenario categories and scenarios.
* Load a selected scenario.
* Evolve the simulation by a specified time interval.
* Obtain the current state of the `Environment`.
* Submit `Actions` to individual aircraft.
* Save logfiles with data on all steps of the simulation.

In order to run the app, with all the correct dependencies for this feature, from the `BluebirdATC/bluebird-api` directory, run the command:

```shell
uv run uvicorn bluebird_api:app --port 8000
```

You should then be able to go to [http://localhost:8000](http://localhost:8000) in a web browser, and see the message "Hello, BluebirdATC!".

To see the full list and description of API endpoints, with the application running, go to [http://localhost:8000/docs](http://localhost:8000/docs).

## Frontend visualisation

The app also serves the frontend visualization (more details on that can be found [here](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-hmi/README.md)), at the URL [http://localhost:8000/hmi](http://localhost:8000/hmi).
