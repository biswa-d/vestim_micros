# Vestim Refactoring Plan

## Proposed Architecture

The new architecture will follow a classic client-server model, with the GUI acting as the client and the backend as the server. The gateway will serve as the communication layer, making requests to the backend API.

Here's a Mermaid diagram illustrating the proposed architecture:

```mermaid
graph TD
    subgraph Client
        A[GUI (PyQt)] --> B{Gateway (Managers)};
    end

    subgraph Server
        D[API (FastAPI)] --> E{Services};
        E --> F[Job Management];
        F --> G[Model Training];
        F --> H[Data Processing];
    end

    B -- HTTP Requests --> D;
    D -- HTTP Responses --> B;
```

## Detailed Plan

1.  **Backend (Server):**
    *   **Technology:** We'll use **FastAPI** to create a robust and efficient REST API for the backend. FastAPI is a modern, high-performance web framework for building APIs with Python.
    *   **API Endpoints:** The API will expose the following endpoints:
        *   `POST /jobs`: Create a new job and return a unique job ID.
        *   `POST /jobs/{job_id}/train`: Start the training process for a specific job.
        *   `GET /jobs/{job_id}/status`: Get the current status of a job (e.g., "running," "completed," "failed").
        *   `GET /jobs/{job_id}/results`: Get the results of a completed job.
        *   `POST /jobs/{job_id}/stop`: Stop a running job.
    *   **Service Consolidation:** The existing services in `vestim/services` will be moved to `vestim/backend/src/services` and integrated with the FastAPI application.

2.  **Gateway (Client-Side Logic):**
    *   **Communication:** The `vestim/gateway` modules will be refactored to make HTTP requests to the backend API using a library like `requests` or `httpx`.
    *   **Singleton Managers:** The singleton pattern for the manager classes will be maintained to ensure a consistent state across the application.
    *   **Job Tracking:** The gateway will be responsible for storing the job ID received from the server and using it for all subsequent requests related to that job.

3.  **GUI (Client-Side Interface):**
    *   **Asynchronous Updates:** The GUI will be updated to interact with the gateway in a non-blocking manner. This will prevent the GUI from freezing while waiting for responses from the server. We can achieve this using Qt's threading capabilities (`QThread`) or by running the gateway in a separate process.
    *   **Job Monitoring:** The GUI will periodically poll the backend for the status of active jobs and update the display accordingly.
    *   **Reconnect to Jobs:** We will implement a mechanism to allow the user to close and reopen the GUI and reconnect to a running job by providing the job ID.

4.  **SSH Connectivity:**
    *   **Remote Execution:** We'll use a library like **`paramiko`** to manage the SSH connection to the remote Linux server.
    *   **Server Deployment:** The client application will have a feature to deploy and launch the backend server on the remote machine via SSH. This will involve transferring the necessary files and executing the command to start the FastAPI server.

## Implementation Steps

1.  **Setup Backend:**
    *   Create a new `main.py` file in `vestim/backend/src` to set up the FastAPI application.
    *   Define the API endpoints for job management and training.
    *   Move the existing services from `vestim/services` to `vestim/backend/src/services` and integrate them with the API.

2.  **Refactor Gateway:**
    *   Update the gateway modules to make HTTP requests to the new backend API.
    *   Replace the direct calls to the service modules with API calls.

3.  **Update GUI:**
    *   Modify the GUI to interact with the refactored gateway.
    *   Implement the asynchronous updates and job monitoring features.
    *   Add a feature to reconnect to a running job.

4.  **Implement SSH Connectivity:**
    *   Add a new module to the gateway for managing the SSH connection.
    *   Implement the logic for deploying and launching the backend server on a remote machine.