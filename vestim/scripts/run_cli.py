import argparse
import json
from vestim.gateway.src.job_manager_qt import JobManager

def main():
    """
    Command-line interface for creating and managing jobs.
    """
    parser = argparse.ArgumentParser(description="VEstim Command-Line Interface")
    parser.add_argument("command", choices=["create_job"], help="The command to execute.")
    parser.add_argument("--params", type=str, help="Path to a JSON file containing hyperparameters.")

    args = parser.parse_args()

    job_manager = JobManager()

    if args.command == "create_job":
        if not args.params:
            print("Error: --params flag is required for the create_job command.")
            return

        try:
            with open(args.params, 'r') as f:
                hyperparams = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{args.params}' was not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: The file '{args.params}' is not a valid JSON file.")
            return

        job_id, job_folder = job_manager.create_new_job(hyperparams)
        if job_id:
            print(f"Job '{job_id}' created successfully in '{job_folder}'.")
        else:
            print("Failed to create job.")

if __name__ == "__main__":
    main()