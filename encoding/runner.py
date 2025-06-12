import subprocess
import signal
import sys
import time
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run multiple upload worker processes")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="2",
        help="Model parameter to pass to upload_worker.py (default: 2)",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/opt/dlami/nvme/zeta",
        help="Base directory containing docs/ and queries/ subdirectories",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=8,
        help="Number of worker processes to run for each type (default: 8)",
    )
    parser.add_argument(
        "--docs-subdir",
        type=str,
        default="docs",
        help="Subdirectory name for document files (default: docs)",
    )
    parser.add_argument(
        "--queries-subdir",
        type=str,
        default="queries",
        help="Subdirectory name for query files (default: queries)",
    )
    return parser.parse_args()


def run_processes(args):
    processes = []

    for i in range(args.count):
        # Start document processing
        docs_file = f"{os.path.join(args.directory, args.docs_subdir)}/{i}.bin"
        cmd = [
            "python",
            "upload_worker.py",
            "-f",
            docs_file,
            "-t",
            "vector",
            "-m",
            args.model,
        ]

        process = subprocess.Popen(cmd)
        processes.append(process)
        print(f"Started process for file {docs_file} with PID {process.pid}")

        # Start query processing
        queries_file = f"{os.path.join(args.directory, args.queries_subdir)}/{i}.bin"
        cmd = [
            "python",
            "upload_worker.py",
            "-f",
            queries_file,
            "-t",
            "vector_q",
            "-m",
            args.model,
        ]

        process = subprocess.Popen(cmd)
        processes.append(process)
        print(f"Started process for file {queries_file} with PID {process.pid}")

    # Handle signals to ensure cleanup of child processes
    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        cleanup_processes(processes)
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        # Wait for all processes to complete
        while True:
            all_done = True
            for i, proc in enumerate(processes):
                if proc.poll() is None:  # Process is still running
                    all_done = False
                else:
                    # Process has ended, check return code
                    return_code = proc.poll()
                    if return_code != 0:
                        print(f"Process {i} failed with return code {return_code}")
                        # Get error output
                        _, stderr = proc.communicate()
                        if stderr:
                            print(f"Error output: {stderr}")

                        # When a process fails, terminate all other processes immediately
                        print("Terminating all processes due to failure...")
                        cleanup_processes(processes)
                        sys.exit(1)

            if all_done:
                print("All processes completed successfully")
                break

            time.sleep(1)  # Avoid excessive CPU consumption

    except Exception as e:
        print(f"Error occurred: {e}")
        cleanup_processes(processes)
        sys.exit(1)

    finally:
        cleanup_processes(processes)


def cleanup_processes(processes):
    """Clean up all child processes"""
    for proc in processes:
        try:
            if proc.poll() is None:  # If process is still running
                # Use taskkill on Windows
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
                # Use SIGTERM on Unix-like systems
                else:
                    proc.terminate()  # Send SIGTERM
                    # Give the process some time to clean up
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()  # Force terminate if process doesn't respond

                print(f"Terminated process {proc.pid}")
        except Exception as e:
            print(f"Error while terminating process {proc.pid}: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    run_processes(args)
