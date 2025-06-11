import subprocess
import signal
import sys
import time
import os
import argparse
import numpy as np
import logging


def get_logger_console(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger_console(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Score datasets using specified model with multiple processes")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./",
        help="Path to the datasets directory (default: ./)",
    )
    parser.add_argument(
        "--all_datasets",
        type=str,
        default="scifact,hotpotqa,fever,nfcorpus,fiqa",
        help="Comma-separated list of datasets to process (default: scifact,hotpotqa,fever,nfcorpus,fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scores",
        help="Directory to save output scores (default: scores)",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default="minilm",
        help="Model identifier to use for scoring (default: minilm)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to run (default: 8)",
    )
    return parser.parse_args()


def run_processes(args):
    processes = []
    
    for dataset_name in args.all_datasets.split(","):
        # Create output directory
        os.makedirs(os.path.join(args.output_dir, dataset_name), exist_ok=True)
        
        # Start worker processes for this dataset
        dataset_processes = []
        
        for rank in range(args.num_processes):
            cmd = [
                "python",
                "scorer_worker.py",
                "--dataset_path", args.dataset_path,
                "--dataset_name", dataset_name,
                "--output_dir", args.output_dir,
                "--identifier", args.identifier,
                "--total_processes", str(args.num_processes),
                "--process_rank", str(rank),
            ]
            
            process = subprocess.Popen(cmd)
            dataset_processes.append(process)
            logger.info(f"Started worker process {rank} for dataset {dataset_name} with PID {process.pid}")
        
        # Wait for all processes of this dataset to complete
        if not wait_for_processes(dataset_processes, dataset_name):
            return False
            
        # Combine scores for this dataset
        combine_scores(args.output_dir, dataset_name, args.identifier, args.num_processes)
    
    return True


def wait_for_processes(processes, dataset_name):
    """Wait for all processes to complete and handle failures"""
    while True:
        all_done = True
        for i, proc in enumerate(processes):
            if proc.poll() is None:  # Process is still running
                all_done = False
            else:
                # Process has ended, check return code
                return_code = proc.poll()
                if return_code != 0:
                    logger.error(f"Process {i} for dataset {dataset_name} failed with return code {return_code}")
                    # Terminate all other processes
                    cleanup_processes(processes)
                    return False
        
        if all_done:
            logger.info(f"All processes for dataset {dataset_name} completed successfully")
            break
        
        time.sleep(1)
    
    return True


def combine_scores(output_dir, dataset_name, identifier, num_processes):
    """Combine scores from all processes"""
    logger.info(f"Combining scores for dataset {dataset_name}")
    
    combined_scores = []
    
    # Read scores from each process in order
    for proc_idx in range(num_processes):
        proc_file = os.path.join(
            output_dir, dataset_name, identifier + f"_{proc_idx}.npy"
        )
        if os.path.exists(proc_file):
            proc_scores = np.load(proc_file)
            combined_scores.append(proc_scores)
            # Delete the individual file after reading
            os.remove(proc_file)
            logger.info(f"Loaded and removed {proc_file}")
        else:
            logger.warning(f"File {proc_file} not found")
    
    if combined_scores:
        # Concatenate all scores
        combined_scores = np.concatenate(combined_scores, axis=0)
        logger.info(f"Combined scores shape: {combined_scores.shape}")
        
        # Save the combined scores
        final_output_file = os.path.join(
            output_dir, dataset_name, identifier + f"_scores.npy"
        )
        np.save(final_output_file, combined_scores)
        logger.info(f"Saved combined scores to {final_output_file}")
    else:
        logger.error(f"No scores found to combine for dataset {dataset_name}")


def cleanup_processes(processes):
    """Clean up all child processes"""
    for proc in processes:
        try:
            if proc.poll() is None:  # If process is still running
                proc.terminate()  # Send SIGTERM
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()  # Force terminate if process doesn't respond
                logger.info(f"Terminated process {proc.pid}")
        except Exception as e:
            logger.error(f"Error while terminating process {proc.pid}: {e}")


def signal_handler(signum, frame, processes):
    logger.info("Received signal to terminate. Cleaning up...")
    cleanup_processes(processes)
    sys.exit(0)


def main():
    args = parse_arguments()
    
    logger.info(f"Starting scorer with {args.num_processes} processes")
    logger.info(f"Datasets: {args.all_datasets}")
    logger.info(f"Identifier: {args.identifier}")
    
    processes = []
    
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, processes))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, processes))
    
    try:
        success = run_processes(args)
        if success:
            logger.info("All datasets processed successfully")
        else:
            logger.error("Some processes failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_processes(processes)
        sys.exit(1)
    finally:
        cleanup_processes(processes)


if __name__ == "__main__":
    main() 