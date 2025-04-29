import subprocess
import signal
import sys
import time
import os


def run_processes():
    processes = []

    for i in range(8):
        cmd = [
            "python",
            "upload_worker.py",
            "-f",
            f"/opt/dlami/nvme/zeta/docs/{i}.bin",
            "-t",
            "vector",
            "-m",
            "2",
        ]

        process = subprocess.Popen(cmd)
        processes.append(process)
        print(
            f"Started process for file /opt/dlami/nvme/zeta/docs/{i}.bin with PID {process.pid}"
        )

        cmd = [
            "python",
            "upload_worker.py",
            "-f",
            f"/opt/dlami/nvme/zeta/queries/{i}.bin",
            "-t",
            "vector_q",
            "-m",
            "2",
        ]

        process = subprocess.Popen(cmd)
        processes.append(process)
        print(
            f"Started process for file /opt/dlami/nvme/zeta/queries/{i}.bin with PID {process.pid}"
        )

    # 处理信号，确保清理子进程
    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        cleanup_processes(processes)
        sys.exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    try:
        # 等待所有进程完成
        while True:
            all_done = True
            for i, proc in enumerate(processes):
                if proc.poll() is None:  # 进程还在运行
                    all_done = False
                else:
                    # 进程已结束，检查返回码
                    return_code = proc.poll()
                    if return_code != 0:
                        print(f"Process {i} failed with return code {return_code}")
                        # 获取错误输出
                        _, stderr = proc.communicate()
                        if stderr:
                            print(f"Error output: {stderr}")

                        # 当有进程失败时，立即终止所有其他进程
                        print("Terminating all processes due to failure...")
                        cleanup_processes(processes)
                        sys.exit(1)

            if all_done:
                print("All processes completed successfully")
                break

            time.sleep(1)  # 避免过度消耗CPU

    except Exception as e:
        print(f"Error occurred: {e}")
        cleanup_processes(processes)
        sys.exit(1)

    finally:
        cleanup_processes(processes)


def cleanup_processes(processes):
    """清理所有子进程"""
    for proc in processes:
        try:
            if proc.poll() is None:  # 如果进程还在运行
                # 在 Windows 上使用 taskkill
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
                # 在 Unix-like 系统上使用 SIGTERM
                else:
                    proc.terminate()  # 发送 SIGTERM
                    # 给进程一些时间来清理
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()  # 如果进程没有及时响应，强制结束

                print(f"Terminated process {proc.pid}")
        except Exception as e:
            print(f"Error while terminating process {proc.pid}: {e}")


if __name__ == "__main__":
    run_processes()
