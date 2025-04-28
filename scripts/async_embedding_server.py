import os
import threading
import uvicorn
import argparse
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import List, Dict, Any, Optional
from dynamo_utils import batch_get_vectors

assert "AWS_REGION_NAME" in os.environ
region_name = os.environ.get("AWS_REGION_NAME")

app = FastAPI(title="Embedding Service")

# 存储已注册任务、获取的嵌入向量和任务状态
registered_tasks = {}
fetched_embeddings = {}
task_events = {}
lock = threading.Lock()
thread_pool = ThreadPoolExecutor(max_workers=10)

# 定义请求模型
class TaskRequest(BaseModel):
    ids: List[int]
    table_name: str
    model_id: int


def fetch_embeddings_from_dynamodb(table_name: str, model_id: int, ids: List[int]):
    """从DynamoDB获取嵌入向量的后台任务"""
    try:
        embeddings = batch_get_vectors(
            table_name, ids, model_id=model_id, region_name=region_name
        )
        # 存储结果并设置事件标志
        with lock:
            key = f"{table_name}_{model_id}_{','.join(map(str, ids))}"
            fetched_embeddings[key] = embeddings
            if key in task_events:  # 确保任务仍然存在
                task_events[key].set()
    except Exception as e:
        # 记录错误并设置事件以避免死锁
        with lock:
            key = f"{table_name}_{model_id}_{','.join(map(str, ids))}"
            fetched_embeddings[key] = {"error": str(e)}
            if key in task_events:
                task_events[key].set()


@app.post("/register_task")
async def register_task(request: TaskRequest):
    """注册一个获取嵌入向量的任务"""
    key = f"{request.table_name}_{request.model_id}_{','.join(map(str, request.ids))}"

    with lock:
        # 更新或创建任务
        registered_tasks[key] = registered_tasks.get(key, 0) + 1

        # 如果是新任务，创建事件并提交到线程池
        if key not in task_events:
            task_events[key] = Event()
            # 释放锁后提交任务，避免长时间持有锁
            needs_submit = True
        else:
            needs_submit = False

    # 在锁外提交任务
    if needs_submit:
        thread_pool.submit(
            fetch_embeddings_from_dynamodb,
            request.table_name,
            request.model_id,
            request.ids,
        )

    return {"status": "success", "task_id": key}


@app.post("/fetch_embedding")
async def fetch_embedding(request: TaskRequest):
    """获取嵌入向量，如果任务未完成则等待"""
    key = f"{request.table_name}_{request.model_id}_{','.join(map(str, request.ids))}"

    # 检查任务是否存在
    with lock:
        if key not in registered_tasks:
            raise HTTPException(status_code=404, detail="Task not registered")

        # 如果结果已经可用，直接返回
        if key in fetched_embeddings:
            embeddings = fetched_embeddings[key]
            event = None
        else:
            # 获取事件对象，后面在锁外等待
            event = task_events[key]
            embeddings = None

    # 在锁外等待任务完成
    if event:
        event.wait()
        # 等待完成后再次获取结果
        with lock:
            if key in fetched_embeddings:
                embeddings = fetched_embeddings[key]
            else:
                # 如果任务已被清理，返回错误
                raise HTTPException(status_code=500, detail="Task result not available")

    # 减少任务计数并清理完成的任务
    with lock:
        registered_tasks[key] -= 1
        if registered_tasks[key] <= 0:
            # 清理已完成的任务
            registered_tasks.pop(key, None)
            fetched_embeddings.pop(key, None)
            task_events.pop(key, None)

    # 检查结果是否包含错误信息
    if isinstance(embeddings, dict) and "error" in embeddings:
        raise HTTPException(
            status_code=500, detail=f"Task failed: {embeddings['error']}"
        )

    return Response(content=embeddings.tobytes(), media_type="application/octet-stream")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="服务运行的端口号 (默认: 8000)")
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
