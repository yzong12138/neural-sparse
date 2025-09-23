import boto3
import numpy as np
import time


def batch_store_vectors_binary(
    table_name,
    vector_ids,
    vectors,
    model_id=0,
    max_batch_size=25,
    region_name="ap-northeast-1",
):
    if len(vector_ids) != vectors.shape[0]:
        raise ValueError(
            f"Vector IDs {len(vector_ids)} not equal to vectors {vectors.shape[0]}"
        )

    total_vectors = len(vector_ids)
    dynamodb_client = boto3.client("dynamodb", region_name=region_name)

    for batch_start in range(0, total_vectors, max_batch_size):
        batch_end = min(batch_start + max_batch_size, total_vectors)
        batch_items = []

        for i in range(batch_start, batch_end):
            vector = vectors[i]
            vector_bytes = vector.tobytes()

            item = {
                "PutRequest": {
                    "Item": {
                        "text_id": {"N": str(vector_ids[i])},
                        "model_id": {"N": str(model_id)},
                        "vector": {"B": vector_bytes},
                    }
                }
            }
            batch_items.append(item)

        if batch_items:
            response = dynamodb_client.batch_write_item(
                RequestItems={table_name: batch_items}
            )

            unprocessed_items = response.get("UnprocessedItems", {})
            retry_attempts = 0
            max_retries = 100

            while unprocessed_items and retry_attempts < max_retries:
                time.sleep(0.5)
                retry_attempts += 1
                print(f"{retry_attempts}次尝试...")
                response = dynamodb_client.batch_write_item(
                    RequestItems=unprocessed_items
                )
                unprocessed_items = response.get("UnprocessedItems", {})

            if unprocessed_items:
                raise Exception(
                    f"警告: 有{len(unprocessed_items.get(table_name, []))}项未能成功写入"
                )


def batch_get_vectors(
    table_name,
    text_id_list,
    model_id=0,
    dtype=np.float16,
    max_batch_size=100,
    region_name="ap-northeast-1",
):
    if not text_id_list:
        return []

    text_id_list = [int(x) for x in text_id_list]
    dynamodb_client = boto3.client("dynamodb", region_name=region_name)
    model_id_str = str(model_id)
    result_vectors = [None] * len(text_id_list)  # 预分配结果列表，保持与输入顺序一致

    # 去重text_id_list，同时保留原始索引信息
    unique_text_ids = []
    original_to_unique_map = {}  # 原始列表索引 -> 去重后列表索引的映射

    for idx, text_id in enumerate(text_id_list):
        if text_id not in original_to_unique_map:
            original_to_unique_map[text_id] = len(unique_text_ids)
            unique_text_ids.append(text_id)

    # 创建一个字典存储获取到的向量，以text_id为键
    id_to_vector = {}

    # 外层按缺失情况重试：如果result_vectors中仍存在None，则仅对缺失ID重试
    max_retries = 10
    retry_attempts = 0
    remaining_ids = list(unique_text_ids)

    while remaining_ids and retry_attempts <= max_retries:
        # 分批获取 remaining_ids
        for batch_start in range(0, len(remaining_ids), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(remaining_ids))
            batch_ids = remaining_ids[batch_start:batch_end]

            keys_list = [
                {"text_id": {"N": str(text_id)}, "model_id": {"N": model_id_str}}
                for text_id in batch_ids
            ]

            response = dynamodb_client.batch_get_item(
                RequestItems={
                    table_name: {
                        "Keys": keys_list,
                        "ConsistentRead": True,
                    }
                }
            )

            if "Responses" in response and table_name in response["Responses"]:
                items = response["Responses"][table_name]
                for item in items:
                    text_id = int(item["text_id"]["N"])
                    vector_binary = item["vector"]["B"]
                    vector = np.frombuffer(vector_binary, dtype=dtype)
                    id_to_vector[text_id] = vector

        # 按照当前已获取结果填充一次，用于判断是否还存在 None
        for idx, text_id in enumerate(text_id_list):
            if text_id in id_to_vector:
                result_vectors[idx] = id_to_vector[text_id]

        # 计算仍缺失的唯一ID
        missing_set = set(unique_text_ids) - set(id_to_vector.keys())
        remaining_ids = list(missing_set)

        if remaining_ids:
            retry_attempts += 1
            print(
                f"重试未获取的向量，第{retry_attempts}次尝试，还剩{len(remaining_ids)}个..."
            )
            time.sleep(0.5)

    if remaining_ids:
        raise Exception(f"警告：有{len(remaining_ids)}个向量未能成功读取")

    # 确保返回顺序与输入一致
    for idx, text_id in enumerate(text_id_list):
        if text_id in id_to_vector:
            result_vectors[idx] = id_to_vector[text_id]

    return np.array(result_vectors)
