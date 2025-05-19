import numpy as np
import os
import logging
import time
from typing import List
import random
import sys

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入两种嵌入服务
from scripts.local_embedding_client import LocalEmbeddingService
from scripts.async_embedding_server import EmbeddingService

def generate_random_ids(count: int, max_id: int) -> List[int]:
    """生成指定数量的随机ID"""
    return random.sample(range(max_id + 1), count)

def compare_embeddings(emb1, emb2) -> bool:
    """比较两个嵌入是否完全一致"""
    if isinstance(emb1, np.ndarray) and isinstance(emb2, np.ndarray):
        if emb1.shape != emb2.shape:
            logger.error(f"嵌入形状不一致: {emb1.shape} vs {emb2.shape}")
            return False
        
        # 检查是否完全相等
        if np.array_equal(emb1, emb2):
            return True
        
        # 如果不完全相等，计算差异
        max_diff = np.max(np.abs(emb1 - emb2))
        logger.error(f"嵌入不完全一致，最大差异: {max_diff}")
        return False
    else:
        # 处理None或其他类型的情况
        equal = emb1 == emb2
        if not equal:
            logger.error(f"嵌入类型不一致: {type(emb1)} vs {type(emb2)}")
        return equal

def test_embedding_services(model_id: int, num_ids: int = 100, max_id: int = 1000000):
    """测试两种嵌入服务返回的结果是否一致"""
    # 设置环境变量（EmbeddingService需要）
    if "AWS_REGION_NAME" not in os.environ:
        os.environ["AWS_REGION_NAME"] = "us-west-2"  # 设置默认区域
    
    # 初始化服务
    logger.info("初始化本地嵌入服务...")
    local_service = LocalEmbeddingService(server_url="http://localhost:8000")
    
    logger.info("初始化DynamoDB嵌入服务...")
    dynamo_service = EmbeddingService()
    
    # 生成随机ID
    logger.info(f"生成{num_ids}个随机ID...")
    random_ids = generate_random_ids(num_ids, max_id)
    logger.info(f"随机ID样本: {random_ids[:5]}...")
    
    # 表名（可以是任意值，因为我们只关心两个服务是否返回相同的结果）
    table_name = "test_table"
    
    try:
        # 从本地服务获取嵌入
        logger.info(f"从本地服务获取model_id={model_id}的嵌入...")
        start_time = time.time()
        local_service.register_task(table_name, model_id, random_ids)
        local_embeddings = local_service.fetch_embedding(table_name, model_id, random_ids)
        logger.info(f"本地服务耗时: {time.time() - start_time:.2f}秒")
        
        # 从DynamoDB服务获取嵌入
        logger.info(f"从DynamoDB服务获取model_id={model_id}的嵌入...")
        start_time = time.time()
        dynamo_service.register_task(table_name, model_id, random_ids)
        dynamo_embeddings = dynamo_service.fetch_embedding(table_name, model_id, random_ids)
        logger.info(f"DynamoDB服务耗时: {time.time() - start_time:.2f}秒")
        
        # 比较结果
        logger.info("比较两种服务返回的嵌入...")
        
        # 检查嵌入数量是否一致
        if len(local_embeddings) != len(dynamo_embeddings):
            logger.error(f"嵌入数量不一致: 本地服务 {len(local_embeddings)} vs DynamoDB服务 {len(dynamo_embeddings)}")
            return False
        
        # 逐个比较嵌入
        all_equal = True
        for i, (local_emb, dynamo_emb) in enumerate(zip(local_embeddings, dynamo_embeddings)):
            if not compare_embeddings(local_emb, dynamo_emb):
                logger.error(f"ID {random_ids[i]} 的嵌入不一致")
                all_equal = False
                break
        
        if all_equal:
            logger.info("测试通过！两种服务返回的嵌入完全一致。")
            return True
        else:
            logger.error("测试失败！两种服务返回的嵌入不一致。")
            return False
    
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        return False
    finally:
        # 清理资源
        logger.info("清理资源...")
        local_service.shutdown()
        dynamo_service.shutdown()

if __name__ == "__main__":
    # 解析命令行参数
    model_id = 4  # 默认使用model_id=4
    if len(sys.argv) > 1:
        try:
            model_id = int(sys.argv[1])
        except ValueError:
            logger.error(f"无效的model_id: {sys.argv[1]}，使用默认值4")
    
    # 运行测试
    logger.info(f"开始测试，使用model_id={model_id}")
    success = test_embedding_services(model_id)
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1)
