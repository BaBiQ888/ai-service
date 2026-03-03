"""知识库 CRUD 端点 - Dataset + Collection 管理"""

import json
import logging

from fastapi import APIRouter, Depends, UploadFile, File, Form

from app.dependencies import verify_api_key
from app.models.requests import (
    CreateDatasetRequest,
    AddTextCollectionRequest,
    AddLinkCollectionRequest,
)
from app.models.responses import APIResponseModel
from app.core.errors import IngestionError, MilvusError
from app.ingestion.pipeline import ingest_text, ingest_file, ingest_link
from app.storage.milvus_client import (
    delete_by_collection_id,
    delete_by_dataset_id,
    count_by_dataset_id,
    count_by_collection_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/datasets", response_model=APIResponseModel)
async def create_dataset(
    req: CreateDatasetRequest,
    _api_key: str = Depends(verify_api_key),
):
    """创建知识库 dataset（Milvus 使用 partition key，无需显式创建）"""
    logger.info(f"Create dataset: id={req.dataset_id}, name={req.name}")

    # Milvus 使用 dataset_id 作为 partition key，数据插入时自动分区
    # 这里只做逻辑确认，不需要在 Milvus 创建额外结构
    return APIResponseModel(
        code=0,
        message="success",
        data={
            "dataset_id": req.dataset_id,
            "status": "created",
        },
    )


@router.get("/datasets/{dataset_id}", response_model=APIResponseModel)
async def get_dataset(
    dataset_id: str,
    _api_key: str = Depends(verify_api_key),
):
    """查询 dataset 信息"""
    logger.info(f"Get dataset: {dataset_id}")

    try:
        chunk_count = count_by_dataset_id(dataset_id)
        return APIResponseModel(
            code=0,
            message="success",
            data={
                "dataset_id": dataset_id,
                "chunk_count": chunk_count,
                "status": "active",
            },
        )
    except MilvusError as e:
        logger.error(f"Error getting dataset info: {e}")
        return APIResponseModel(code=500, message=str(e), data=None)


@router.delete("/datasets/{dataset_id}", response_model=APIResponseModel)
async def delete_dataset(
    dataset_id: str,
    _api_key: str = Depends(verify_api_key),
):
    """删除 dataset 及其所有向量"""
    logger.info(f"Delete dataset: {dataset_id}")

    try:
        count = delete_by_dataset_id(dataset_id)
        return APIResponseModel(
            code=0,
            message="success",
            data={
                "dataset_id": dataset_id,
                "deleted_count": count,
                "status": "deleted",
            },
        )
    except MilvusError as e:
        logger.error(f"Error deleting dataset: {e}")
        return APIResponseModel(code=500, message=str(e), data=None)


@router.post("/collections/text", response_model=APIResponseModel)
async def add_text_collection(
    req: AddTextCollectionRequest,
    _api_key: str = Depends(verify_api_key),
):
    """摄入文本内容 → 分块 → 向量化 → 存储"""
    logger.info(
        f"Add text collection: dataset={req.dataset_id}, "
        f"name={req.name}, text_len={len(req.text)}"
    )

    try:
        result = ingest_text(
            dataset_id=req.dataset_id,
            collection_id=req.collection_id,
            name=req.name,
            text=req.text,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        return APIResponseModel(code=0, message="success", data=result)

    except IngestionError as e:
        logger.error(f"Text ingestion error: {e}")
        return APIResponseModel(code=500, message=str(e), data=None)
    except Exception as e:
        logger.error(f"Unexpected error in text ingestion: {e}")
        return APIResponseModel(code=500, message=f"Internal error: {e}", data=None)


@router.post("/collections/link", response_model=APIResponseModel)
async def add_link_collection(
    req: AddLinkCollectionRequest,
    _api_key: str = Depends(verify_api_key),
):
    """摄入网页链接 → 爬取 → 分块 → 向量化 → 存储"""
    logger.info(
        f"Add link collection: dataset={req.dataset_id}, url={req.url}")

    try:
        result = ingest_link(
            dataset_id=req.dataset_id,
            collection_id=req.collection_id,
            url=req.url,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        return APIResponseModel(code=0, message="success", data=result)

    except IngestionError as e:
        logger.error(f"Link ingestion error: {e}")
        return APIResponseModel(code=500, message=str(e), data=None)
    except Exception as e:
        logger.error(f"Unexpected error in link ingestion: {e}")
        return APIResponseModel(code=500, message=f"Internal error: {e}", data=None)


@router.post("/collections/file", response_model=APIResponseModel)
async def add_file_collection(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    _api_key: str = Depends(verify_api_key),
):
    """摄入文件 → 解析 → 分块 → 向量化 → 存储"""
    logger.info(
        f"Add file collection: filename={file.filename}, metadata={metadata}")

    try:
        meta = json.loads(metadata)
        dataset_id = meta.get("dataset_id", "")
        collection_id = meta.get("collection_id", "")
        chunk_size = meta.get("chunk_size", 512)
        chunk_overlap = meta.get("chunk_overlap", 50)

        if not dataset_id or not collection_id:
            return APIResponseModel(
                code=400,
                message="metadata must contain dataset_id and collection_id",
                data=None,
            )

        # 读取文件数据
        file_data = await file.read()

        # 检查文件大小
        from app.config import get_settings
        settings = get_settings()
        if len(file_data) > settings.MAX_FILE_SIZE:
            return APIResponseModel(
                code=400,
                message=f"File too large. Max size: {settings.MAX_FILE_SIZE // (1024*1024)}MB",
                data=None,
            )

        result = ingest_file(
            dataset_id=dataset_id,
            collection_id=collection_id,
            file_data=file_data,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return APIResponseModel(code=0, message="success", data=result)

    except IngestionError as e:
        logger.error(f"File ingestion error: {e}")
        return APIResponseModel(code=500, message=str(e), data=None)
    except json.JSONDecodeError:
        return APIResponseModel(code=400, message="Invalid metadata JSON", data=None)
    except Exception as e:
        logger.error(f"Unexpected error in file ingestion: {e}")
        return APIResponseModel(code=500, message=f"Internal error: {e}", data=None)


@router.delete("/collections/{collection_id}", response_model=APIResponseModel)
async def delete_collection(
    collection_id: str,
    _api_key: str = Depends(verify_api_key),
):
    """删除 collection 的所有向量"""
    logger.info(f"Delete collection: {collection_id}")

    try:
        count = delete_by_collection_id(collection_id)
        return APIResponseModel(
            code=0,
            message="success",
            data={
                "collection_id": collection_id,
                "deleted_count": count,
                "status": "deleted",
            },
        )
    except MilvusError as e:
        logger.error(f"Error deleting collection: {e}")
        return APIResponseModel(code=500, message=str(e), data=None)


@router.get("/collections/{dataset_id}", response_model=APIResponseModel)
async def list_collections(
    dataset_id: str,
    _api_key: str = Depends(verify_api_key),
):
    """列出 dataset 下的所有 collections"""
    logger.info(f"List collections for dataset: {dataset_id}")

    try:
        chunk_count = count_by_dataset_id(dataset_id)
        return APIResponseModel(
            code=0,
            message="success",
            data={
                "dataset_id": dataset_id,
                "total_chunks": chunk_count,
            },
        )
    except MilvusError as e:
        logger.error(f"Error listing collections: {e}")
        return APIResponseModel(code=500, message=str(e), data=None)
