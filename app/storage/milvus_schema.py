"""Milvus collection schema 定义"""

from pymilvus import CollectionSchema, FieldSchema, DataType

COLLECTION_NAME = "kb_vectors"

# 字段定义
FIELDS = [
    FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        max_length=64,
        description="UUID primary key",
    ),
    FieldSchema(
        name="dataset_id",
        dtype=DataType.VARCHAR,
        max_length=64,
        is_partition_key=True,
        description="Maps to kb_datasets.id, partition key for agent isolation",
    ),
    FieldSchema(
        name="collection_id",
        dtype=DataType.VARCHAR,
        max_length=64,
        description="Maps to kb_collections.id, groups chunks from same source",
    ),
    FieldSchema(
        name="chunk_index",
        dtype=DataType.INT32,
        description="Order of this chunk within the source document",
    ),
    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=16384,
        description="The actual text content of this chunk",
    ),
    FieldSchema(
        name="source_type",
        dtype=DataType.VARCHAR,
        max_length=16,
        description="text | link | file",
    ),
    FieldSchema(
        name="source_name",
        dtype=DataType.VARCHAR,
        max_length=512,
        description="Original filename, URL, or text collection name",
    ),
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=1536,
        description="OpenAI text-embedding-3-small vector",
    ),
    FieldSchema(
        name="created_at",
        dtype=DataType.INT64,
        description="Unix timestamp of insertion",
    ),
]

# Collection schema
SCHEMA = CollectionSchema(
    fields=FIELDS,
    description="Knowledge base vectors for TG-Agent RAG",
)

# 索引配置
INDEX_PARAMS = {
    "field_name": "embedding",
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}

# 搜索参数
SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16},
}
