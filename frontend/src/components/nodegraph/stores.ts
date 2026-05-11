import type { Schema } from "#/service/compute";

/*
STORE_SCHEMAS defines the static schema catalog for all supported data stores.
Each store exposes a single store_ref output so it can be wired into any
operation that reads from or writes to persistent storage.
*/
export const STORE_SCHEMAS: Record<string, Schema> = {
	s3: {
		kind: "Store",
		category: "stores",
		op: "s3",
		name: "S3",
		label: "S3",
		description: "Amazon S3 object storage bucket",
		initial_width: 220,
		inputs: [],
		outputs: [{ name: "ref", type: "store_ref", description: "S3 bucket reference" }],
		config: [
			{ name: "bucket", type: "string", description: "Bucket name" },
			{ name: "region", type: "string", description: "AWS region" },
			{ name: "prefix", type: "string", description: "Key prefix" },
		],
	},
	deeplake: {
		kind: "Store",
		category: "stores",
		op: "deeplake",
		name: "DeepLake",
		label: "DeepLake",
		description: "Activeloop DeepLake tensor dataset",
		initial_width: 220,
		inputs: [],
		outputs: [{ name: "ref", type: "store_ref", description: "Dataset reference" }],
		config: [
			{ name: "path", type: "string", description: "Dataset path or hub URL" },
			{ name: "token", type: "string", description: "Activeloop token" },
		],
	},
	qdrant: {
		kind: "Store",
		category: "stores",
		op: "qdrant",
		name: "Qdrant",
		label: "Qdrant",
		description: "Qdrant vector database collection",
		initial_width: 220,
		inputs: [],
		outputs: [{ name: "ref", type: "store_ref", description: "Collection reference" }],
		config: [
			{ name: "host", type: "string", description: "Qdrant host" },
			{ name: "port", type: "int", default: 6333, description: "gRPC port" },
			{ name: "collection", type: "string", description: "Collection name" },
		],
	},
	lakefs: {
		kind: "Store",
		category: "stores",
		op: "lakefs",
		name: "LakeFS",
		label: "LakeFS",
		description: "LakeFS versioned data lake repository",
		initial_width: 220,
		inputs: [],
		outputs: [{ name: "ref", type: "store_ref", description: "Repository reference" }],
		config: [
			{ name: "endpoint", type: "string", description: "LakeFS endpoint URL" },
			{ name: "repository", type: "string", description: "Repository name" },
			{ name: "branch", type: "string", default: "main", description: "Branch" },
		],
	},
	neo4j: {
		kind: "Store",
		category: "stores",
		op: "neo4j",
		name: "Neo4j",
		label: "Neo4j",
		description: "Neo4j graph database",
		initial_width: 220,
		inputs: [],
		outputs: [{ name: "ref", type: "store_ref", description: "Graph database reference" }],
		config: [
			{ name: "uri", type: "string", description: "Bolt URI" },
			{ name: "database", type: "string", default: "neo4j", description: "Database name" },
		],
	},
	elasticsearch: {
		kind: "Store",
		category: "stores",
		op: "elasticsearch",
		name: "ElasticSearch",
		label: "ElasticSearch",
		description: "Elasticsearch index",
		initial_width: 220,
		inputs: [],
		outputs: [{ name: "ref", type: "store_ref", description: "Index reference" }],
		config: [
			{ name: "hosts", type: "string", description: "Comma-separated host list" },
			{ name: "index", type: "string", description: "Index name" },
		],
	},
	postgres: {
		kind: "Store",
		category: "stores",
		op: "postgres",
		name: "Postgres",
		label: "Postgres",
		description: "PostgreSQL relational database",
		initial_width: 220,
		inputs: [],
		outputs: [{ name: "ref", type: "store_ref", description: "Database reference" }],
		config: [
			{ name: "dsn", type: "string", description: "Connection string (DSN)" },
			{ name: "schema", type: "string", default: "public", description: "Schema" },
		],
	},
};
