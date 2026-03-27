import json
import os
from typing import List, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from psycopg_pool import ConnectionPool


def _normalize_dsn(dsn: str) -> str:
    """Remove query params psycopg doesn't recognize (e.g., pgbouncer)."""
    parsed = urlparse(dsn)
    if not parsed.query:
        return dsn
    kept = [(k, v) for (k, v) in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "pgbouncer"]
    new_query = urlencode(kept)
    return urlunparse(parsed._replace(query=new_query))


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

# pgbouncer + psycopg: disable prepared statements
_POOL_KWARGS = {"prepare_threshold": None}


def _configure_conn(conn) -> None:
    try:
        conn.prepare_threshold = None
    except Exception:
        pass


pool = ConnectionPool(
    _normalize_dsn(DATABASE_URL),
    min_size=1,
    max_size=5,
    kwargs=_POOL_KWARGS,
    configure=_configure_conn,
)

# database schema start
def init_db() -> None:
    direct_url = os.getenv("DIRECT_URL")
    dsn = _normalize_dsn(direct_url) if direct_url else _normalize_dsn(DATABASE_URL)
    ddl = [
        "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'domain') THEN
                CREATE TYPE domain AS ENUM ('ICFR', 'AARAMBH', 'NGO', 'TAXLEGIT', 'CSR', 'DASHBOARD');
            END IF;
        END $$;
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL DEFAULT ''
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_file_name TEXT,
            uploaded_by_email TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
            domains domain[] NOT NULL DEFAULT '{}',
            index_array TEXT[] NOT NULL DEFAULT '{}',
            summarization BOOLEAN NOT NULL DEFAULT TRUE,
            tree_json JSONB NOT NULL,
            doc_summary TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS index_array TEXT[] NOT NULL DEFAULT '{}';",
        "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS summarization BOOLEAN NOT NULL DEFAULT TRUE;",
        "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS company_types TEXT[] NOT NULL DEFAULT '{}';",
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'rag_documents' AND column_name = 'company_type'
            ) THEN
                UPDATE rag_documents
                SET company_types = ARRAY[LOWER(company_type)]
                WHERE company_type IS NOT NULL AND company_types = '{}';
                ALTER TABLE rag_documents DROP COLUMN company_type;
            END IF;
        END $$;
        """,
        "DROP TRIGGER IF EXISTS set_updated_at_rag_documents ON rag_documents;",
        """
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        """
        CREATE TRIGGER set_updated_at_rag_documents
        BEFORE UPDATE ON rag_documents
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """,
    ]

    with ConnectionPool(
        dsn,
        min_size=1,
        max_size=1,
        kwargs=_POOL_KWARGS,
        configure=_configure_conn,
    ).connection() as conn:
        with conn.cursor() as cur:
            for stmt in ddl:
                cur.execute(stmt)
        conn.commit()
# databse schema end

def _domains_literal(domains: Optional[List[str]]) -> str:
    if not domains:
        return "{}"
    cleaned = [d.strip().upper() for d in domains if d and d.strip()]
    if not cleaned:
        return "{}"
    return "{" + ",".join(cleaned) + "}"


def _clean_index_array(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    return [v.strip() for v in values if v and v.strip()]


def _clean_company_types(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    return [v.strip().lower() for v in values if v and v.strip()]


def insert_rag_document(
    *,
    source_file_name: Optional[str],
    uploaded_by_email: str,
    domains: Optional[List[str]],
    index_array: Optional[List[str]] = None,
    summarization: bool = True,
    tree_json: dict,
    doc_summary: Optional[str] = None,
    company_types: Optional[List[str]] = None,
) -> str:
    domains_literal = _domains_literal(domains)
    tree_json_str = json.dumps(tree_json, ensure_ascii=False)
    cleaned_index_array = _clean_index_array(index_array)
    cleaned_company_types = _clean_company_types(company_types)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, password) VALUES (%s, '') ON CONFLICT (email) DO NOTHING;",
                (uploaded_by_email,),
            )
            cur.execute(
                """
                INSERT INTO rag_documents (
                    source_file_name, uploaded_by_email, domains, index_array, summarization, tree_json, doc_summary, company_types
                ) VALUES (
                    %s, %s, %s::domain[], %s::text[], %s, %s::jsonb, %s, %s::text[]
                ) RETURNING id;
                """,
                (
                    source_file_name,
                    uploaded_by_email,
                    domains_literal,
                    cleaned_index_array,
                    summarization,
                    tree_json_str,
                    doc_summary,
                    cleaned_company_types,
                ),
            )
            doc_id = cur.fetchone()[0]
        conn.commit()

    return str(doc_id)


def get_rag_document_tree(doc_id: str) -> Optional[dict]:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tree_json FROM rag_documents WHERE id = %s;",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return row[0]


def get_latest_rag_document_by_domains(domains: Optional[List[str]]) -> Optional[tuple[str, dict]]:
    if not domains:
        return None
    domains_literal = _domains_literal(domains)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, tree_json
                FROM rag_documents
                WHERE domains && %s::domain[]
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (domains_literal,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return str(row[0]), row[1]


def get_all_rag_documents_by_domains(domains: Optional[List[str]]) -> List[tuple[str, dict]]:
    if not domains:
        return []
    domains_literal = _domains_literal(domains)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, tree_json
                FROM rag_documents
                WHERE domains && %s::domain[]
                ORDER BY created_at DESC;
                """,
                (domains_literal,),
            )
            rows = cur.fetchall()
            return [(str(row[0]), row[1]) for row in rows]


def get_all_rag_documents_with_meta_by_domains(
    domains: Optional[List[str]],
    company_type: Optional[str] = None,
) -> List[tuple[str, dict, List[str], Optional[str]]]:
    if not domains:
        return []
    domains_literal = _domains_literal(domains)
    clean_company_type = company_type.strip().lower() if company_type and company_type.strip() else None

    with pool.connection() as conn:
        with conn.cursor() as cur:
            if clean_company_type:
                cur.execute(
                    """
                    SELECT id, tree_json, index_array, doc_summary
                    FROM rag_documents
                    WHERE domains && %s::domain[]
                      AND (company_types = '{}' OR %s = ANY(company_types))
                    ORDER BY
                      CASE WHEN %s = ANY(company_types) THEN 0 ELSE 1 END,
                      created_at DESC;
                    """,
                    (domains_literal, clean_company_type, clean_company_type),
                )
            else:
                cur.execute(
                    """
                    SELECT id, tree_json, index_array, doc_summary
                    FROM rag_documents
                    WHERE domains && %s::domain[]
                    ORDER BY created_at DESC;
                    """,
                    (domains_literal,),
                )
            rows = cur.fetchall()
            return [
                (str(row[0]), row[1], row[2] or [], row[3])
                for row in rows
            ]
