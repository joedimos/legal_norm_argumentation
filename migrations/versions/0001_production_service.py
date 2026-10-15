"""production service persistence and immutable audit trail

Revision ID: 0001_production_service
Revises:
Create Date: 2026-10-14
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_production_service"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "source_documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source_key", sa.String(256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("candidates", sa.JSON(), nullable=False),
        sa.Column("extraction_audit", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("source_key", "content_hash", name="uq_source_key_content_hash"),
    )
    op.create_index("ix_source_documents_source_key", "source_documents", ["source_key"])
    op.create_index("ix_source_documents_content_hash", "source_documents", ["content_hash"])
    op.create_index("ix_source_documents_created_by", "source_documents", ["created_by"])

    op.create_table(
        "candidate_reviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source_id", sa.String(36), sa.ForeignKey("source_documents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("candidate_id", sa.String(128), nullable=False),
        sa.Column("candidate_fingerprint", sa.String(64), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("reviewer", sa.String(128), nullable=False),
        sa.Column("note", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_candidate_reviews_source_id", "candidate_reviews", ["source_id"])
    op.create_index("ix_candidate_reviews_candidate_id", "candidate_reviews", ["candidate_id"])
    op.create_index("ix_candidate_reviews_reviewer", "candidate_reviews", ["reviewer"])
    op.create_index("ix_candidate_reviews_created_at", "candidate_reviews", ["created_at"])
    op.create_index("ix_candidate_review_latest", "candidate_reviews", ["source_id", "candidate_id", "created_at"])

    op.create_table(
        "interpretation_selections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source_id", sa.String(36), sa.ForeignKey("source_documents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("candidate_id", sa.String(128), nullable=False),
        sa.Column("candidate_fingerprint", sa.String(64), nullable=False),
        sa.Column("interpretation_id", sa.String(128), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("reviewer", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_interpretation_selections_source_id", "interpretation_selections", ["source_id"])
    op.create_index("ix_interpretation_selections_candidate_id", "interpretation_selections", ["candidate_id"])
    op.create_index("ix_interpretation_selections_reviewer", "interpretation_selections", ["reviewer"])
    op.create_index("ix_interpretation_selections_created_at", "interpretation_selections", ["created_at"])
    op.create_index("ix_interpretation_selection_latest", "interpretation_selections", ["source_id", "candidate_id", "created_at"])

    op.create_table(
        "lifecycle_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source_id", sa.String(36), sa.ForeignKey("source_documents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("requested_by", sa.String(128), nullable=False),
        sa.Column("high_stakes", sa.Boolean(), nullable=False),
        sa.Column("decision_authorized", sa.Boolean(), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("context_payload", sa.JSON(), nullable=False),
        sa.Column("result_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lifecycle_runs_source_id", "lifecycle_runs", ["source_id"])
    op.create_index("ix_lifecycle_runs_requested_by", "lifecycle_runs", ["requested_by"])
    op.create_index("ix_lifecycle_runs_request_hash", "lifecycle_runs", ["request_hash"])
    op.create_index("ix_lifecycle_runs_created_at", "lifecycle_runs", ["created_at"])

    op.create_table(
        "audit_events",
        sa.Column("sequence", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("resource_id", sa.String(128), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("previous_hash", sa.String(64)),
        sa.Column("event_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("actor", "action", "resource_id", "request_id", "created_at"):
        op.create_index(f"ix_audit_events_{column}", "audit_events", [column])

    op.create_table(
        "idempotency_records",
        sa.Column("key_hash", sa.String(64), primary_key=True),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("endpoint", sa.String(256), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("response_payload", sa.JSON(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_idempotency_records_actor", "idempotency_records", ["actor"])
    op.create_index("ix_idempotency_records_expires_at", "idempotency_records", ["expires_at"])

    op.create_table(
        "rate_limit_buckets",
        sa.Column("key_hash", sa.String(64), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("count", sa.Integer(), nullable=False),
    )

    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            """
            CREATE OR REPLACE FUNCTION vle_reject_mutation() RETURNS trigger AS $$
            BEGIN
              RAISE EXCEPTION 'append-only table % cannot be updated or deleted', TG_TABLE_NAME;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        for table in ("source_documents", "candidate_reviews", "interpretation_selections", "lifecycle_runs", "audit_events"):
            op.execute(
                f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION vle_reject_mutation();"
            )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for table in ("source_documents", "candidate_reviews", "interpretation_selections", "lifecycle_runs", "audit_events"):
            op.execute(f"DROP TRIGGER IF EXISTS {table}_append_only ON {table};")
        op.execute("DROP FUNCTION IF EXISTS vle_reject_mutation();")
    op.drop_table("rate_limit_buckets")
    op.drop_table("idempotency_records")
    op.drop_table("audit_events")
    op.drop_table("lifecycle_runs")
    op.drop_table("interpretation_selections")
    op.drop_table("candidate_reviews")
    op.drop_table("source_documents")
