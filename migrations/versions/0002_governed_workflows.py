"""Durable, append-only inference and independent-review workflows."""
from alembic import op
import sqlalchemy as sa

revision = "0002_governed_workflows"
down_revision = "0001_production_service"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("workflow_requests",
                    sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("kind", sa.String(32), nullable=False),
                    sa.Column("created_by", sa.String(128), nullable=False),
                    sa.Column("request_hash", sa.String(64), nullable=False),
                    sa.Column("request_payload", sa.JSON(), nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.UniqueConstraint("kind", "created_by", "request_hash", name="uq_workflow_permanent_dedup"))
    op.create_table("workflow_events",
                    sa.Column("sequence", sa.Integer(), primary_key=True, autoincrement=True),
                    sa.Column("workflow_id", sa.String(36), sa.ForeignKey("workflow_requests.id", ondelete="RESTRICT"), nullable=False),
                    sa.Column("slot", sa.String(64), nullable=False),
                    sa.Column("actor", sa.String(128), nullable=False),
                    sa.Column("payload", sa.JSON(), nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.UniqueConstraint("workflow_id", "slot", name="uq_workflow_event_slot"))
    op.create_index("ix_workflow_events_workflow_id", "workflow_events", ["workflow_id"])
    if op.get_bind().dialect.name == "postgresql":
        for table in ("workflow_requests", "workflow_events"):
            op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
                       "FOR EACH ROW EXECUTE FUNCTION vle_reject_mutation()")


def downgrade():
    op.drop_table("workflow_events")
    op.drop_table("workflow_requests")
