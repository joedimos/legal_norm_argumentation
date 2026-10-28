"""Indexes for private queue listings and ordered pending-job selection."""
from alembic import op

revision = "0003_workflow_indexes"
down_revision = "0002_governed_workflows"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index("ix_workflow_owner_created", "workflow_requests", ["created_by", "created_at"])
    op.create_index("ix_workflow_kind_created", "workflow_requests", ["kind", "created_at"])


def downgrade():
    op.drop_index("ix_workflow_kind_created", table_name="workflow_requests")
    op.drop_index("ix_workflow_owner_created", table_name="workflow_requests")
