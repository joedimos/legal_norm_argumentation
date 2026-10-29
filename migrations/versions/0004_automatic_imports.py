"""Durable resumable IP-source import state, separate from immutable evidence."""
from alembic import op
import sqlalchemy as sa

revision = "0004_automatic_imports"
down_revision = "0003_workflow_indexes"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("import_cursors",
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("position", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_token", sa.String(36)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("last_success_at", sa.DateTime(timezone=True)),
        sa.Column("report", sa.JSON(), nullable=False))
    op.create_index("ix_import_cursors_due_at", "import_cursors", ["due_at"])


def downgrade():
    op.drop_index("ix_import_cursors_due_at", table_name="import_cursors")
    op.drop_table("import_cursors")
