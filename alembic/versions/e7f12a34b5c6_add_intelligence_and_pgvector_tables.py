"""add intelligence and pgvector tables

Revision ID: e7f12a34b5c6
Revises: 83b00afb8ca1
Create Date: 2026-10-27 01:36:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e7f12a34b5c6'
down_revision: Union[str, Sequence[str], None] = '83b00afb8ca1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Safely query if vector extension is installed on PostgreSQL system
    res = conn.execute(sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")).fetchone()
    has_vector = bool(res)

    if has_vector:
        try:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
            from pgvector.sqlalchemy import Vector
            embedding_type = Vector(768)
        except Exception:
            embedding_type = sa.Text()
    else:
        embedding_type = sa.Text()

    # Add rating and review_count to leads if not existing
    insp = sa.inspect(conn)
    existing_cols = [c['name'] for c in insp.get_columns('leads')]

    if 'rating' not in existing_cols:
        op.add_column('leads', sa.Column('rating', sa.Float(), nullable=True))

    if 'review_count' not in existing_cols:
        op.add_column('leads', sa.Column('review_count', sa.Integer(), nullable=True))

    existing_tables = insp.get_table_names()

    # Create business_analyses if not existing
    if 'business_analyses' not in existing_tables:
        op.create_table(
            'business_analyses',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('lead_id', sa.Integer(), sa.ForeignKey('leads.id', ondelete='CASCADE'), nullable=False),
            sa.Column('services_found', sa.Text(), nullable=True),
            sa.Column('has_online_booking', sa.Boolean(), server_default='false', nullable=True),
            sa.Column('has_whatsapp', sa.Boolean(), server_default='false', nullable=True),
            sa.Column('has_contact_form', sa.Boolean(), server_default='false', nullable=True),
            sa.Column('pricing_found', sa.Boolean(), server_default='false', nullable=True),
            sa.Column('social_links', sa.Text(), nullable=True),
            sa.Column('website_summary', sa.Text(), nullable=True),
            sa.Column('crawl_raw', sa.Text(), nullable=True),
            sa.Column('crawled_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        )

    # Create business_embeddings if not existing
    if 'business_embeddings' not in existing_tables:
        op.create_table(
            'business_embeddings',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('lead_id', sa.Integer(), sa.ForeignKey('leads.id', ondelete='CASCADE'), nullable=False),
            sa.Column('content_type', sa.String(length=50), nullable=False),
            sa.Column('content_text', sa.Text(), nullable=False),
            sa.Column('embedding', embedding_type, nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        )

    # Create lead_scores if not existing
    if 'lead_scores' not in existing_tables:
        op.create_table(
            'lead_scores',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('lead_id', sa.Integer(), sa.ForeignKey('leads.id', ondelete='CASCADE'), unique=True, nullable=False),
            sa.Column('score', sa.Float(), nullable=True),
            sa.Column('priority', sa.String(length=10), nullable=True),
            sa.Column('opportunities', sa.Text(), nullable=True),
            sa.Column('recommended_actions', sa.Text(), nullable=True),
            sa.Column('pitch_whatsapp', sa.Text(), nullable=True),
            sa.Column('pitch_email', sa.Text(), nullable=True),
            sa.Column('scored_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        )


def downgrade() -> None:
    op.drop_table('lead_scores')
    op.drop_table('business_embeddings')
    op.drop_table('business_analyses')
    op.drop_column('leads', 'review_count')
    op.drop_column('leads', 'rating')
