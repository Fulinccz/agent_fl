"""init tables

Revision ID: 001
Revises:
Create Date: 2026-05-04

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 职位表
    op.create_table(
        'job_descriptions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('job_id', sa.String(64), nullable=False, unique=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('company', sa.String(255)),
        sa.Column('salary', sa.String(100)),
        sa.Column('location', sa.String(255)),
        sa.Column('tags', sa.JSON()),
        sa.Column('jd', sa.Text()),
        sa.Column('source', sa.String(50)),
        sa.Column('source_url', sa.String(500)),
        sa.Column('raw_data', sa.JSON()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        mysql_charset='utf8mb4',
        mysql_collate='utf8mb4_unicode_ci',
        comment='职位信息表'
    )
    op.create_index('idx_title', 'job_descriptions', ['title'])
    op.create_index('idx_company', 'job_descriptions', ['company'])
    op.create_index('idx_source', 'job_descriptions', ['source'])
    op.create_index('idx_created_at', 'job_descriptions', ['created_at'])

    # 简历表
    op.create_table(
        'resumes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('resume_id', sa.String(64), nullable=False, unique=True),
        sa.Column('gender', sa.String(10)),
        sa.Column('age', sa.Integer()),
        sa.Column('target_position', sa.String(255)),
        sa.Column('degree', sa.String(50)),
        sa.Column('university_type', sa.String(50)),
        sa.Column('work_description', sa.Text()),
        sa.Column('project_description', sa.Text()),
        sa.Column('source', sa.String(50), server_default='tianchi'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        mysql_charset='utf8mb4',
        mysql_collate='utf8mb4_unicode_ci',
        comment='简历信息表'
    )
    op.create_index('idx_resume_id', 'resumes', ['resume_id'])
    op.create_index('idx_target_position', 'resumes', ['target_position'])


def downgrade() -> None:
    op.drop_table('resumes')
    op.drop_table('job_descriptions')
