"""Lightweight schema sync for SQLite.

`Base.metadata.create_all()` only creates tables that don't exist yet - it never
adds columns to a table that's already there. Since this project has no
Alembic setup, new nullable columns added to a model would silently go
missing on any database file that predates the model change (exactly what
happened when evaluations.db was copied to the server before it existed
there). This module diffs the ORM models against the actual DB schema and
adds any missing nullable columns via ALTER TABLE, preserving existing rows.

Adding a NOT NULL column or dropping/renaming one still needs a real
migration - this only handles the common "I added an optional column" case.
"""

import logging

from sqlalchemy import MetaData, inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def sync_schema(engine: Engine, metadata: MetaData) -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all() already handles brand-new tables

            existing_columns = {
                col["name"] for col in inspector.get_columns(table.name)
            }

            for column in table.columns:
                if column.name in existing_columns:
                    continue

                if not column.nullable:
                    logger.warning(
                        "Column %s.%s is new and NOT NULL - skipping "
                        "auto-migration, add it manually.",
                        table.name,
                        column.name,
                    )
                    continue

                column_type = column.type.compile(dialect=engine.dialect)
                conn.exec_driver_sql(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'
                )
                logger.info("Added missing column %s.%s", table.name, column.name)
