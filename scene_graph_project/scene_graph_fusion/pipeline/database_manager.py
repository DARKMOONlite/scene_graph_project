#!/usr/bin/env python3
#! This was originally written for the neurosymbolic_ILP project, its dependencies have been stripped and its been added here to make running code easier.

import sqlite3
import os
import sys
import json
from typing import TypedDict,Any
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
# Ensure the project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

POPPER_ROOT = "/home/sebastian/Documents/scene_graph/inductive_logic_programming/inductive_logic_programming/datasets/hl_coco"

class DBRow(dict):
    pass
class DatabaseManager:
    def __init__(self, db_path:str|os.PathLike[str]="db/similarity_results.db"):
        """Open a SQLite connection and initialise tables."""
        self.db_path = db_path
        try:
            self.con = sqlite3.connect(db_path)
            self.cur = self.con.cursor()
        except sqlite3.Error as e:
            raise ValueError(f"Error connecting to database at {db_path}: {e}")


    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        """Validate a SQL identifier and return a quoted variant.
        e.g.
        """
        if identifier is None:
            raise ValueError("SQL identifier cannot be None.")
        identifier = identifier.strip()
        if identifier == "":
            return '""'
        if not (identifier[0].isalpha() or identifier[0] == "_"):
            raise ValueError(f"Invalid SQL identifier '{identifier}'.")
        if not all(ch.isalnum() or ch == "_" for ch in identifier):
            raise ValueError(f"Invalid SQL identifier '{identifier}'.")
        return f'"{identifier}"'

    @staticmethod
    def _normalise_sql_value(value: Any) -> Any:
        """Convert complex Python values to SQLite-compatible values."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, bool):
            return int(value)
        return value

    def create_table_from_schema(
        self,
        table_name: str,
        columns: dict[str, str],
        primary_key: tuple[str, ...] | None = None,
        indexes: list[tuple[str, ...]] | None = None,
    ) -> None:
        """Create a table from a caller-provided schema description.

        Args:
            table_name: Name of the table to create.
            columns: Mapping from column name to SQL type/constraint text.
            primary_key: Optional composite primary key column names.
            indexes: Optional list of index column tuples.
        """
        if not columns:
            raise ValueError("At least one column is required to create a table.")

        table_sql = self._validate_identifier(table_name)

        for column_name in columns:
            self._validate_identifier(column_name)

        if primary_key:
            missing_primary_key_columns = [name for name in primary_key if name not in columns]
            if missing_primary_key_columns:
                raise ValueError(
                    f"Primary key columns missing from schema for table '{table_name}': {missing_primary_key_columns}"
                )

        column_defs = ", ".join(
            f"{self._validate_identifier(name)} {sql_type}"
            for name, sql_type in columns.items()
        )

        primary_key_clause = ""
        if primary_key:
            primary_key_cols = ", ".join(self._validate_identifier(name) for name in primary_key)
            primary_key_clause = f", PRIMARY KEY ({primary_key_cols})"

        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_sql} ({column_defs}{primary_key_clause})"
        )

        for index_columns in indexes or []:
            if not index_columns:
                continue
            missing_index_columns = [name for name in index_columns if name not in columns]
            if missing_index_columns:
                raise ValueError(
                    f"Index columns missing from schema for table '{table_name}': {missing_index_columns}"
                )

            index_name = f"idx_{table_name}_{'_'.join(index_columns)}"
            index_sql = self._validate_identifier(index_name)
            index_cols_sql = ", ".join(self._validate_identifier(name) for name in index_columns)
            self.cur.execute(
                f"CREATE INDEX IF NOT EXISTS {index_sql} ON {table_sql} ({index_cols_sql})"
            )
    def add_column_if_not_exists(self, table_name: str, column_name: str, column_type: str) -> None:
        """Add a column to an existing table if it does not already exist."""
        try:
            self.cur.execute(f"ALTER TABLE {self._validate_identifier(table_name)} ADD COLUMN {self._validate_identifier(column_name)} {column_type}")
            self.con.commit()
        except sqlite3.OperationalError as e:
            if f"duplicate column name: {column_name}" in str(e).lower():
                pass  # Column already exists, ignore
            else:
                raise
    def upsert_dict_rows(self, table_name: str, rows: list[dict[str, Any]], append_columns: bool = False) -> None:
        """Insert or replace rows represented as dict objects into a table."""
        if not rows:
            return

        table_sql = self._validate_identifier(table_name)
        columns = list(rows[0].keys())
        if not columns:
            raise ValueError("Rows must contain at least one column.")

        column_set = set(columns)
        for row in rows:
            if set(row.keys()) != column_set:
                raise ValueError(
                    f"All rows must contain the same columns for table '{table_name}'."
                )

        columns_sql = ", ".join(self._validate_identifier(column) for column in columns)
        placeholders = ", ".join("?" for _ in columns)

        normalised_rows = [
            tuple(self._normalise_sql_value(row[column]) for column in columns)
            for row in rows
        ]
        if append_columns:
            for column in columns:
                self.add_column_if_not_exists(table_name, column, "TEXT")
        try:
            self.cur.executemany(
                f"INSERT OR REPLACE INTO {table_sql} ({columns_sql}) VALUES ({placeholders})",
                normalised_rows,
            )
            self.con.commit()
        except sqlite3.OperationalError as e:
            raise ValueError(f"Error inserting rows into table '{table_name}': {e}")

    def create_table_and_upsert_rows(
        self,
        table_name: str,
        columns: dict[str, str],
        rows: list[dict[str, Any]],
        primary_key: tuple[str, ...] | None = None,
        indexes: list[tuple[str, ...]] | None = None,
        append_columns: bool = False,
    ) -> None:
        """Create a table from schema arguments and upsert provided rows."""
        self.create_table_from_schema(
            table_name=table_name,
            columns=columns,
            primary_key=primary_key,
            indexes=indexes,
        )
        self.upsert_dict_rows(table_name=table_name, rows=rows, append_columns=append_columns)

    def get_table(
        self,
        table: str,
        file_name: str | None = None,
        by_file: bool = False,
    ) -> dict[str, list[str]] | list[str] | dict[str, str | list[str]] | dict[str, dict[str, str | list[str]]]:
        """Fetch rows from ``table``.

        Preferred layout is a single value column named the same as the table
        (e.g. ``verbs(verbs, file_name)``).  For dict-backed tables inserted via
        :meth:`insert_data`, a ``key``/``value`` layout is also supported.

        Args:
            table: Table name. For file-centric tables the column named the same
                as the table is used as the value column (e.g. ``verbs``), or a
                ``key``/``value`` two-column layout is also accepted.
            file_name: If given, return values for that specific file only.
                ``by_file`` is ignored when ``file_name`` is provided.
            by_file: Controls grouping when ``file_name`` is omitted.
                ``False`` (default) → ``{value: [file, ...]}``
                ``True``            → ``{file: [value, ...]}``
                For key/value tables and ``by_file=True``: ``{file: {key: value}}``.
        """
        columns = {
            row[1]
            for row in self.cur.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not columns:
            raise ValueError(f"Table '{table}' does not exist.")

        is_key_value_table = {"key", "value"}.issubset(columns) and table not in columns

        if table in columns:
            value_expr = table
        elif is_key_value_table:
            value_expr = "value"
        elif "value" in columns:
            value_expr = "value"
        elif "key" in columns:
            value_expr = "key"
        else:
            raise ValueError(
                f"Table '{table}' has no supported value column; expected '{table}' or key/value."
            )

        if is_key_value_table:
            if file_name is not None:
                rows = self.cur.execute(
                    f"SELECT key, value FROM {table} WHERE file_name = ?", (file_name,)
                ).fetchall()
                result: dict[str, str | list[str]] = {}
                for key, value in rows:
                    if key not in result:
                        result[key] = value
                    elif isinstance(result[key], list):
                        result[key].append(value)
                    else:
                        result[key] = [result[key], value]
                return result

            rows = self.cur.execute(f"SELECT file_name, key, value FROM {table}").fetchall()
            if by_file:
                result: dict[str, dict[str, str | list[str]]] = {}
                for file, key, value in rows:
                    file_dict = result.setdefault(file, {})
                    if key not in file_dict:
                        file_dict[key] = value
                    elif isinstance(file_dict[key], list):
                        file_dict[key].append(value)
                    else:
                        file_dict[key] = [file_dict[key], value]
                return result

            result: dict[str, dict[str, str | list[str]]] = {}
            for file, key, value in rows:
                key_dict = result.setdefault(key, {})
                if file not in key_dict:
                    key_dict[file] = value
                elif isinstance(key_dict[file], list):
                    key_dict[file].append(value)
                else:
                    key_dict[file] = [key_dict[file], value]
            return result

        if file_name is not None:
            rows = self.cur.execute(
                f"SELECT {value_expr} FROM {table} WHERE file_name = ?", (file_name,)
            ).fetchall()
            return [row[0] for row in rows]

        # Single query; avoid N+1 by loading all rows at once.
        rows = self.cur.execute(f"SELECT {value_expr}, file_name FROM {table}").fetchall()
        result: dict[str, list[str]] = {}
        if by_file:
            for value, file in rows:
                result.setdefault(file, []).append(value)
        else:
            for value, file in rows:
                result.setdefault(value, []).append(file)
        return result

    def get_rows(
        self,
        table_name: str,
        filters: dict[str, Any] | None = None,
        like_filters: dict[str, str] | None = None,
    ) -> list[DBRow]:
        """Return rows from an arbitrary-schema table as a list of dicts.

        Intended for tables created via `create_table_from_schema` that do
        not follow the file-centric layout expected by `get_table`.

        Args:
            table_name: Table to query.
            filters: Optional equality filters, e.g. ``{"scene_name": "scene-0061"}``.
            like_filters: Optional SQL ``LIKE`` pattern filters, e.g.
                ``{"filename": "samples/%"}`` to match only rows whose filename
                starts with ``"samples/"``.  Use ``%`` as the wildcard character.
        """
        table_sql = self._validate_identifier(table_name)
        columns = [
            row[1]
            for row in self.cur.execute(f"PRAGMA table_info({table_name})").fetchall()
        ]
        if not columns:
            raise ValueError(f"Table '{table_name}' does not exist in database {self.db_path}.")

        query = f"SELECT * FROM {table_sql}"
        params: list[Any] = []
        clauses: list[str] = []
        if filters:
            clauses += [f"{self._validate_identifier(k)} = ?" for k in filters]
            params += list(filters.values())
        if like_filters:
            clauses += [f"{self._validate_identifier(k)} LIKE ?" for k in like_filters]
            params += list(like_filters.values())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        rows = self.cur.execute(query, params).fetchall()
        return [dict(zip(columns, row)) for row in rows]


    def get_tables(self) -> list[str]:
        """Return a list of all table names in the database."""
        rows = self.cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [row[0] for row in rows]
        
    def get_file_names(self) -> list[str]:
        """Return a list of all file names in the database."""
        rows = self.cur.execute("SELECT file_name FROM files").fetchall()
        return [row[0] for row in rows]

    def print(self, file_name: str) -> None:
        """Print all data stored for a given file across every linked table."""
        tables = [
            row[0] for row in self.cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'files'"
            ).fetchall()
        ]
        print(f"Details for {file_name}:") 
        for table in tables:
            rows = self.cur.execute(
                f"SELECT * FROM {table} WHERE file_name = ?", (file_name,)
            ).fetchall()
            columns = [desc[0] for desc in self.cur.description if desc[0] != "file_name"]
            print(f"  {table}:")
            for row in rows:
                values = row[1:]  # strip file_name
                print(f"    - {', '.join(f'{col}={val}' for col, val in zip(columns, values))}")

    def _create_table(self, table: str, *columns: str, index_on: tuple[str, ...] = ("file_name",)) -> None:
        """Create a file-linked table, silently skipping if it already exists.

        The table always has a ``file_name`` FK to ``files`` as its first column,
        followed by the given ``columns`` (all TEXT).  A UNIQUE constraint is
        enforced across all columns (including ``file_name``).

        Args:
            table: Table name to create.
            *columns: Names of the value columns.
            index_on: Columns to index on (default: ``("file_name",)``).
        """
        col_defs = ", ".join(f"{c} TEXT" for c in columns)
        unique = ", ".join(("file_name", *columns))
        self.cur.execute(f"""CREATE TABLE IF NOT EXISTS {table}(
            file_name TEXT, {col_defs},
            FOREIGN KEY(file_name) REFERENCES files(file_name),
            UNIQUE({unique}))""")
        self.cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table} ON {table}({', '.join(index_on)})")


    def delete_table(self, table_name: str) -> None:
        """Delete a table from the database."""
        table_sql = self._validate_identifier(table_name)
        self.cur.execute(f"DROP TABLE IF EXISTS {table_sql}")
        self.con.commit()



# ------------------------------ MAIN ------------------------------

if __name__ == "__main__":
    db_manager = DatabaseManager()
    pass