from rich.console import Console
from rich.panel import Panel
from argparse import ArgumentParser

from .db_setup import DB
from .csv_to_neon import ingest_csv_folder_to_neon
import os


def main():
	console = Console()
	parser = ArgumentParser(description="FloatChat data executor")
	parser.add_argument("command", choices=["init-neon", "ingest-neon"], help="What to run")
	parser.add_argument("--env-var", default="NEON_CONN", dest="env_var")
	parser.add_argument("--csv-folder", default="data/csv", dest="csv_folder")
	parser.add_argument("--chunk-rows", default=200000, type=int, dest="chunk_rows")
	args = parser.parse_args()

	if args.command == "init-neon":
		db = DB.from_env(env_var=args.env_var)
		db.init_db()
		console.print(Panel("[green]NeonDB schema initialized[/green]"))
	elif args.command == "ingest-neon":
		# Ensure we can read env, even if DB is constructed inside
		os_env_var = args.env_var  # reserved for future override
		ingest_csv_folder_to_neon(csv_folder=args.csv_folder, chunk_rows=args.chunk_rows)
		console.print(Panel(f"[green]Ingestion complete: {args.csv_folder} -> Neon[/green]"))


if __name__ == "__main__":
	main()
