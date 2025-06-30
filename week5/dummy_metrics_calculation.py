import datetime
import time
import random
import logging 
import uuid
import pytz
import psycopg
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	value1 integer,
	value2 varchar,
	value3 float
)
"""

def prep_db(host, port, dbname, user, password):
	"""
    Prepares the PostgreSQL database by creating it if it doesn't exist and creating the dummy_metrics table.

    Args:
        host (str): The PostgreSQL host.
        port (int): The PostgreSQL port.
        dbname (str): The PostgreSQL database name.
        user (str): The PostgreSQL user.
        password (str): The PostgreSQL password.
	"""
	try:
		with psycopg.connect(f"host={host} port={port} user={user} password={password}", autocommit=True) as conn:
			res = conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
			if len(res.fetchall()) == 0:
				conn.execute("create database %s;" % (dbname,))
		with psycopg.connect(f"host={host} port={port} dbname={dbname} user={user} password={password}") as conn:
			conn.execute(create_table_statement)
	except Exception as e:
		logging.error(f"Error prepping database: {e}")
		raise

def calculate_dummy_metrics_postgresql(curr):
	"""
	Calculates dummy metrics and inserts them into the dummy_metrics table.

	Args:
		curr (psycopg.cursor): The PostgreSQL cursor object.
	"""
	value1 = random.randint(0, 1000)
	value2 = str(uuid.uuid4())
	value3 = random.random()

	curr.execute(
		"insert into dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
	)

def test_connection(host, port, dbname, user, password):
	"""
	Tests the connection to the PostgreSQL database.

	Args:
		host (str): The PostgreSQL host.
		port (int): The PostgreSQL port.
		dbname (str): The PostgreSQL database name.
		user (str): The PostgreSQL user.
		password (str): The PostgreSQL password.

	Returns:
		bool: True if the connection is successful, False otherwise.
	"""
	try:
		with psycopg.connect(f"host={host} port={port} dbname={dbname} user={user} password={password}", autocommit=True) as conn:
			conn.execute("SELECT 1")
			return True
	except Exception as e:
		logging.error(f"Connection test failed: {e}")
		return False

def main(host, port, dbname, user, password, iterations, send_timeout, test_only, prep_only):
	"""
    Args:
        host (str): The PostgreSQL host.
        port (int): The PostgreSQL port.
        dbname (str): The PostgreSQL database name.
        user (str): The PostgreSQL user.
        password (str): The PostgreSQL password.
        iterations (int): The number of iterations to run.
        send_timeout (int): The timeout between sends in seconds.
        test_only (bool): Whether to only test the connection and exit.
		prep_only (bool): Whether to only prep the database and exit.
    """
    
	if prep_only:
		try:
			prep_db(host, port, dbname, user, password)
			logging.info("Database prepped. Exiting.")
		except Exception as e:
			logging.error(f"An error occurred: {e}")
		return
    
    
	if test_only:
		if test_connection(host, port, dbname, user, password):
			logging.info("Connection successful.")
		else:
			logging.error("Connection failed.")
			return
		return

	try:
		if not test_connection(host, port, dbname, user, password):
			logging.error("Failed to connect to PostgreSQL. Exiting.")
			return

		prep_db(host, port, dbname, user, password)
		last_send = datetime.datetime.now() - datetime.timedelta(seconds=send_timeout)
		with psycopg.connect(f"host={host} port={port} dbname={dbname} user={user} password={password}", autocommit=True) as conn:
			for i in range(0, iterations):
				with conn.cursor() as curr:
					calculate_dummy_metrics_postgresql(curr)

				new_send = datetime.datetime.now()
				seconds_elapsed = (new_send - last_send).total_seconds()
				if seconds_elapsed < send_timeout:
					time.sleep(send_timeout - seconds_elapsed)
				while last_send < new_send:
					last_send = last_send + datetime.timedelta(seconds=send_timeout)
				logging.info("data sent")
	except Exception as e:
		logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Generate and send dummy metrics to PostgreSQL.")
	parser.add_argument("--host", type=str, default="localhost", help="PostgreSQL host")
	parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
	parser.add_argument("--dbname", type=str, default="test", help="PostgreSQL database name")
	parser.add_argument("--user", type=str, default="postgres", help="PostgreSQL user")
	parser.add_argument("--password", type=str, default="example", help="PostgreSQL password")
	parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to run")
	parser.add_argument("--send_timeout", type=int, default=10, help="Timeout between sends in seconds")
	parser.add_argument("--test_only", action="store_true", help="Only test the connection and exit")
	parser.add_argument("--prep_only", action="store_true", help="Only prep the database and exit")

	args = parser.parse_args()


	main(args.host, args.port, args.dbname, args.user, args.password, args.iterations, args.send_timeout, args.test_only, args.prep_only)
