import sqlite3
import pandas as pd


# Db builder function instantiate DB with X variables + Y target
def init_db(df: pd.DataFrame) -> None:
    """
    Initializes a SQLite database with data from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing data to be inserted into the database.

    Returns:
        None
    """
    # Create SQLite database connection
    conn = sqlite3.connect('SectionB/data/db/music.db')
    # Convert DataFrame to SQLite table
    df.to_sql('music', conn, if_exists='replace', index=False)


# Get all records in music db
def get_all() -> list:
    """
    Retrieves all records from the 'music' table in the SQLite database.

    Returns:
        list: A list of tuples representing all records in the 'music' table.
    """

    # Create SQLite database connection
    conn = sqlite3.connect('SectionB/data/db/music.db')
    # # Create a cursor object
    c = conn.cursor()
    c.execute("SELECT * FROM music")
    return c.fetchall()


# Add for single entries only for this DB
def insert_track(combined_df: pd.DataFrame) -> None:
    """
    Inserts track data into the 'music' table in the SQLite database.

    Args:
        combined_df (pd.DataFrame): DataFrame containing track data to be inserted.

    Returns:
        None
    """
    # Create SQLite database connection
    conn = sqlite3.connect('SectionB/data/db/music.db')

    with conn:
        combined_df.to_sql('music', conn, if_exists='append', index=False)


def lookup_genre(genre: str) -> list:
    """
    Looks up tracks in the 'music' table with the specified genre.

    Args:
        genre (str): Genre to be searched for.

    Returns:
        list: A list of tuples representing tracks with the specified genre.
    """
    # Create SQLite database connection
    conn = sqlite3.connect('SectionB/data/db/music.db')
    c = conn.cursor()

    c.execute("SELECT title FROM music WHERE genre=:genre;",
              {'genre': genre})
    return c.fetchall()
