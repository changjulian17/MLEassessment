import sqlite3
import pandas as pd

# Create SQLite database connection
conn = sqlite3.connect('../../data/db/music.db')
# # Create a cursor object
c = conn.cursor()


# Db builder function instantiate DB with X variables + Y target
def init_db(df: pd.DataFrame):
    # Convert DataFrame to SQLite table
    df.to_sql('music', conn, if_exists='replace', index=False)

    # Commit changes and close connection
    conn.commit()

# Get all records in music db
def get_all():
    c.execute("SELECT * FROM music")
    return c.fetchall()


# Add for single entries only for this DB
def insert_track(combined_df: pd.DataFrame) -> None:
    with conn:
        combined_df.to_sql('music', conn, if_exists='append', index=False)


# Util for getting pydantic schema
def generate_pydantic_schema():
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(music)")
    schema = cursor.fetchall()
    cursor.close()
    conn.close()

    pydantic_schema = generate_pydantic_schema('music', schema)

    with open('../../data/db/music_schema.txt', 'w') as file:
        file.write(pydantic_schema)
    file.close()
    print("Pydantic schema saved to output_schema.txt")

    return pydantic_schema


def lookup_genre(genre: str) -> list:
    c.execute("SELECT title FROM music WHERE genre=:genre;",
              {'genre': genre})
    return c.fetchall()

