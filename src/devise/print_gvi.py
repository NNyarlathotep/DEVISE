import sqlite3

conn = sqlite3.connect("gvi.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM gvi ORDER BY image_id ASC")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
