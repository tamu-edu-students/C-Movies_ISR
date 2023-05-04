import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user='root',
  password='',
  database='cmovies'
)

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE IF NOT EXISTS userpreference ( 	userID int(11) NOT NULL,movieID int(20) NOT NULL,CREATED_AT timestamp DEFAULT CURRENT_TIMESTAMP);")