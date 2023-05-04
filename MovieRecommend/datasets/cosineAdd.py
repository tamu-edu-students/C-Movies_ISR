import pymysql
import pandas as pd

# create a connection to your MySQL database
connection = pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='',
    db='cmovies'
)
moviesf= pd.read_csv('/Users/SahithiRavipati/Documents/TAMU Courses/ISR/C-Movies_ISR/MovieRecommend/datasets/books.csv',sep='\t')
moviesf.columns =['movieA','movieB','score']

# create a cursor object
cursor = connection.cursor()

# # create a DataFrame with your data
# data = {
#     'col1': [1, 2, 3],
#     'col2': ['a', 'b', 'c'],
#     'col3': [True, False, True]
# }
# df = pd.DataFrame(data)

# create your MySQL table (optional)
create_table_query = '''CREATE TABLE IF NOT EXISTS cosinescores1 (
                        movieA INT,
                        movieB INT,
                        score DOUBLE
                    );'''
cursor.execute(create_table_query)

# insert the data into the MySQL table
for index, row in moviesf.iterrows():
    insert_query = f'''INSERT INTO cosinescores1 (movieA, movieB, score)
                       VALUES ({row['movieA']}, '{row['movieB']}', {row['score']});'''
    cursor.execute(insert_query)

# commit the changes to the database
connection.commit()

# close the cursor and connection objects
cursor.close()
connection.close()