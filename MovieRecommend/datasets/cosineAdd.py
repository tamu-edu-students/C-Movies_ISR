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
results = pd.DataFrame(columns=['movieA','movieB','score'])

# loop through each unique value in column "movieA
for a_value in moviesf["movieA"].unique():
    # loop through each unique value in column "B"
    for b_value in moviesf["movieB"].unique():
        # select the rows from dataframe A that match the current "movieA value
        a_rows = moviesf[moviesf["movieA"] == a_value]
        # select the rows from dataframe B that match the current "B" value
        b_rows = moviesf[moviesf["movieB"] == b_value]
        # merge the selected rows from dataframes A and B
        merged = pd.merge(a_rows, b_rows, on=["movieA", "movieB"])
        # sort the merged dataframe by Score in descending order
        sorted_merged = merged.sort_values(by=["score"], ascending=False)
        # select the top 50 rows from the sorted merged dataframe
        top_50 = sorted_merged.head(50)
        # append the top 50 rows to the results dataframe
        results = pd.concat([results, top_50])

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
for index, row in results.iterrows():
    insert_query = f'''INSERT INTO cosinescores1 (movieA, movieB, score)
                       VALUES ({row['movieA']}, '{row['movieB']}', {row['score']});'''
    cursor.execute(insert_query)

# commit the changes to the database
connection.commit()

# close the cursor and connection objects
cursor.close()
connection.close()