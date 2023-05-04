import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import bs4 as bs
import urllib.request
import pickle
import ssl
import requests
import time
from flask_mysqldb import MySQL
import MySQLdb.cursors
from ranking import bm25_search
import base64

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ssl._create_default_https_context = ssl._create_unverified_context
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

stop_words = set(stopwords.words('english'))


def create_similarity():
    data = pd.read_csv(
        '/Users/anudeepika/Documents/ISR_project/C-Movies_ISR/MovieRecommend/datasets/main_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity


def getTopRankedMovies(userQuery):
    list = bm25_search(userQuery)
    return list


def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return ('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l


def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list


def get_suggestions():
    data = pd.read_csv(
        '/Users/anudeepika/Documents/ISR_project/C-Movies_ISR/MovieRecommend/datasets/main_data.csv')
    return list(data['movie_title'].str.capitalize())


app = Flask(__name__, static_url_path='/static')
# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'secret'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'ABCabc123$#@'
app.config['MYSQL_DB'] = 'login_database'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Intialize MySQL
mysql = MySQL(app)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


@app.route("/upload", methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        # print(request)
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:
            my_string = base64.b64encode(file.read())
            # print(my_string)
            hugging_face_url = "https://snehithb-get-image-context.hf.space/api/queue/push/"
            hugging_face_status_url = "https://snehithb-get-image-context.hf.space/api/queue/status/"

        #     Needs to be removed, added to test for 1 case
        #     if index > 1:
        #         continue
            my_string = "data:image/jpeg;base64,"+my_string.decode("utf-8")
            payload = json.dumps({
                "fn_index": 0,
                "data": [
                    my_string
                ],
                "action": "predict",
                "session_hash": "7p1oh5oyfh8"
            })
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0',
                'Accept': '/',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://snehithb-get-image-context.hf.space/?__theme=light',
                'Content-Type': 'application/json',
                'Origin': 'https://snehithb-get-image-context.hf.space',
                'Connection': 'keep-alive',
                'Cookie': 'session-space-cookie=acb1440f515e0f84f41ddb669ad094d1',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'TE': 'trailers'
            }
            try:
                response = requests.request(
                    "POST", hugging_face_url, headers=headers, data=payload)
            #     print(response.text["hash"])
            #     payload = json.dumps({
            #       response.text
            #     })
                temp = {}
                while True:
                    # average duration for hugging face to get the context of poster
                    time.sleep(4)
                    response1 = requests.request(
                        "POST", hugging_face_status_url, headers=headers, data=response)

        #             print(response1.text)
                    temp = json.loads(response1.text)
                #     print(temp)
                    if temp['status'] == 'COMPLETE':
                        break
                print(temp['data']['data'][0])
                cont = temp['data']['data'][0]

                word_tokens = word_tokenize(cont)
                # converts the words in word_tokens to lower case and then checks whether
                # they are present in stop_words or not
                filtered_sentence = [
                    w for w in word_tokens if not w.lower() in stop_words]
                # with no lower case conversion
                filtered_sentence = []

                for w in word_tokens:
                    if w not in stop_words:
                        filtered_sentence.append(w)

                # print(word_tokens)
                print(filtered_sentence)
                sentence = " ".join(filtered_sentence)
                bmResult = getTopRankedMovies(sentence)
                bmResult = bmResult.head(10)[['title', 'id']]
                movielist = []
                for index, row in bmResult.iterrows():
                    image_path = "posters/" + str(row['id'])+'.jpg'
                    tempdict = {}
                    tempdict['title'] = row['title']
                    tempdict['poster'] = image_path
                    movielist.append(tempdict)
                return render_template('bmranktable.html', items=movielist)
            except Exception as e:
                print(e)

# @app.route("/")


@app.route("/home")
def home():
    suggestions = get_suggestions()
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', suggestions=suggestions, username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


@app.route("/", methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM users WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            # return 'Logged in successfully!'
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg='')


@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    # Redirect to login page
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute(
                'INSERT INTO users VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users


@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', username=account['username'], email=account['email'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


@app.route("/bmrank", methods=["GET", "POST"])
def personlisedRanking():
    movie = request.form.get('searchName')
    bmResult = getTopRankedMovies(movie)
    bmResult = bmResult.head(10)[['title', 'id']]
    movielist = []
    for index, row in bmResult.iterrows():
        image_path = "posters/" + str(row['id'])+'.jpg'
        tempdict = {}
        tempdict['title'] = row['title']
        tempdict['poster'] = image_path
        movielist.append(tempdict)
    return render_template('bmrank.html', items=movielist)


@ app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    # print("rc", rc)
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str


@ app.route("/recommend", methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i]
                   for i in range(len(rec_posters))}

    casts = {cast_names[i]: [cast_ids[i], cast_chars[i],
                             cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i],
                                    cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen(
        'https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = []  # list of reviews
    reviews_status = []  # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i]
                     for i in range(len(reviews_list))}

    # passing all the data to the html file
    return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
                           vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
                           movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details)


if __name__ == '__main__':
    app.run(debug=True)
