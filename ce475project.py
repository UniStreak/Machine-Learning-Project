# EMRE ASLAN - 20160602075 - CE475 PROJECT

import pandas as pd
import matplotlib.pyplot as plt
import warnings

#define dataset
warnings.filterwarnings("ignore",category =Warning)
books = pd.read_csv('Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

#make a graph for ratings distribution
def RatingsDistribution():
    plt.rc("font", size=15)
    ratings.bookRating.value_counts(sort=False).plot(kind='bar')
    plt.title('Rating Distribution\n')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

#make a graph for age distribution
def AgeDistribution():
    users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
    plt.title('Age Distribution\n')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

#Recommendations based on rating counts:
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False)

#Recommendations based on correlations:
average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
average_rating.sort_values('ratingCount', ascending=False).head()

#To ensure statistical significance, users with less than 200 ratings, and books with less than 100 ratings are excluded:
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

#We convert the ratings table to a 2D matrix. The matrix will be sparse because not every user rated every book:
ratings_pivot = ratings.pivot(index='userID', columns='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns

#Find out which books are correlated with the given ISBN (Content-based filtering)
def contentBasedFiltering(isbn_no):
    bones_ratings = ratings_pivot[isbn_no]
    similar_to_bones = ratings_pivot.corrwith(bones_ratings)
    corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
    corr_bones.dropna(inplace=True)
    corr_summary = corr_bones.join(average_rating['ratingCount'])
    recommended_books = corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR', ascending=False)
    print("Recommendations for the book with the {0} ISBN number:\n".format(isbn_no))
    print(recommended_books.head(5))

#Find out which books are correlated with the given ISBN (Collaborative filtering)
def collaborativeFiltering(input):
    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

    book_ratingCount = (combine_book_rating.
         groupby(by = ['bookTitle'])['bookRating'].
         count().
         reset_index().
         rename(columns = {'bookRating': 'totalRatingCount'})
         [['bookTitle', 'totalRatingCount']]
        )

    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')

    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

    #In order to improve computing speed, and not run into the “MemoryError” issue, set limit user data for the US and Canada.
    combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

    us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
    us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)

    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)

    from scipy.sparse import csr_matrix
    us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

    from sklearn.neighbors import NearestNeighbors

    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(us_canada_user_rating_matrix)
    query_index = int(input)
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1),n_neighbors=6)

    for i in range(0, len(distances.flatten())):
        if i == 0:
           print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
        else:
           print('{0}: {1}'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]]))


#user interface
choice=int(input("What would you like to do?\n"
                 "1. Getting book recommendation with content-based filtering\n"
                 "2. Getting book recommendation with collaborative filtering\n"
                 "3. Dataset informations\n"
                 "4. Ratings distribution graph\n"
                 "5. Age distribution graph\n"
                 "Your choice: "))
if choice==1:
    contentBasedFiltering(input("For content-based recommendations, write the ISBN of book\n(e.g 0316666343): "))
elif choice==2:
    collaborativeFiltering(input("For collaborative recommendations, write the query index of book\n(e.g 123)(max 746): "))
elif choice==3:
    print("Records of ratings data (row&column): {0}\nColumns: {1}\n".format(ratings.shape, list(ratings.columns)))
    print("Records of books data (row&column): {0}\nColumns: {1}\n".format(books.shape, list(books.columns)))
    print("Records of users data (row&column): {0}\nColumns: {1}\n".format(users.shape, list(users.columns)))
    print("Shape of rating matrix (row&column): {0}\n".format(ratings_pivot.shape))
elif choice==4:
    RatingsDistribution()
elif choice==5:
    AgeDistribution()
else:
    print("Error. Try again.")

