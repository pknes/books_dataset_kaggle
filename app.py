import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle


def location_transformation(record: str) -> str:
    country = record.split(',')[-1].strip()
    if country == '':
        return 'NO DATA'
    else:
        return country


def convert_release_year(record):
    try:
        record = int(record)
        if record == 0 or record > 2004:
            return np.NaN
        else:
            return record
    except ValueError:
        return np.NaN


def load_data(file: str) -> pd.DataFrame:
    return pd.read_csv('data/' + file + '.csv',
                       delimiter=';',
                       quotechar='"',
                       encoding='latin1',
                       header=0,
                       error_bad_lines=False,
                       warn_bad_lines=True,
                       )


if __name__ == '__main__':

    # Configurations:
    pages = ['Data analysis and initial preprocessing', 'Recommendation system', 'Regression task']

    # Loading files:
    users = load_data('users')
    books = load_data('books')
    ratings = load_data('ratings')

    # Project title:
    st.title("Books dataset task")

    # Page selection:
    page = st.sidebar.radio("Part to review: ", pages)

    if page == 'Data analysis and initial preprocessing':

        # Page title:
        st.title("Data analysis and initial preprocessing")

        # users dataset:
        st.header("'users' dataset")
        st.dataframe(users)

        # Users countries:
        st.subheader("'location' may be converted to a country only to be more informative.")

        users['Location'] = users['Location'].apply(lambda x: location_transformation(x))

        st.subheader('Top 15 countries of origin:')
        st.bar_chart(users['Location'].value_counts()[:15], height=500)

        st.subheader('Most users come from USA.')

        # Users age:
        st.subheader('Users age data:')
        fig, ax = plt.subplots()
        ax.hist(users['Age'], bins=100)
        st.pyplot(fig)

        st.subheader('Number of users without any information about their age:')

        st.write(users['Age'].isnull().value_counts())

        st.subheader('Most users are 50 years old or younger. However, some of them are over 100 years old or close '
                     'to 0, which is quite doubtful. Moreover, there is a lack of data concerning age for a lot of '
                     'users.')

        # books dataset
        st.header("'books' dataset")

        # Dropping unnecessary data
        books.drop([col for col in books.columns if col.startswith('Image')], axis=1, inplace=True)

        books.dropna(subset=['Book-Author', 'Publisher'], inplace=True)

        # Years transformation
        books['Year'] = books['Year-Of-Publication'].apply(lambda x: convert_release_year(x))
        books.drop('Year-Of-Publication', axis=1, inplace=True)

        st.subheader("Dataset after dropping unnecessary columns related to URLs as well as transforming the 'year' "\
                     "column so that it doesn't include any non-numeric strings or dates such as 0:")

        st.dataframe(books)

        # Release dates distribution:
        st.subheader('Books release dates:')
        fig, ax = plt.subplots()
        ax.hist(books['Year'], bins=40)
        st.pyplot(fig)

        st.subheader("Most books were written in the second half of the XX century.")

        # Top authors
        st.subheader('Top 10 authors:')

        st.bar_chart(books['Book-Author'].value_counts()[:10], height=500)

        # ratings dataset
        st.header("'ratings' dataset")

        st.dataframe(ratings)

        # Ratings distribution
        st.subheader('Ratings distribution:')

        fig, ax = plt.subplots()
        ax.hist(ratings['Book-Rating'], bins=10)
        st.pyplot(fig)

        st.subheader("Most grades are 0 which is rather not reliable - users probably acted in emotions or some "
                     "external factors could apply.")

        # Ratings transformation
        ratings['Book-Rating'] = ratings['Book-Rating'].apply(lambda x: np.NaN if x == 0 else x)
        ratings.dropna(how='any', inplace=True)

        st.subheader('Ratings distribution after dropping 0s:')
        fig, ax = plt.subplots()
        ax.hist(ratings['Book-Rating'], bins=9)
        st.pyplot(fig)

    elif page == 'Recommendation system':

        # Page title:
        st.title("Recommendation system")

        st.subheader("Recommendation system is supposed to propose 5 top book based on their average ratings. "
                     "Therefore, only books graded at lest twice are considered. Moreover, a user's reading history "
                     "is also taken into account to avoid suggesting already read books. Last but not least, books "
                     "must be present in the 'books' dataset so that there will be information about them.")

        # Data initial preprocessing:
        books.drop([col for col in books.columns if col.startswith('Image')], axis=1, inplace=True)
        books.dropna(subset=['Book-Author', 'Publisher'], inplace=True)
        books['Year'] = books['Year-Of-Publication'].apply(lambda x: convert_release_year(x))
        books.drop('Year-Of-Publication', axis=1, inplace=True)

        ratings['Book-Rating'] = ratings['Book-Rating'].apply(lambda x: np.NaN if x == 0 else x)
        ratings.dropna(how='any', inplace=True)

        # Filtering books to be graded at lest twice and present in 'books':
        ratings_recommendations = ratings[ratings['ISBN'].isin(books['ISBN'])]

        books_to_consider = (ratings_recommendations['ISBN'].value_counts() > 1).apply(
            lambda x: np.NaN if x is False else x).dropna(how='any').index

        ratings_recommendations = ratings_recommendations[ratings_recommendations['ISBN'].isin(books_to_consider)]

        st.header("Filtered 'ratings' dataset to meet criteria:")
        st.dataframe(ratings_recommendations)

        # Books to recommend:
        books_recommendation = books.merge(ratings_recommendations.drop('User-ID', axis=1).groupby('ISBN').mean(),
                                           on='ISBN').sort_values(['Book-Rating', 'Book-Author'], ascending=False)

        st.header("Books to recommend:")
        st.dataframe(books_recommendation)

        # Recommendation system:
        def recommendations(books_read: pd.Series) -> list:

            books_suggestion, item_no = list(), 0

            while len(books_suggestion) < 5:  # 5 top books

                item = books_recommendation.iloc[item_no]
                if item['ISBN'] not in books_read:
                    books_suggestion.append(item)

                item_no += 1

            return books_suggestion

        st.header("Draw a user to see their read books and recommendations for them.")

        button = st.button("Draw user ID")
        if button:
            user_id = np.random.choice(users['User-ID'], 1)[0]

            st.subheader("Draw user id is: {}".format(user_id))

            read_books = ratings_recommendations[ratings_recommendations['User-ID'] == user_id]['ISBN']

            if len(read_books) == 0:
                st.subheader("A chosen user did not read any book.")
            else:
                st.subheader("Chosen user's read books:")

                st.dataframe(books_recommendation[books_recommendation['ISBN'].isin(read_books)])

            st.subheader("Recommendations:")

            recommended_books = recommendations(read_books)

            st.dataframe(recommended_books)

    else:

        # Page title:
        st.title("Regression task")

        st.subheader("A regression model will try to predict grades given by users based on their information. "
                     "In this case, there are only age and origin country available. Therefore, it will be hard to "
                     "provide reliable predictions, since there are much more factors affecting rating. However, it "
                     "may be interesting to investigate how these two parameters influence the process.")

        # Data initial preprocessing:
        ratings['Book-Rating'] = ratings['Book-Rating'].apply(lambda x: np.NaN if x == 0 else x)
        ratings.dropna(how='any', inplace=True)

        users['Location'] = users['Location'].apply(lambda x: location_transformation(x))

        # users preprocessing:
        st.subheader("Only users with known age should be considered. Furthermore, only adults below 100 are left "
                     "as this seems to be more reliable data. As for locations, only top 7 will be used to not "
                     "include countries with very little users.")

        # Considered countries:
        top_countries = users['Location'].value_counts()[:7].index

        st.header("Considered top 7 countries:")

        st.write(top_countries)

        users = users[users['Location'].isin(top_countries)]

        # Filtering age:
        def filter_age(record):
            if 18 <= record < 100:
                return int(record)
            else:
                return np.NaN


        users['Age'] = users['Age'].apply(lambda x: filter_age(x))
        users = users.dropna(how='any')

        # Encoding countries:
        st.subheader("The One Hot Encoding is performed on countries so that they may be processed. Also the 'age' "
                     "column is standardized.")

        encoder_location = OneHotEncoder(drop='first', sparse=False)
        encoded_countries = encoder_location.fit_transform(np.array(users['Location']).reshape(-1, 1))

        users_numeric = users.drop('Location', axis=1).reset_index()\
            .join(pd.DataFrame(encoded_countries, columns=['location_' + str(i) for i in range(1, 7)]))\
            .set_index('User-ID').drop('index', axis=1)

        # Scaling users age:
        scaler_age = StandardScaler()

        users_numeric['Age'] = scaler_age.fit_transform(np.array(users_numeric['Age']).reshape(-1, 1))

        st.header("'users' dataset after transformations:")

        st.dataframe(users_numeric)

        # Merging datasets:
        ids = np.intersect1d(users_numeric.index, ratings['User-ID'])  # users present in both datasets
        users_numeric = users_numeric[users_numeric.index.isin(ids)]
        ratings = ratings[ratings['User-ID'].isin(ids)]

        data = ratings.drop('ISBN', axis=1).merge(users_numeric.reset_index(), on='User-ID')

        st.header("Merged transformed 'users' and 'ratings' (users common to both datasets are considered):")

        st.dataframe(data.drop('User-ID', axis=1))

        # Predicting ratings using the best model:
        inputs, target = data.drop(['User-ID', 'Book-Rating'], axis=1), data['Book-Rating']
        input_train, input_test, target_train, target_test = train_test_split(inputs, target, test_size=.15)

        st.header("Regression model")

        st.subheader("Dataset was divided into training and testing groups in a proportion 85% - 15%. The chosen "
                     "estimator was the Random Forest Regressor and its optimal parameters were found with the "
                     "Grid Search with Cross-Validation (with 5 folds).")

        st.subheader("The best estimator:")

        st.write(RandomForestRegressor(max_features=2, min_samples_split=6, n_estimators=200))

        with open('data/model.pkl', 'rb') as file:
            estimator = pickle.load(file)

        st.subheader("Model performance:")

        target_predicted = estimator.predict(input_test)

        st.write("Coefficient of determination: " + str(round(r2_score(target_test, target_predicted), 3)))

        results = pd.DataFrame(target_test)
        results.rename(columns={'Book-Rating': 'Actual rating'}, inplace=True)
        results['Predicted rating'] = target_predicted.round()

        st.dataframe(results)

        st.subheader("As expected, it may be seen that the model performed poorly as age and country only are not "
                     "sufficient to predict users ratings. Further tuning or choosing a different algorithm probably "
                     "would not significantly improve results. It may be also noticed that the estimator tends to give "
                     "high notes - 8 or 7 - more frequently, since these ratings occurred most often.")

        st.subheader("Having said this, for around 85% of all predictions, the difference was at most 2, thus "
                     "the model could rather be used to roughly estimate whether a rating would be positive or negative"
                     " instead of predicting exact value:")

        diff = abs(results['Predicted rating'] - results['Actual rating']).value_counts()

        st.write("For {}% of all predictions, the absolute difference in a rating value was at most 2"
                 .format(round(100 * ((diff[0] + diff[1] + diff[2]) / results.shape[0]), 2)))
