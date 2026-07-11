# ============================================================
# Intelligent Movie Recommendation System
# KNN + SVD + Genre-Based Hybrid Recommendation
# ============================================================

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


# ============================================================
# Movie Database
# ============================================================

movies = {
    "Inception": "Sci-Fi",
    "Interstellar": "Sci-Fi",
    "The Dark Knight": "Action",
    "The Matrix": "Sci-Fi",
    "Titanic": "Romance",
    "The Notebook": "Romance",
    "Avengers: Endgame": "Action",
    "Joker": "Drama",
    "Parasite": "Thriller",
    "Toy Story": "Animation",
    "The Shawshank Redemption": "Drama",
    "Gladiator": "Action",
    "La La Land": "Romance",
    "Finding Nemo": "Animation",
    "Shutter Island": "Thriller",
}


# ============================================================
# User Ratings Database
# ============================================================

user_ratings = {
    "Ali": {
        "Inception": 5,
        "Interstellar": 4,
        "The Dark Knight": 5,
        "Titanic": 2,
        "Toy Story": 3,
        "Gladiator": 5,
    },

    "Sara": {
        "Inception": 4,
        "Titanic": 5,
        "The Notebook": 5,
        "Joker": 4,
        "Parasite": 3,
        "La La Land": 5,
    },

    "Omar": {
        "The Matrix": 5,
        "Inception": 4,
        "Avengers: Endgame": 5,
        "The Dark Knight": 4,
        "Joker": 3,
        "Gladiator": 5,
    },

    "Lama": {
        "Titanic": 5,
        "The Notebook": 4,
        "Toy Story": 4,
        "Parasite": 4,
        "La La Land": 5,
        "Finding Nemo": 4,
    },

    "Huda": {
        "Interstellar": 5,
        "The Matrix": 4,
        "Avengers: Endgame": 4,
        "Toy Story": 3,
        "Finding Nemo": 4,
        "Inception": 5,
    },

    "Ahmed": {
        "Inception": 5,
        "Interstellar": 5,
        "The Matrix": 4,
        "The Dark Knight": 5,
        "Joker": 4,
        "Shutter Island": 4,
    },

    "Nora": {
        "Titanic": 4,
        "The Notebook": 5,
        "La La Land": 5,
        "Toy Story": 4,
        "Finding Nemo": 5,
        "Parasite": 3,
    },

    "Khalid": {
        "The Dark Knight": 5,
        "Avengers: Endgame": 4,
        "Gladiator": 5,
        "Joker": 4,
        "The Shawshank Redemption": 5,
        "Inception": 4,
    },
}


# ============================================================
# Utility Functions
# ============================================================

def clamp(value, minimum=1.0, maximum=5.0):
    """
    Keep predicted ratings between 1 and 5.
    """
    return max(minimum, min(float(value), maximum))


def user_average(ratings):
    """
    Calculate the average rating given by a user.
    """
    if not ratings:
        return 3.0

    return sum(ratings.values()) / len(ratings)


def movie_average(movie_name):
    """
    Calculate the average rating of a movie.
    """

    ratings = []

    for user_movies in user_ratings.values():
        if movie_name in user_movies:
            ratings.append(user_movies[movie_name])

    if not ratings:
        return 3.0

    return sum(ratings) / len(ratings)


def movie_rating_count(movie_name):
    """
    Count how many users rated a movie.
    """

    count = 0

    for user_movies in user_ratings.values():
        if movie_name in user_movies:
            count += 1

    return count


# ============================================================
# Create User-Movie Matrix
# ============================================================

def create_rating_matrix():
    """
    Convert the ratings dictionary into a numerical matrix.

    Rows represent users.
    Columns represent movies.
    Zero means that the movie was not rated.
    """

    users = list(user_ratings.keys())
    movie_names = list(movies.keys())

    matrix = np.zeros(
        (len(users), len(movie_names)),
        dtype=float,
    )

    for user_index, username in enumerate(users):
        for movie_index, movie_name in enumerate(movie_names):
            matrix[user_index, movie_index] = (
                user_ratings[username].get(movie_name, 0)
            )

    return users, movie_names, matrix


# ============================================================
# Genre Preference
# ============================================================

def calculate_genre_preferences(username):
    """
    Calculate the user's average rating for each genre.
    """

    preferences = {}
    counts = {}

    for movie_name, rating in user_ratings[username].items():
        genre = movies[movie_name]

        preferences[genre] = (
            preferences.get(genre, 0) + rating
        )

        counts[genre] = counts.get(genre, 0) + 1

    for genre in preferences:
        preferences[genre] /= counts[genre]

    return preferences


def genre_prediction(username, movie_name):
    """
    Predict a rating using the user's genre preferences.
    """

    genre = movies[movie_name]
    preferences = calculate_genre_preferences(username)

    if genre in preferences:
        return preferences[genre]

    return user_average(user_ratings[username])


# ============================================================
# KNN Recommendation Model
# ============================================================

def knn_predictions(username, number_of_neighbors=3):
    """
    Predict unseen movie ratings using User-Based KNN.

    The model finds users whose rating behavior is most
    similar to the target user.
    """

    users, movie_names, matrix = create_rating_matrix()

    target_index = users.index(username)
    target_vector = matrix[target_index]

    # Center each user's known ratings around their average.
    centered_matrix = np.zeros_like(matrix)

    for user_index in range(len(users)):
        rated_positions = matrix[user_index] > 0

        if np.any(rated_positions):
            average = np.mean(
                matrix[user_index][rated_positions]
            )

            centered_matrix[user_index][rated_positions] = (
                matrix[user_index][rated_positions] - average
            )

    number_of_neighbors = min(
        number_of_neighbors + 1,
        len(users),
    )

    model = NearestNeighbors(
        n_neighbors=number_of_neighbors,
        metric="cosine",
        algorithm="brute",
    )

    model.fit(centered_matrix)

    distances, indices = model.kneighbors(
        centered_matrix[target_index].reshape(1, -1)
    )

    target_average = user_average(
        user_ratings[username]
    )

    predictions = {}
    neighbor_details = {}

    for movie_index, movie_name in enumerate(movie_names):

        # Skip movies already watched by the target user.
        if target_vector[movie_index] > 0:
            continue

        weighted_rating = 0.0
        similarity_total = 0.0
        contributors = []

        for distance, neighbor_index in zip(
            distances[0],
            indices[0],
        ):
            # Skip the target user.
            if neighbor_index == target_index:
                continue

            neighbor_rating = matrix[
                neighbor_index,
                movie_index,
            ]

            # Skip if the neighbor did not rate this movie.
            if neighbor_rating == 0:
                continue

            similarity = 1 - distance

            if similarity <= 0:
                continue

            neighbor_username = users[neighbor_index]

            neighbor_average = user_average(
                user_ratings[neighbor_username]
            )

            normalized_rating = (
                neighbor_rating - neighbor_average
            )

            weighted_rating += (
                similarity * normalized_rating
            )

            similarity_total += similarity

            contributors.append(
                {
                    "user": neighbor_username,
                    "rating": neighbor_rating,
                    "similarity": similarity,
                }
            )

        if similarity_total > 0:
            prediction = (
                target_average
                + weighted_rating / similarity_total
            )
        else:
            # Fallback when no neighbor rated the movie.
            prediction = movie_average(movie_name)

        predictions[movie_name] = clamp(prediction)

        contributors.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        neighbor_details[movie_name] = contributors

    return predictions, neighbor_details


# ============================================================
# SVD Recommendation Model
# ============================================================

def svd_predictions(username):
    """
    Predict unseen movie ratings using SVD Matrix Factorization.

    Missing ratings are initially filled using each user's
    average before applying SVD.
    """

    users, movie_names, matrix = create_rating_matrix()

    target_index = users.index(username)

    filled_matrix = matrix.copy()
    user_means = []

    # Fill missing values with each user's average rating.
    for user_index in range(len(users)):
        rated_positions = matrix[user_index] > 0

        if np.any(rated_positions):
            mean_rating = np.mean(
                matrix[user_index][rated_positions]
            )
        else:
            mean_rating = 3.0

        user_means.append(mean_rating)

        missing_positions = matrix[user_index] == 0

        filled_matrix[
            user_index,
            missing_positions,
        ] = mean_rating

    user_means = np.array(user_means).reshape(-1, 1)

    # Mean-center the matrix.
    centered_matrix = filled_matrix - user_means

    maximum_components = min(
        centered_matrix.shape[0] - 1,
        centered_matrix.shape[1] - 1,
    )

    # The small database needs a small number of components.
    number_of_components = min(4, maximum_components)

    if number_of_components < 1:
        return {
            movie_name: movie_average(movie_name)
            for movie_name in movie_names
            if movie_name not in user_ratings[username]
        }

    svd = TruncatedSVD(
        n_components=number_of_components,
        random_state=42,
    )

    user_features = svd.fit_transform(
        centered_matrix
    )

    reconstructed_matrix = (
        user_features @ svd.components_
    )

    reconstructed_matrix += user_means

    predictions = {}

    for movie_index, movie_name in enumerate(movie_names):

        if movie_name in user_ratings[username]:
            continue

        predicted_rating = reconstructed_matrix[
            target_index,
            movie_index,
        ]

        predictions[movie_name] = clamp(
            predicted_rating
        )

    return predictions


# ============================================================
# Hybrid KNN + SVD Recommendation
# ============================================================

def get_recommendations(username, top_n=5):
    """
    Generate recommendations by combining:

    45% KNN prediction
    45% SVD prediction
    10% Genre preference
    """

    if username not in user_ratings:
        return []

    knn_scores, neighbor_details = knn_predictions(
        username=username,
        number_of_neighbors=3,
    )

    svd_scores = svd_predictions(username)

    recommendations = []

    for movie_name in movies:

        # Do not recommend watched movies.
        if movie_name in user_ratings[username]:
            continue

        knn_score = knn_scores.get(
            movie_name,
            movie_average(movie_name),
        )

        svd_score = svd_scores.get(
            movie_name,
            movie_average(movie_name),
        )

        genre_score = genre_prediction(
            username,
            movie_name,
        )

        popularity_score = movie_average(movie_name)
        rating_count = movie_rating_count(movie_name)

        # Main hybrid prediction.
        hybrid_score = (
            0.45 * knn_score
            + 0.45 * svd_score
            + 0.10 * genre_score
        )

        # Slight confidence adjustment for movies
        # that have multiple ratings.
        confidence = min(rating_count / 4, 1.0)

        final_score = (
            hybrid_score * 0.95
            + popularity_score * 0.05 * confidence
        )

        explanation = create_explanation(
            username=username,
            movie_name=movie_name,
            knn_score=knn_score,
            svd_score=svd_score,
            genre_score=genre_score,
            contributors=neighbor_details.get(
                movie_name,
                [],
            ),
        )

        recommendations.append(
            {
                "movie": movie_name,
                "genre": movies[movie_name],
                "final_score": clamp(final_score),
                "knn_score": knn_score,
                "svd_score": svd_score,
                "genre_score": genre_score,
                "explanation": explanation,
            }
        )

    recommendations.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return recommendations[:top_n]


# ============================================================
# Explain Recommendations
# ============================================================

def create_explanation(
    username,
    movie_name,
    knn_score,
    svd_score,
    genre_score,
    contributors,
):
    """
    Explain why the movie was recommended.
    """

    genre = movies[movie_name]

    if contributors:
        closest_user = contributors[0]

        return (
            f"{closest_user['user']} has similar preferences "
            f"and rated this movie "
            f"{closest_user['rating']}/5. "
            f"SVD also predicted {svd_score:.2f}/5, "
            f"and your {genre} preference is "
            f"{genre_score:.2f}/5."
        )

    return (
        f"SVD discovered a hidden preference pattern for "
        f"this movie and predicted {svd_score:.2f}/5. "
        f"Your estimated preference for {genre} movies is "
        f"{genre_score:.2f}/5."
    )


# ============================================================
# Evaluation
# ============================================================

def calculate_rmse():
    """
    Evaluate the SVD model using leave-one-out testing.

    One rating is temporarily hidden from each eligible user.
    The model then attempts to predict it.
    """

    errors = []
    hidden_ratings = []

    for username in list(user_ratings.keys()):

        if len(user_ratings[username]) < 3:
            continue

        movie_name = list(
            user_ratings[username].keys()
        )[-1]

        actual_rating = user_ratings[username].pop(
            movie_name
        )

        hidden_ratings.append(
            (username, movie_name, actual_rating)
        )

        try:
            predictions = svd_predictions(username)

            predicted_rating = predictions.get(
                movie_name,
                movie_average(movie_name),
            )

            errors.append(
                (actual_rating - predicted_rating) ** 2
            )

        finally:
            user_ratings[username][movie_name] = (
                actual_rating
            )

    if not errors:
        return None

    return sqrt(sum(errors) / len(errors))


# ============================================================
# User Interface Functions
# ============================================================

def list_users():
    print("\nAvailable users:")

    for username, ratings in user_ratings.items():
        print(
            f"- {username}: "
            f"{len(ratings)} rated movies"
        )


def print_movies():
    print("\nAvailable movies:")

    for movie_name, genre in movies.items():
        print(f"- {movie_name} ({genre})")


def show_user_profile():
    username = input(
        "Enter user name: "
    ).strip()

    if username not in user_ratings:
        print("User not found.")
        return

    print(f"\n{username}'s ratings:")

    for movie_name, rating in (
        user_ratings[username].items()
    ):
        print(
            f"- {movie_name}: {rating}/5 "
            f"({movies[movie_name]})"
        )

    preferences = calculate_genre_preferences(
        username
    )

    print("\nGenre preferences:")

    sorted_preferences = sorted(
        preferences.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for genre, score in sorted_preferences:
        print(f"- {genre}: {score:.2f}/5")


def add_new_user():
    username = input(
        "Enter new user name: "
    ).strip()

    if not username:
        print("Name cannot be empty.")
        return

    if username in user_ratings:
        print("User already exists.")
        return

    new_ratings = {}

    print_movies()

    print("\nRate movies from 1 to 5.")
    print("Enter 0 to skip.\n")

    for movie_name in movies:

        while True:
            try:
                rating = int(
                    input(f"{movie_name}: ")
                )

                if rating == 0:
                    break

                if 1 <= rating <= 5:
                    new_ratings[movie_name] = rating
                    break

                print(
                    "Enter a rating between 0 and 5."
                )

            except ValueError:
                print("Please enter a valid number.")

    if len(new_ratings) < 3:
        print(
            "Please rate at least three movies "
            "to generate meaningful recommendations."
        )
        return

    user_ratings[username] = new_ratings

    print(
        f"\nUser {username} was added successfully."
    )


def show_recommendations():
    username = input(
        "Enter user name: "
    ).strip()

    if username not in user_ratings:
        print("User not found.")
        return

    recommendations = get_recommendations(
        username=username,
        top_n=5,
    )

    if not recommendations:
        print("No unseen movies are available.")
        return

    print("\n========================================")
    print(f" Intelligent Recommendations for {username}")
    print("========================================")

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"\n{index}. {recommendation['movie']}"
        )

        print(
            f"   Genre: {recommendation['genre']}"
        )

        print(
            f"   Final predicted rating: "
            f"{recommendation['final_score']:.2f}/5"
        )

        print(
            f"   KNN prediction: "
            f"{recommendation['knn_score']:.2f}/5"
        )

        print(
            f"   SVD prediction: "
            f"{recommendation['svd_score']:.2f}/5"
        )

        print(
            f"   Genre preference: "
            f"{recommendation['genre_score']:.2f}/5"
        )

        print(
            f"   Why recommended: "
            f"{recommendation['explanation']}"
        )


def show_model_evaluation():
    rmse = calculate_rmse()

    if rmse is None:
        print(
            "Not enough ratings to evaluate the model."
        )
        return

    print("\nModel Evaluation:")
    print(f"RMSE: {rmse:.3f}")

    if rmse < 0.75:
        print("Model accuracy: Excellent")
    elif rmse < 1.0:
        print("Model accuracy: Good")
    elif rmse < 1.5:
        print("Model accuracy: Acceptable")
    else:
        print(
            "Model needs more users and ratings."
        )


# ============================================================
# Main Menu
# ============================================================

def main_menu():

    while True:
        print("\n========================================")
        print(" AI Movie Recommendation System")
        print(" KNN + SVD Hybrid Model")
        print("========================================")

        print("1) List users")
        print("2) Add new user")
        print("3) Get intelligent recommendations")
        print("4) Show movie list")
        print("5) Show user profile")
        print("6) Evaluate AI model")
        print("0) Exit")

        choice = input(
            "\nChoose an option: "
        ).strip()

        if choice == "1":
            list_users()

        elif choice == "2":
            add_new_user()

        elif choice == "3":
            show_recommendations()

        elif choice == "4":
            print_movies()

        elif choice == "5":
            show_user_profile()

        elif choice == "6":
            show_model_evaluation()

        elif choice == "0":
            print("Goodbye.")
            break

        else:
            print(
                "Invalid choice. Please try again."
            )


# ============================================================
# Start Program
# ============================================================

if __name__ == "__main__":
    main_menu()