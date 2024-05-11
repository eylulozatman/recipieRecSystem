from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from recipeRecDTO import RecipeRecommendationObject

app = Flask(__name__)

# Load recipe data
recipe_data = pd.read_csv('https://blobrecipeimages.blob.core.windows.net/data-set-kaggle/recipes_with_no_tags_and_cuisine.csv')
recipe_data = recipe_data.dropna(subset=['ingredients', 'cuisine_path'])
recipe_data['total_time'] = recipe_data['total_time'].astype(str)
recipe_data['combined_features'] = recipe_data['total_time'] + ' ' + recipe_data['ingredients'] + ' ' + recipe_data['cuisine_path']

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(recipe_data['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def get_recommendations(title):
   
    # Check if the query is a substring of any recipe names
    matching_recipes = recipe_data[recipe_data['recipe_name'].str.contains(title, case=False)]

    if matching_recipes.empty:
        return print(f"No recipes found for '{title}'. Try a different search term.")

    # Use the first matching recipe as the selected recipe
    selected_recipe_row = matching_recipes.iloc[0]

    # Get the index of the selected recipe
    idx = selected_recipe_row.name

    # Get the pairwise similarity scores based on ingredients only
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the recipes based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 similar recipes
    sim_recipes = sim_scores[1:4]  # Adjusted to get top 20 similar recipes

    # Get the recipe indices
    recipe_indices = [i[0] for i in sim_recipes]

    # Create a list to store recommended recipes
    recommended_recipes = []

    # Iterate over the indices and add recipe information to the list
    for idx in recipe_indices:
        recommended_recipe = recipe_data.iloc[idx]
        recommended_recipes.append(create_recommendation_object(recommended_recipe))

    # Return the list of recommended recipes
    return recommended_recipes


@app.route('/hello', methods=['GET'])
def hello():
    return "hellooooo"


@app.route('/get-recommendations', methods=['GET'])
def recommend():
    title = request.args.get('title')
    recommendations = get_recommendations(title)

    # Önerileri JSON formatında geri döndür
    return jsonify({'recommendations': [vars(recommendation) for recommendation in recommendations]})



def create_recommendation_object(recipe_row):
    return RecipeRecommendationObject(
        title=recipe_row['recipe_name'],
        ingredients=recipe_row['ingredients'],
        description=recipe_row['directions'],  
        cuisine=recipe_row['cuisine_path'],  
        timing=format_time(recipe_row['total_time']),  
        photoPathURL=recipe_row['img_src']
    )


@app.route('/add-recipe-to-dataset', methods=['POST'])
def add_recipe_to_dataset():
    data = request.json

    # Extract data from request body
    recipe_name = data.get('title', '')
    prep_time = format_time(data.get('preparationTime', None))
    cook_time = None  
    total_time = None  
    servings = None  
    yield_value = None  
    ingredients = data.get('ingredients', '')
    directions = data.get('description', '')
    rating = None  
    url = None  
    cuisine_path = data.get('cuisine', '')
    nutrition = None  
    timing = None
    img_src = data.get('photoPathURL', '')

    # Process the data and add it to the dataset
    # You can perform any necessary processing here, such as data validation and formatting
    # Then add the recipe to the dataset
    new_recipe = {
        'recipe_name': recipe_name,
        'prep_time': prep_time,
        'cook_time': cook_time,
        'total_time': total_time,
        'servings': servings,
        'yield': yield_value,
        'ingredients': ingredients,
        'directions': directions,
        'rating': rating,
        'url': url,
        'cuisine_path': cuisine_path,
        'nutrition': nutrition,
        'timing': timing,
        'img_src': img_src
    }


    recipe_data.append(new_recipe, ignore_index=True)

    # Dataseti tekrar kaydet
    recipe_data.to_csv('recipes_with_no_tags_and_cuisine_updated.csv', index=False)


    return jsonify({"message": "Recipe added to dataset successfully"})


def format_time(minutes):
    if not minutes:
        return ''

    hours = minutes // 60
    mins = minutes % 60

    if hours > 0 and mins > 0:
        return f"{hours} hrs {mins} mins"
    elif hours > 0:
        return f"{hours} hrs"
    else:
        return f"{mins} mins"


if __name__ == '__main__':
    app.run()