from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
import re
from recipeRecDTO import RecipeRecommendationObject

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
CORS(app)

sas_token = "sp=racwdyti&st=2024-05-14T12:06:34Z&se=2024-06-04T20:06:34Z&sv=2022-11-02&sr=b&sig=%2FYZk57JEkcpWlOKhm9rl5y2roVMQbwl%2Fa%2FgyPS5uL5A%3D"
blob_service_client = BlobServiceClient(account_url="https://blobrecipeimages.blob.core.windows.net", credential=sas_token)
container_client = blob_service_client.get_container_client("data-set-kaggle")

# Load recipe data
blob_url = "https://blobrecipeimages.blob.core.windows.net/data-set-kaggle/recipes_with_no_tags_and_cuisine.csv?" + sas_token
recipe_data = pd.read_csv(blob_url)
recipe_data = recipe_data.dropna(subset=['ingredients', 'cuisine_path'])

# Preprocess recipe data
lemmatizer = WordNetLemmatizer()

def preprocess_data(data):
    data['total_time'] = data['total_time'].astype(str)
    data['combined_features'] = data['total_time'] + ' ' + data['ingredients'] + ' ' + data['cuisine_path']
    data['normalized_ingredients'] = data['ingredients'].apply(lambda x: ', '.join([normalize_ingredient(i) for i in x.split(',')]))
    data['combined_features'] = data['normalized_ingredients'] + ' ' + data['recipe_name'] + ' ' + data['directions'] + ' ' + data['cuisine_path']

def normalize_ingredient(ingredient):
    ingredient = ingredient.lower()
    ingredient = re.sub(r'\s*\(.*?\)\s*', '', ingredient)
    ingredient = re.sub(r'\bfinely\b|\bchopped\b|\bsliced\b|\bdiced\b|\bground\b|\bfresh\b|\bdry\b|\bcrushed\b', '', ingredient)
    ingredient = re.sub(r'\bslices\b|\bpieces\b|\bleaves\b|\bparts\b|\bstalks\b|\bheads\b', '', ingredient)
    ingredient = ' '.join([lemmatizer.lemmatize(word) for word in ingredient.split()])
    ingredient = re.sub(r'\s+', ' ', ingredient).strip()
    return ingredient

preprocess_data(recipe_data)

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(recipe_data['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Synonyms dictionary
synonyms = {
    "meat": ["beef", "chicken", "pork", "lamb"],
    "cake": ["pie", "dessert", "pastry"],
    "burger": ["cheeseburger", "hamburger", "sandwich"],
    "pasta": ["spaghetti"]
}

def get_synonyms(word):
    for key, value in synonyms.items():
        if word in value:
            return value
    return [word]

def get_recommendations(title=None, ingredients=None, mealTypes=None):
    matching_recipes = recipe_data.copy()
    
    if title:
        title_words = title.lower().split()
        expanded_title_words = [get_synonyms(word) for word in title_words]
        title_query = '|'.join('|'.join(synonym) for synonym in expanded_title_words)
        title_condition = matching_recipes['cuisine_path'].str.contains(title_query, case=False, na=False) | \
                          matching_recipes['recipe_name'].str.contains(title_query, case=False, na=False) | \
                          matching_recipes['directions'].str.contains(title_query, case=False, na=False) | \
                          matching_recipes['normalized_ingredients'].str.contains(title_query, case=False, na=False)
        matching_recipes = matching_recipes[title_condition]
    
    if ingredients:
        normalized_ingredients = '|'.join([normalize_ingredient(ingredient) for ingredient in ingredients.lower().split(',')])
        ingredients_condition = matching_recipes['normalized_ingredients'].str.contains(normalized_ingredients, case=False, na=False) | \
                                matching_recipes['directions'].str.contains(normalized_ingredients, case=False, na=False) | \
                                matching_recipes['cuisine_path'].str.contains(normalized_ingredients, case=False, na=False) | \
                                matching_recipes['recipe_name'].str.contains(normalized_ingredients, case=False, na=False)
        matching_recipes = matching_recipes[ingredients_condition]

    if mealTypes:
        mealType_queries = '|'.join([mealType.lower().rstrip('s') for mealType in mealTypes])
        mealType_condition = matching_recipes['cuisine_path'].str.contains(mealType_queries, case=False, na=False)
        matching_recipes = matching_recipes[mealType_condition]

    if matching_recipes.empty:
        return []

    combined_features = matching_recipes['normalized_ingredients'] + ' ' + matching_recipes['recipe_name'] + ' ' + matching_recipes['directions'] + ' ' + matching_recipes['cuisine_path']
    tfidf_matrix_with_ingredients = tfidf_vectorizer.fit_transform(combined_features)
    cosine_sim_with_ingredients = linear_kernel(tfidf_matrix_with_ingredients, tfidf_matrix_with_ingredients)

    sim_scores = []
    for i in range(len(matching_recipes)):
        sim_scores.append((i, cosine_sim_with_ingredients[i, -1]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_recipes = [matching_recipes.iloc[idx] for idx, score in sim_scores[:5]]

    return recommended_recipes

@app.route('/get-recommendations', methods=['GET'])
def recommend():
    title = request.args.get('title')
    ingredients = request.args.get('ingredients')
    mealType = request.args.get('mealNames')

    if not title and not ingredients:
        return jsonify({'error': 'Please provide at least one parameter.'}), 400

    if ingredients and ":" in ingredients:
        ingredients = format_data_ingredients(ingredients)

    if mealType and "," in mealType:
        mealType = mealType.split(',')


    recommendations = get_recommendations(title, ingredients, mealType)


    if not recommendations:
        return jsonify({'message': 'No similar recipes found.'}), 200

    recommendation_objects = [create_recommendation_object(recipe) for recipe in recommendations]

    return jsonify({'recommendations': [vars(obj) for obj in recommendation_objects]})
    

def format_data_ingredients(ingredients):
    formatted_ingredients = []
    for ingredient in ingredients.split(','):
        if ":" in ingredient:
            # ":" karakterinden böl
            parts = ingredient.split(':')
            # İlk kısmı al, boşlukları kaldır
            formatted_ingredient = parts[0].strip()

            formatted_ingredients.append(formatted_ingredient)
        else:
            formatted_ingredients.append(ingredient.strip())
    return ', '.join(formatted_ingredients)


def create_recommendation_object(recipe_row):
    return RecipeRecommendationObject (
        title=recipe_row['recipe_name'],
        ingredients=recipe_row['ingredients'],
        description=recipe_row['directions'],  
        cuisine=recipe_row['cuisine_path'],  
        timing=recipe_row['total_time'],
        photoPathURL=recipe_row['img_src']
    )

@app.route('/hello', methods=['GET'])
def hello():
    return "hellooooo"

@app.route('/add-recipe-to-dataset', methods=['POST'])
def add_recipe_to_dataset():

    global recipe_data  # Declare recipe_data as global

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
    new_recipe = pd.DataFrame.from_dict({
        'recipe_name': [recipe_name],
        'prep_time': [prep_time],
        'cook_time': [cook_time],
        'total_time': [total_time],
        'servings': [servings],
        'yield': [yield_value],
        'ingredients': [ingredients],
        'directions': [directions],
        'rating': [rating],
        'url': [url],
        'cuisine_path': [cuisine_path],
        'nutrition': [nutrition],
        'timing': [timing],
        'img_src': [img_src]
    })

    recipe_data = pd.concat([recipe_data, new_recipe], ignore_index=True)

    # Dosyayı bloba yükle
    data = recipe_data.to_csv(index=False)
    blob_client = container_client.get_blob_client(blob="recipes_with_no_tags_and_cuisine.csv")
    blob_client.upload_blob(data, overwrite=True)

    print("Veri başarıyla bloba yüklendi.")

    return jsonify({"message": "Recipe added to dataset successfully"})
 
def format_time(minutes):
    if not minutes:
        return ''

    hours, mins = divmod(minutes, 60)

    if hours > 0 and mins > 0:
        return f"{hours} hrs {mins} mins"
    elif hours > 0:
        return f"{hours} hrs"
    else:
        return f"{mins} mins"

if __name__ == '__main__':
    app.run()