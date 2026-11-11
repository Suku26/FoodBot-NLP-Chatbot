
from flask import Flask, request, jsonify, render_template_string
import sqlite3
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# =====================================
# 1. SQL DATABASE
# =====================================

DB_NAME = "foodbot.db"

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            ingredients TEXT,
            instructions TEXT,
            category TEXT,
            cooking_time INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    recipes = [
        ("Aloo Paratha",
         "potato,wheat flour,onion,chilli,salt",
         "Boil and mash potatoes. Mix with spices. Stuff into wheat dough and cook on a hot pan.",
         "Vegetarian", 25),

        ("Vegetable Pulao",
         "rice,carrot,peas,onion,spices",
         "Saute onions and vegetables. Add rice, water and spices. Cook until rice is soft.",
         "Vegetarian", 30),

        ("Paneer Curry",
         "paneer,tomato,onion,cream,spices",
         "Cook onion and tomato gravy. Add spices and paneer. Simmer and add cream.",
         "Vegetarian", 25),

        ("Masala Dosa",
         "rice,urad dal,potato,onion",
         "Prepare dosa batter. Make potato masala. Spread batter on a pan and serve with filling.",
         "Vegetarian", 35),

        ("Tomato Rice",
         "rice,tomato,onion,chilli,spices",
         "Cook tomatoes and onions with spices. Mix in cooked rice.",
         "Vegetarian", 20),

        ("Poha",
         "poha,onion,peanuts,mustard,lemon",
         "Rinse poha. Saute mustard, onion and peanuts. Add poha and lemon juice.",
         "Vegetarian", 15),

        ("Veg Sandwich",
         "bread,potato,tomato,cucumber,butter",
         "Spread butter on bread. Add sliced vegetables and potato. Grill until golden.",
         "Vegetarian", 10),

        ("Chana Masala",
         "chickpeas,onion,tomato,garlic,spices",
         "Cook onion, tomato and garlic. Add boiled chickpeas and spices. Simmer.",
         "Vegetarian", 30),

        ("Egg Bhurji",
         "egg,onion,tomato,chilli,salt",
         "Saute onion and tomato. Add beaten eggs and cook while stirring.",
         "Non-Vegetarian", 12),

        ("Chicken Curry",
         "chicken,onion,tomato,ginger,garlic",
         "Cook onion, ginger and garlic. Add chicken and tomato. Add spices and cook thoroughly.",
         "Non-Vegetarian", 40),

        ("Veg Fried Rice",
         "rice,carrot,cabbage,beans,soy sauce",
         "Stir fry vegetables. Add cooked rice and soy sauce. Mix well.",
         "Vegetarian", 20),

        ("Dal Tadka",
         "dal,onion,tomato,garlic,ghee",
         "Boil dal. Prepare a tempering with garlic, onion and tomato. Mix into dal.",
         "Vegetarian", 25),

        ("Potato Fry",
         "potato,onion,chilli,turmeric,salt",
         "Slice potatoes. Fry with onions, chilli and turmeric until cooked.",
         "Vegetarian", 20),

        ("Banana Smoothie",
         "banana,milk,honey,ice",
         "Blend banana, milk, honey and ice until smooth.",
         "Vegetarian", 5),

        ("Tomato Soup",
         "tomato,onion,garlic,butter",
         "Cook tomatoes with onion and garlic. Blend and simmer with seasoning.",
         "Vegetarian", 20),

        ("Besan Chilla",
         "besan,onion,tomato,chilli,salt",
         "Mix besan with water and chopped vegetables. Spread batter on a pan and cook both sides.",
         "Vegetarian", 15),

        ("Mango Lassi",
         "mango,yogurt,milk,sugar",
         "Blend mango, yogurt, milk and sugar until creamy.",
         "Vegetarian", 5),

        ("Upma",
         "semolina,onion,peas,mustard,curry leaves",
         "Roast semolina. Saute spices and vegetables, add water and semolina. Cook until thick.",
         "Vegetarian", 20)
    ]

    cursor.execute("SELECT COUNT(*) FROM recipes")

    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO recipes
            (name, ingredients, instructions, category, cooking_time)
            VALUES (?, ?, ?, ?, ?)
        """, recipes)

    conn.commit()
    conn.close()

initialize_database()

# =====================================
# 2. MACHINE LEARNING MODEL
# =====================================

training_texts = [
    "hello", "hi", "hey", "good morning",
    "good evening", "how are you",

    "what can I cook", "suggest a recipe",
    "recommend some food", "give me food ideas",
    "what should I make today",

    "I have potato and onion",
    "recipes using tomato",
    "what can I cook with rice",
    "suggest food with available ingredients",

    "show me paneer curry recipe",
    "how to make poha",
    "give me cooking instructions",
    "how do I prepare dosa",

    "show vegetarian food",
    "show non vegetarian recipes",
    "give me veg recipes",
    
    "help me", "what can you do",
    "show your features"
]

training_labels = [
    "greeting", "greeting", "greeting", "greeting",
    "greeting", "greeting",

    "recommend", "recommend", "recommend",
    "recommend", "recommend",

    "ingredients", "ingredients", "ingredients",
    "ingredients",

    "recipe", "recipe", "recipe", "recipe",

    "category", "category", "category",

    "help", "help", "help"
]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(training_texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, training_labels)

def predict_intent(message):
    transformed = vectorizer.transform([message])
    return model.predict(transformed)[0]

# =====================================
# 3. RECIPE RECOMMENDATION
# =====================================

def get_recipes():
    conn = get_db()
    recipes = conn.execute("SELECT * FROM recipes").fetchall()
    conn.close()
    return recipes

def find_ingredient_recipes(message):
    ingredients = [
        "potato", "onion", "tomato", "rice",
        "paneer", "chicken", "egg", "bread",
        "carrot", "peas", "milk", "banana",
        "chickpeas", "poha", "mango",
        "cucumber", "beans", "cabbage",
        "dal", "besan", "semolina"
    ]

    found = [
        item for item in ingredients
        if re.search(r"\b" + re.escape(item) + r"\b",
                     message.lower())
    ]

    if not found:
        return []

    results = []

    for recipe in get_recipes():
        recipe_ingredients = recipe["ingredients"].lower().split(",")

        matches = sum(
            1 for item in found if item in recipe_ingredients
        )

        if matches > 0:
            score = matches / len(found)
            results.append((recipe, score, matches))

    results.sort(
        key=lambda x: (x[1], x[2]),
        reverse=True
    )

    return [item[0] for item in results[:5]]

def find_recipe_by_name(message):
    message = message.lower()

    for recipe in get_recipes():
        if recipe["name"].lower() in message:
            return recipe

    return None

# =====================================
# 4. CHATBOT RESPONSE
# =====================================

def generate_response(message):

    intent = predict_intent(message)
    lower_message = message.lower()

    if intent == "greeting":
        return {
            "text": "Hello! Welcome to FoodBot. 🍲 Tell me what ingredients you have, and I'll suggest recipes!",
            "recipes": []
        }

    if intent == "help":
        return {
            "text": "I can recommend recipes, find dishes using your ingredients, show cooking instructions and suggest vegetarian or non-vegetarian food.",
            "recipes": []
        }

    if intent == "category":
        category = (
            "Vegetarian"
            if "veg" in lower_message and "non" not in lower_message
            else "Non-Vegetarian"
        )

        conn = get_db()
        results = conn.execute(
            "SELECT * FROM recipes WHERE category=? LIMIT 5",
            (category,)
        ).fetchall()
        conn.close()

        return {
            "text": "Here are some " + category + " recipes:",
            "recipes": [dict(r) for r in results]
        }

    if intent == "recipe":
        recipe = find_recipe_by_name(lower_message)

        if recipe:
            return {
                "text": "Here is how to prepare " + recipe["name"] + "!",
                "recipes": [dict(recipe)]
            }

        return {
            "text": "Please enter a recipe name from my recipe collection, such as Poha, Aloo Paratha or Paneer Curry.",
            "recipes": []
        }

    if intent == "ingredients":
        results = find_ingredient_recipes(message)

        if results:
            return {
                "text": "Here are recipes matching your ingredients:",
                "recipes": [dict(r) for r in results]
            }

        return {
            "text": "I couldn't identify ingredients. Try words like potato, rice, tomato, onion or paneer.",
            "recipes": []
        }

    if intent == "recommend":
        results = get_recipes()[:5]

        return {
            "text": "Here are some popular recipe ideas from my collection:",
            "recipes": [dict(r) for r in results]
        }

    return {
        "text": "Sorry, I didn't understand. Try asking for a recipe or entering your ingredients.",
        "recipes": []
    }

# =====================================
# 5. HTML + CSS FRONTEND
# =====================================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>FoodBot - Recipe Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #fff4e6, #ffe0c2);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 15px;
        }

        .container {
            width: 100%;
            max-width: 750px;
            height: 88vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: linear-gradient(135deg, #ff702e, #e84c19);
            color: white;
            padding: 22px;
            text-align: center;
        }

        .header h2 {
            margin: 0 0 7px;
        }

        .header p {
            margin: 0;
            font-size: 13px;
        }

        #chatbox {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #fffaf5;
        }

        .message {
            padding: 12px 15px;
            border-radius: 14px;
            margin-bottom: 14px;
            max-width: 85%;
            line-height: 1.5;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }

        .user {
            background: #ff7135;
            color: white;
            margin-left: auto;
        }

        .bot {
            background: #f0eee9;
            color: #333;
        }

        .recipe-card {
            background: white;
            border: 1px solid #ffd4bd;
            border-radius: 12px;
            padding: 13px;
            margin: 10px 0;
            color: #333;
            box-shadow: 0 3px 8px rgba(0,0,0,0.04);
        }

        .recipe-card h3 {
            color: #e85a20;
            margin-top: 0;
        }

        .recipe-card p {
            font-size: 13px;
            line-height: 1.6;
        }

        .input-area {
            display: flex;
            gap: 10px;
            padding: 15px;
            border-top: 1px solid #eee;
        }

        input {
            flex: 1;
            padding: 14px;
            border: 1px solid #ddd;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
            min-width: 0;
        }

        button {
            background: #f15b25;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
        }

        button:hover {
            background: #d94a18;
        }

        .suggestions {
            padding: 0 15px 12px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .suggestions button {
            background: #fff0e8;
            color: #d94a18;
            border: 1px solid #ffd4bd;
            font-size: 12px;
            padding: 8px 12px;
        }

        @media(max-width: 500px) {
            .container {
                height: 94vh;
            }

            .message {
                max-width: 95%;
            }

            button {
                padding: 12px;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h2>🍲 FoodBot</h2>
        <p>Your Intelligent Food Recipe Assistant</p>
    </div>

    <div id="chatbox">
        <div class="message bot">
            Hello! 👋 Welcome to FoodBot.

            Tell me which ingredients you have.
            I'll help you discover delicious recipes!
        </div>
    </div>

    <div class="suggestions">
        <button onclick="quickSend('I have potato and onion')">🥔 Potato recipes</button>
        <button onclick="quickSend('Show vegetarian food')">🥗 Veg recipes</button>
        <button onclick="quickSend('Suggest a recipe')">🍛 Suggest food</button>
    </div>

    <div class="input-area">
        <input
            id="userInput"
            type="text"
            placeholder="Enter ingredients or ask something..."
            onkeydown="if(event.key==='Enter') sendMessage()"
        >
        <button onclick="sendMessage()">Send</button>
    </div>

</div>

<script>

function addMessage(text, type) {
    const chatbox = document.getElementById("chatbox");

    const div = document.createElement("div");
    div.className = "message " + type;

    // Use textContent to safely display user and bot text
    div.textContent = text;

    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function addRecipes(recipes) {
    const chatbox = document.getElementById("chatbox");

    recipes.forEach(recipe => {
        const card = document.createElement("div");
        card.className = "recipe-card";

        const title = document.createElement("h3");
        title.textContent = "🍽️ " + recipe.name;

        const ingredients = document.createElement("p");
        ingredients.textContent =
            "Ingredients: " + recipe.ingredients;

        const time = document.createElement("p");
        time.textContent =
            "Cooking time: " + recipe.cooking_time + " minutes";

        const category = document.createElement("p");
        category.textContent = "Category: " + recipe.category;

        const instructions = document.createElement("p");
        instructions.textContent =
            "Preparation: " + recipe.instructions;

        card.append(title, ingredients, time, category, instructions);
        chatbox.appendChild(card);
    });

    chatbox.scrollTop = chatbox.scrollHeight;
}

async function sendMessage() {

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({message: message})
        });

        const data = await response.json();

        addMessage(data.text, "bot");
        addRecipes(data.recipes || []);

    } catch(error) {
        addMessage("Something went wrong. Please try again.", "bot");
    }
}

function quickSend(message) {
    document.getElementById("userInput").value = message;
    sendMessage();
}

</script>

</body>
</html>
"""

# =====================================
# 6. FLASK ROUTES
# =====================================

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "text": "Please enter a message.",
            "recipes": []
        })

    response = generate_response(message)

    # Save conversation in SQL database
    conn = get_db()

    conn.execute("""
        INSERT INTO chat_history (user_message, bot_response)
        VALUES (?, ?)
    """, (message, response["text"]))

    conn.commit()
    conn.close()

    return jsonify(response)

@app.route("/history")
def history():
    conn = get_db()

    rows = conn.execute("""
        SELECT * FROM chat_history ORDER BY id DESC LIMIT 50
    """).fetchall()

    conn.close()

    return jsonify([dict(row) for row in rows])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
