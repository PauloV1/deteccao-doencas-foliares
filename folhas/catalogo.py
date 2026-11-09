"""Mapeamentos entre as pastas/classes de cada base e a lista de classes do projeto.

As 38 classes do PlantVillage ocupam sempre os índices 0..37; as culturas novas (PlantWild) vêm depois.
"""

# PlantDoc (nome da pasta) -> classe do PlantVillage
PD_MAP = {
    "Apple leaf": "Apple___healthy", "Apple rust leaf": "Apple___Cedar_apple_rust", "Apple Scab Leaf": "Apple___Apple_scab",
    "Bell_pepper leaf": "Pepper,_bell___healthy", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot", "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy", "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_", "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot", "Peach leaf": "Peach___healthy", "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy", "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew", "Strawberry leaf": "Strawberry___healthy", "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf": "Tomato___healthy", "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus", "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold", "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}

# PlantWild (nome em classes.txt) -> classe do PlantVillage. Só entram classes com correspondência clara;
# "x leaf" = folha saudável.
PW_MAP = {
    "apple black rot": "Apple___Black_rot", "apple leaf": "Apple___healthy", "apple rust": "Apple___Cedar_apple_rust", "apple scab": "Apple___Apple_scab",
    "bell pepper leaf": "Pepper,_bell___healthy", "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot", "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy", "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)", "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf": "Corn_(maize)___healthy", "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "corn rust": "Corn_(maize)___Common_rust_",
    "grape black rot": "Grape___Black_rot", "grape leaf": "Grape___healthy", "peach leaf": "Peach___healthy",
    "potato early blight": "Potato___Early_blight", "potato late blight": "Potato___Late_blight", "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy", "soybean leaf": "Soybean___healthy", "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy", "strawberry leaf scorch": "Strawberry___Leaf_scorch", "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight", "tomato late blight": "Tomato___Late_blight", "tomato leaf": "Tomato___healthy",
    "tomato leaf mold": "Tomato___Leaf_Mold", "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot", "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

# Culturas do Brasil: classes do PlantWild sem equivalente no PlantVillage.
# Soja só tem a classe saudável; não há dados para cana, trigo e algodão.
PW_BR = {
    "rice leaf": "Rice___healthy", "rice blast": "Rice___Blast", "rice sheath blight": "Rice___Sheath_blight",
    "coffee leaf": "Coffee___healthy", "coffee leaf rust": "Coffee___Leaf_rust",
    "bean leaf": "Bean___healthy", "bean halo blight": "Bean___Halo_blight", "bean mosaic virus": "Bean___Mosaic_virus", "bean rust": "Bean___Rust",
    "banana leaf": "Banana___healthy", "banana panama disease": "Banana___Panama_disease",
    "citrus canker": "Citrus___Canker", "tobacco leaf": "Tobacco___healthy", "tobacco mosaic virus": "Tobacco___Mosaic_virus",
    "corn smut": "Corn_(maize)___Smut",
}
PW_HORTALICAS = {
    "lettuce leaf": "Lettuce___healthy", "lettuce downy mildew": "Lettuce___Downy_mildew", "lettuce mosaic virus": "Lettuce___Mosaic_virus",
    "garlic leaf": "Garlic___healthy", "garlic leaf blight": "Garlic___Leaf_blight", "garlic rust": "Garlic___Rust",
    "cucumber leaf": "Cucumber___healthy", "cucumber angular leaf spot": "Cucumber___Angular_leaf_spot", "cucumber bacterial wilt": "Cucumber___Bacterial_wilt",
    "cucumber powdery mildew": "Cucumber___Powdery_mildew", "eggplant leaf": "Eggplant___healthy", "eggplant cercospora leaf spot": "Eggplant___Cercospora_leaf_spot",
}

BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

# As 38 classes do PlantVillage (nomes das pastas), em ordem alfabética: índices 0..37 em todo o projeto.
CLASSES_PV = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def mapa_plantwild(usar_br: bool, hortalicas: bool) -> dict:
    """PlantWild -> classe do projeto, conforme a configuração."""
    m = dict(PW_MAP)
    if usar_br:
        m.update(PW_BR)
        if hortalicas:
            m.update(PW_HORTALICAS)
    return m


def classes_do_projeto(classes_pv: list, mapa_pw: dict) -> list:
    """38 classes do PlantVillage (ordem alfabética das pastas) + classes novas ordenadas."""
    novas = sorted({v for v in mapa_pw.values() if v not in classes_pv})
    return list(classes_pv) + novas
