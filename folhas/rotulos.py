"""Tradução dos rótulos para PT-BR: classe -> (cultura, condição, saudável)."""
CULTURAS = {"Apple": "Maçã", "Blueberry": "Mirtilo", "Cherry_(including_sour)": "Cereja", "Corn_(maize)": "Milho",
            "Grape": "Uva", "Orange": "Laranja", "Peach": "Pêssego", "Pepper,_bell": "Pimentão", "Potato": "Batata",
            "Raspberry": "Framboesa", "Soybean": "Soja", "Squash": "Abóbora", "Strawberry": "Morango", "Tomato": "Tomate",
            # culturas do Brasil (PlantWild)
            "Rice": "Arroz", "Coffee": "Café", "Bean": "Feijão", "Banana": "Banana", "Citrus": "Citros", "Tobacco": "Fumo",
            "Lettuce": "Alface", "Garlic": "Alho", "Cucumber": "Pepino", "Eggplant": "Berinjela"}
CONDICOES = {"healthy": "Saudável", "Apple_scab": "Sarna", "Black_rot": "Podridão negra", "Cedar_apple_rust": "Ferrugem do cedro",
             "Powdery_mildew": "Oídio", "Cercospora_leaf_spot Gray_leaf_spot": "Mancha de cercospora / mancha cinzenta",
             "Common_rust_": "Ferrugem comum", "Northern_Leaf_Blight": "Helmintosporiose (queima do norte)",
             "Esca_(Black_Measles)": "Esca (sarampo negro)", "Leaf_blight_(Isariopsis_Leaf_Spot)": "Mancha de isariopsis",
             "Haunglongbing_(Citrus_greening)": "Greening (HLB)", "Bacterial_spot": "Mancha bacteriana",
             "Early_blight": "Pinta preta (requeima precoce)", "Late_blight": "Requeima (mela)", "Leaf_scorch": "Queima das folhas",
             "Leaf_Mold": "Mofo foliar", "Septoria_leaf_spot": "Septoriose", "Spider_mites Two-spotted_spider_mite": "Ácaro-rajado",
             "Target_Spot": "Mancha-alvo", "Tomato_Yellow_Leaf_Curl_Virus": "Vírus do enrolamento amarelo (TYLCV)",
             "Tomato_mosaic_virus": "Vírus do mosaico (ToMV)",
             "Smut": "Carvão comum", "Blast": "Brusone", "Sheath_blight": "Mancha-da-bainha (queima da bainha)", "Leaf_rust": "Ferrugem do cafeeiro",
             "Halo_blight": "Crestamento bacteriano comum (halo)", "Mosaic_virus": "Mosaico (vírus)", "Rust": "Ferrugem", "Panama_disease": "Mal-do-Panamá (murcha de fusarium)",
             "Canker": "Cancro cítrico", "Downy_mildew": "Míldio", "Leaf_blight": "Crestamento foliar",
             "Angular_leaf_spot": "Mancha angular", "Bacterial_wilt": "Murcha bacteriana", "Cercospora_leaf_spot": "Cercosporiose"}


def traduzir(classe: str):
    cult, cond = classe.split("___")
    return CULTURAS[cult], CONDICOES[cond], cond == "healthy"


def n_culturas(classes):
    return len({traduzir(c)[0] for c in classes})
