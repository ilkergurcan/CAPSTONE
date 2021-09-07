import pickle

filename = 'trained_model.sav'

loaded_model = pickle.load(open(filename, "rb"))
example = [[False,False,False,False,False]]


# Serverdan datanın Boolean olarak geleceğini düşünerek böyle yazdım. Farklı şekilde gelecekse değişirtiririm. 
def transform_data(cough, fever, sore_throat, shortness_of_breath, head_ache):
    cough = int(cough)
    fever = int(fever)
    sore_throat = int(sore_throat)
    shortness_of_breath = int(shortness_of_breath)
    head_ache = int(head_ache)
    
    data = [[cough, fever, sore_throat, shortness_of_breath, head_ache]]
    return data

def predict_corona(data):
    
    pred = loaded_model.predict(data)
    
    return pred[0]


data = transform_data(False, False, False, False, False)
pred = predict_corona(data)
print(pred)
