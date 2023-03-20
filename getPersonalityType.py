from random import sample
import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Preprocessing tools
ps = PorterStemmer()
wnl = WordNetLemmatizer()
str_punc = string.punctuation

engstopwords = stopwords.words("english")
engstopwordsV2 = re.sub('[' + re.escape(string.punctuation) + ']', '',
                        ' '.join(engstopwords)).split()

engstopwords = set(engstopwords).union(set(engstopwordsV2))

# Function to lemmatize a word using the three types: adjective, verb, noun
def lemmatize_all_types(word):
    word = wnl.lemmatize(word, 'a')
    word = wnl.lemmatize(word, 'v')
    word = wnl.lemmatize(word, 'n')
    return word

# Function to clean text
def clean(text):
    # Remove URLs from text
    text = re.sub("http.*?([ ]|\|\|\||$)", "", text).lower()
    url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    text = re.sub(url_regex, "", text)

    # Remove specific punctuation (usually associated with a word)
    text = re.sub(r'(:|;).', " ", text)
    
    # Remove punctuations
    text = re.sub('['+re.escape(str_punc)+']'," ",  text)
    
    # Remove parantheses, brackets
    text = re.sub('(\[|\()*\d+(\]|\))*', ' ', text)
    
    # Remove string marks
    text = re.sub('[’‘“\.”…–]', '', text)
    text = re.sub('[^(\w|\s)]', '', text)
    text = re.sub('(gt|lt)', '', text)
    
    #Check that each word is not stopword, and lemmatize it
    text = list(map(lemmatize_all_types, text.split()))
    text = [word for word in text if (word not in engstopwords)]
    text = " ".join(text)
    return text

# Convert personality type into a dominant cognitive function
def letters_to_functions(personality_type):
    translator = {
            'ENxP': 'Ne',
            'INxJ': 'Ni',
            'ESxP': 'Se',
            'ISxJ': 'Si',
            'ExTJ': 'Te',
            'IxTP': 'Ti',
            'IxFP': 'Fi',
            'ExFJ': 'Fe',
            
            'xNxx': 'N',
            'xSxx': 'S',
            'xxTx' : 'T',
            'xxFx': 'F',
            'Ixxx':'I',
            'Exxx':'E',
            }
    return translator[personality_type]

# Convert a Cognitive Functions Stack into Personality Type
def functions_to_letters(functions):
    translator = {
            'NiTe': 'INTJ',
            'NiFe': 'INFJ',
            'NeTi': 'ENTP',
            'NeFi': 'ENFP',
            'SiTe': 'ISTJ',
            'SiFe': 'ISFJ',
            'SeTi': 'ESTP',
            'SeFi': 'ESFP',
            'TeNi': 'ENTJ',
            'FeNi': 'ENFJ',
            'TiNe': 'INTP',
            'FiNe': 'INFP',
            'TeSi': 'ESTJ',
            'FeSi': 'ESFJ',
            'TiSe': 'ISTP',
            'FiSe': 'ISFP',
            }
    return translator[functions]

cf_info = {
        "Ni": {
                'name':'intuition',
                'role':'input',
                'direction':'internal'
                },
        "Ne": {
                'name':'intuition',
                'role':'input',
                'direction':'external'
                },
        "Si": {
                'name':'sensing',
                'role':'input',
                'direction':'internal'
                },
        "Se": {
                'name':'sensing',
                'role':'input',
                'direction':'external'
                },
        'Ti': {
                'name':'thinking',
                'role':'output',
                'direction':'internal'
                },
        'Te': {
                'name':'thinking',
                'role':'output',
                'direction':'external'
                },
        'Fi': {
                'name':'feeling',
                'role':'output',
                'direction':'internal',
                },
        'Fe': {
                'name':'feeling',
                'role':'output',
                'direction':'external'
                }
        }

# Dict of models
models = {
        # Cognitive Functions (cf)
        'NiNe':None,
        'NiSi':None,
        'NiSe':None,
        'NeSi':None,
        'NeSe':None,
        'SiSe':None,
        
        'TiTe':None,
        'TiFi':None,
        'TiFe':None,
        'TeFi':None,
        'TeFe':None,
        'FiFe':None,
            
        # G1 Method (detect dominant cognitive function)
        'NiTe':None,
        'NiFe':None,
        'SiTe':None,
        'SiFe':None,
        
        'NeFi':None,
        'NeTi':None,
        'SeFi':None,
        'SeTi':None,
        
        # G2 Method (detect correct direction)
        'NiTi':None,
        'NiFi':None,
        'NeTe':None,
        'NeFe':None,
        
        'SiTi':None,
        'SiFi':None,
        'SeTe':None,
        'SeFe':None,
        }

# Supporter models, increases accuracy ~3%
supporters = {
        'NvS':None,
        'FvT':None,
        'IvE':None,
        }

# Read models
path = "pickle/"
# Read models
# path = "../input/mbti-pretrained-models/"

for key in models:
    with open(path + key + '.pickle', 'rb') as file:
        models[key] = pickle.load(file)

for _key in supporters:
    with open(path + _key + '.pickle', 'rb') as file:
        supporters[_key] = pickle.load(file)

# Convert the direction of a cognitive function (i => e | e => i)
def flip_cf_direction(cognitiveFunction):
    direction = cognitiveFunction[1]
    new_direction = 'i' if direction == 'e' else 'e'
    return cognitiveFunction[0] + new_direction

# Pipeline for using a model to predict sample
def process_classify_sample(modelObject, sample):
    vectorizer = modelObject['cv']
    label_encoder = modelObject['labelEncoder']
    model = modelObject['model']
    
    # Preprocessing
    clean_sample = clean(sample)
    x = vectorizer.transform([clean_sample]).toarray()
    
    # Classification
    y = model.predict(x)
    y_probability = max(model.predict_proba(x)[0])
    classified_cf = label_encoder.inverse_transform(y)[0]
    return letters_to_functions(classified_cf), y_probability

    
# Phase 1 of classifying a personality type
# Classify dominant cognitive function
def phase1(sample):
    input_cf_acc = pd.Series({"Ni":0, "Ne":0, "Si":0, "Se":0}, dtype=float)
    output_cf_acc = pd.Series({"Ti":0, "Te":0, "Fi":0, "Fe":0}, dtype=float)
    
    input_cf = np.array(['Ni', 'Ne', 'Si', 'Se'])
    output_cf = np.array(['Ti', 'Te', 'Fi', 'Fe'])
    
    # nested loop for input, other nested loop for output
    # so models are: NiNe, NiSi, NiSe,  ... , TiTe, TiFi, TiFe, ...
    for i in range(3):
        for j in range(i+1, 4, 1):
            model_name = input_cf[i] + input_cf[j]
            modelObject = models[model_name]
            cognitive_fn, probability = process_classify_sample(modelObject, sample)
            
            # Incrase likelihood of the prediction class
            input_cf_acc[cognitive_fn] += probability
            other_cf = input_cf[i] if input_cf[j] == cognitive_fn else input_cf[j]
            
            # Increase likelihood (smaller value) of other classes
            input_cf_acc[other_cf] += 1 - probability
            
    # TiTe, TiFi, TiFe, ...
    for i in range(3):
        for j in range(i+1, 4, 1):
            model_name = output_cf[i] + output_cf[j]
            modelObject = models[model_name]
            cognitive_fn, probability = process_classify_sample(modelObject, sample)
            
            # Incrase likelihood of the prediction class
            output_cf_acc[cognitive_fn] += probability
            other_cf = output_cf[i] if output_cf[j] == cognitive_fn else output_cf[j]
            
            # Increase likelihood (smaller value) of other classes
            output_cf_acc[other_cf] += 1 - probability
    
    # Use supporter model (iNtuition vs Sensing)
    modelObject = supporters["NvS"]
    cognitive_fn, probability = process_classify_sample(modelObject, sample)
    
    # Increase iNtuitive functions
    if cognitive_fn == "N":
        input_cf_acc[['Ni', 'Ne']] += probability
        input_cf_acc[['Si', 'Se']] += 1 - probability
    # Increase Sensing functions
    else:
        input_cf_acc[['Si', 'Se']] += probability
        input_cf_acc[['Ni', 'Ne']] += 1 - probability
    
    # Use supporter model (Feeling vs Thinking)
    modelObject = supporters["FvT"]
    cognitive_fn, probability = process_classify_sample(modelObject, sample)
    
    # Increase Feeling functions likelihood
    if cognitive_fn == "F":
        output_cf_acc[['Fi', 'Fe']] += probability
        output_cf_acc[['Ti', 'Te']] += 1 - probability
    # Increase Thinking Functions likelihood
    else:
        output_cf_acc[['Ti', 'Te']] += probability
        output_cf_acc[['Fi', 'Fe']] += 1 - probability   
    
    # Use supporter model (Introvert vs Extrovert)
    modelObject = supporters["IvE"]
    cognitive_fn, probability = process_classify_sample(modelObject, sample)
    
    # Increase Introverted functions likelihood
    if cognitive_fn == "I":
        input_cf_acc[['Ni', 'Si']] += probability
        input_cf_acc[['Ne', 'Se']] += 1 - probability
        
        output_cf_acc[['Fi', 'Ti']] += probability
        output_cf_acc[['Fe', 'Te']] += 1 - probability
        
    # Increase  Extroverted functions likelihood
    else:
        input_cf_acc[['Ne', 'Se']] += probability
        input_cf_acc[['Ni', 'Si']] += 1 - probability
        
        output_cf_acc[['Fe', 'Te']] += probability
        output_cf_acc[['Fi', 'Ti']] += 1 - probability
    
    # Return: likelihoods of perceiving (input) cognitive functions
    # and judging (output) cogitive function
    return input_cf_acc, output_cf_acc

    
    
# Phase 2 of classification algorithm
# Detect the direction of the dominant function
# & Fix the classification if necessary
# Necessary: phase 1 results a two cognitive functions
# of the same direction (ex: Ni-Ti), this is not acceptable
# since there's no personality with these Dom-Aux functions
def phase2(sample, input_acc, output_acc):
    # Number of classifications completed until this step
    counter_models_ran = 6
    
    # Get max-likelihood data (probability & className)
    maxInput = {'name':input_acc.idxmax(), 'proba':input_acc.max()}
    maxOutput = {'name':output_acc.idxmax(), 'proba':output_acc.max()}
    
    # Get direction of cognitive function (className)
    maxInputDirection = cf_info[maxInput['name']]['direction']
    maxOutputDirection = cf_info[maxOutput['name']]['direction']
    
    # Get the next models ready by concatenating cognitive functions (ie. 'NiTe')
    cf_stack = np.array([input_acc.idxmax(), output_acc.idxmax()])
    phase2_model_name = "".join(cf_stack)
    
    # if both perceiving & judging classes (functions) are opposite direction
    if maxInputDirection != maxOutputDirection:
        # We know the max-likelihood two cognitive_functions (ie. Ni, Te)
        # this path will run the appropriate models to
        # determine which of these functions is Dominant (primary)
        # and which is Auxiliary (secondary)

        # determine which cognitive_function is the dominant one
        modelObject = models[phase2_model_name]
        dominant_cf_name, probability = process_classify_sample(modelObject, sample)
        counter_models_ran += 1
        
    # both perceiving & judging have same direction (they must NOT)
    else:
        # There is an ambiguity, the top 2 cognitive_functions cannot be
        # the same direction (ie. Ni, Ti) (they're both 'i')
        # One of them is correct, this function will detect which one is
        
        # Detect which of these functions is more accurate
        modelObject = models[phase2_model_name]
        dominant_cf_name, probability = process_classify_sample(modelObject, sample)
        counter_models_ran += 1
        
        
        # if dominant cognitive function is an input (perceiving) 
        # (ie: Ni, Ne, Si, Se)
        # flip the direction of the other -output- function
        # example: Ni-Ti  => Ni-Te
        if dominant_cf_name == input_acc.idxmax():
            problematic_cf = output_acc.idxmax()
            input_acc[input_acc.idxmax()] += probability
            fixed_cf_name = flip_cf_direction(problematic_cf)
            temp_acc = output_acc[fixed_cf_name]
            output_acc[fixed_cf_name] = output_acc[problematic_cf] + (1 - probability)
            output_acc[problematic_cf] = temp_acc
        
        # else if the dominant cognitive function is an output (judging)
        # (ie: Ti, Te, Fi, Fe)
        # flip the direction of the other -input- function
        # example: Ni-Ti => Ne-Ti
        else:
            problematic_cf = input_acc.idxmax()
            output_acc[output_acc.idxmax()] += probability
            fixed_cf_name = flip_cf_direction(problematic_cf)
            temp_acc = input_acc[fixed_cf_name]
            input_acc[fixed_cf_name] = input_acc[problematic_cf] + (1 - probability)
            input_acc[problematic_cf] = temp_acc
        
        # Now we fixed the direction of the (less) accurate cognitive function
        # We need to determine which function is dominant
        
        # Get the next models ready by concatenating cognitive functions (ie. 'NiTe')
        cf_stack = np.array([input_acc.idxmax(), output_acc.idxmax()])
        phase2_final_model_name = "".join(cf_stack)

        # Run model
        modelObject = models[phase2_final_model_name]
        dominant_cf_name, probability = process_classify_sample(modelObject, sample)
        counter_models_ran += 1
    
    # Get the winner (dominant) & auxiliary cognitive functions
    
    # If dominant function is input (perceiving)
    # increase its likelihood, decrease the other
    if dominant_cf_name == input_acc.idxmax():
        input_acc[input_acc.idxmax()] += probability
        output_acc[output_acc.idxmax()] += 1 - probability

    # else, dominant function is output (judging)
    # increase its likelihood, decrease the other
    else:
        output_acc[output_acc.idxmax()] += probability
        input_acc[input_acc.idxmax()] += 1 - probability

    # Stack the cognitive functions as: Dominant-Auxiliary
    cognitive_functions_stack = pd.Series({
            input_acc.idxmax(): input_acc[input_acc.idxmax()],
            output_acc.idxmax(): output_acc[output_acc.idxmax()]
            })
        
    dominant_function = cognitive_functions_stack.idxmax()
    auxiliary_function = cognitive_functions_stack.idxmin()
    

    # Convert cognitive functions to personality types
    personality_type = functions_to_letters(dominant_function + auxiliary_function)
    
    # Calculate probability
    probability = (cognitive_functions_stack[dominant_function] / counter_models_ran 
                   + cognitive_functions_stack[auxiliary_function] / counter_models_ran) / 2
    
    return personality_type, probability

# Run classification algorithm phases
def run(sample):
    input_acc, output_acc = phase1(sample)
    personality, probability = phase2(sample, input_acc, output_acc)
    return personality, probability


sentence = '''Hey, they call me the Adventurer, I do artworks and sometimes play guitar,
I enjoy spending time alone listening to music, I appreciate authenticity
and honesty, I'm always connected to my feelings and alert of it,
some people consider as an artist, but I'm not, and I won't let them
define what I am, regonized me?'''

personality, probability = run(sentence) # ISFP
print(f"Personality: {personality}\nLikelihood: {probability}")