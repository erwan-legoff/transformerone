
import os


def save_dict_to_file(data, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def save_list_to_file(data, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item}\n")

def save_str_to_file(data, file_name):
    dirs = os.path.dirname(file_name)
    if(dirs != ''):
        os.makedirs(dirs, exist_ok=True)
        
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(data)


def write_first_2000_chars_to_file(text, file_name):
    save_str_to_file(text[:2000], file_name)
    
def load_data(training_file, evaluation_file):
    with open(training_file, 'r', encoding='utf-8') as f:
        training_text = f.read()
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        eval_text = f.read()
    return training_text, eval_text