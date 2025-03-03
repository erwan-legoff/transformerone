
def save_dict_to_file(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def save_list_to_file(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item}\n")

def save_str_to_file(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(data)


def write_first_2000_chars_to_file(text, file_name):
    save_str_to_file(text[:2000], file_name)
    