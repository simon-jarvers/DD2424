import numpy as np
import pickle


def get_book_data(book_fname: str) -> str:
    with open(book_fname, mode="r", encoding="utf-8") as f:
        content = f.read()
    return content


def softmax(x):
    p = np.exp(x - np.max(x))
    return p / np.sum(p)


def load_model(fn):
    with open(f"{fn}.pkl", "rb") as f:
        model = pickle.load(f)
    print(f'Loaded {fn}.pkl')
    return model


def save_text(fn, text):
    with open(f'{fn}.txt', 'a') as f:
        f.write(text)

