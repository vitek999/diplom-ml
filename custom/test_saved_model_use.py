import keras
from Handler import Dataset
import numpy as np

classes = ['Приседания', 'Отжимания', 'Прыжки', 'Выпады', 'Пресс']
idx_to_class = { v: k for v, k in enumerate(classes) }
print(idx_to_class)

loaded_model = keras.models.load_model("my_model", compile=False)


# print()

def predict_clazz_exercise(text: str):
    dataset = Dataset(3, window_time=5, raw_json=text)
    print(dataset.x_feature.shape)
    res = loaded_model.predict(dataset.x_feature)
    answer = []
    for i in res:
        clazz = np.where(i == np.max(i))
        print(idx_to_class[clazz[0][0]])
        answer.append(idx_to_class[clazz[0][0]])
    return answer

# predict_clazz_exercise('')