# Престейший нейрон: y = w * x + b
def predict_taxi_cost(distance):
    weight = 30  # w: цена за 1 км (вес признака)
    bias = 100  # b: стоимость подачи машины (смещение)

    prediction = weight * distance + bias
    return prediction


# Вызываем нейрон на разных растояниях
print("Поездка на 0 км:", predict_taxi_cost(0), "руб.")
print("Поездка на 5 км:", predict_taxi_cost(5), "руб.")
print("Поездка на 12 км:", predict_taxi_cost(12), "руб.")


# Считаем ошибку нейрона (реальная цена 420 руб. за 10 км)
real_cost = 420
error = real_cost - predict_taxi_cost(10)
print("Ошибка предсказания на 10 км:", error, "руб.")