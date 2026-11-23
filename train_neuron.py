# Исходные данные из реальной поездки: 10 км, 20 минут, реальный чек — 460 руб.
distance = 10
minutes = 20
real_cost = 460


# Начальные «догадки» нейрона (пока он ошибается)
weight_km = 20.0
weight_min = 5.0
bias = 100.0
learning_rate = 0.001


# Запускаем цикл обучения на 15 шагов (эпох)
for step in range(15):
    # 1. Делаем прогноз (Forward pass)
    prediction = weight_km * distance + weight_min * minutes + bias

    # 2. Считаем ошибку
    error = real_cost - prediction

    # 3. Обновляем веса (Gradient Descent)
    weight_km = weight_km + error * distance * learning_rate
    weight_min = weight_min + error * minutes * learning_rate

    print(f"Шаг {step +1}: прогноз = {round(prediction, 2)} руб., ошибка = {round(error, 2)}")


print("\n Итоговые настройки нейрона:")
print(f"Цена за 1 км: {round(weight_km, 2)} руб., за 1 мин: {round(weight_min, 2)} руб.")




