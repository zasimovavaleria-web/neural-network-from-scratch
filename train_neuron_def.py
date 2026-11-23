def predict(distance, minutes, weight_km, weight_min, bias):
    return weight_km * distance + weight_min * minutes + bias


def train_neuron(trips, epochs, learning_rate):
    # Начальные случайные догадки нейрона
    weight_km = 5.0
    weight_min = 5.0
    bias = 100.0

    for epoch in range(epochs):
        total_error = 0

        # Проходим по каждой поездке из списка
        for distance, minutes, real_cost in trips:
            # 1. Делаем прогноз (Forward pass)
            prediction = predict(distance, minutes, weight_km, weight_min, bias)

            # 2. Считаем ошибку
            error = real_cost - prediction
            total_error += abs(error)

            # 3. Обновляем веса (Gradient Descent)
            weight_km = weight_km + error * distance * learning_rate
            weight_min = weight_min + error * minutes * learning_rate

        # Печатаем прогресс каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch + 1:3d}: суммарная ошибка = {total_error:.2f} руб.")
    return weight_km, weight_min, bias


# Обучающий набор данных: (колометры, минуты, итоговая цена)
trips_data = [
    (10, 20, 460), 
    (5, 10, 280),
    (2, 15, 260),
    (15, 25, 600),
]

final_w_km, final_w_min, final_bias = train_neuron(
    trips=trips_data,
    epochs=300,learning_rate=0.001
)

print("\n--- Итоговые настройки нейрона ---")
print(f"Цена за 1 км: {final_w_km:.2f} руб. (ожидалось 20,00)")
print(f"Цена за 1 мин: {final_w_min:.2f} руб. (ожидалось 8,00)")

