from manage_data import fn
from lineal_model import Logistic_Regresion

model = Logistic_Regresion(alpha=0.1, epochs=10, batch_count=5, fn="sigmoid", verbose_epoch=True)

csvFn = fn.CSV()
TranformTime = fn.TextToTime("%H:%M:%S")

dataset = csvFn.read("dataset.csv", headers=True)
dataset = [[round(TranformTime.timetext_to_hours(x), 4), int(y)] for x, y in dataset]

x, y = csvFn.process(dataset, random_state=42, split=False)

model.fit(x, y)


print(f"\nModel evaluation: {model.evaluate(x, y) * 100:.2f}% accuracy")


print(f"Final weights: {model.weights.data}")
print(f"Final bias: {model.bias}")


print(f"\nPrediction for 30 minutes of study: {round(model.predict(.5)[0], 2) * 100}% chance of passing")
print(f"Prediction for 2 hours of study: {round(model.predict(2)[0], 2) * 100}% chance of passing")
print(f"Prediction for 5 hours of study: {round(model.predict(5)[0], 2) * 100}% chance of passing")


print("\numbral de 50% para clasificación:\n")


print(f"Classification for 30 minutes of study: {'Pass' if model.classify(.5, threshold=0.5)[0] == 1 else 'Fail'}")
print(f"Classification for 2 hours of study: {'Pass' if model.classify(2, threshold=0.5)[0] == 1 else 'Fail'}")
print(f"Classification for 5 hours of study: {'Pass' if model.classify(5, threshold=0.5)[0] == 1 else 'Fail'}")
