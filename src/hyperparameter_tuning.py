# hyperparameter_tuning.py

from train import train_model
from sklearn.metrics import accuracy_score
from itertools import product
import joblib

# Define search grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [200, 500, 1000]
}

best_acc = 0
best_params = None
best_model = None
best_vectorizer = None

# Train and evaluate for each combination
for C, solver, max_iter in product(param_grid['C'], param_grid['solver'], param_grid['max_iter']):
    print(f"Training with C={C}, solver={solver}, max_iter={max_iter}")

    model, vectorizer, X_test, y_test = train_model(C=C, solver=solver, max_iter=max_iter)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_params = {'C': C, 'solver': solver, 'max_iter': max_iter}
        best_model = model
        best_vectorizer = vectorizer

print("\n✅ Best Accuracy:", best_acc)
print("✅ Best Parameters:", best_params)

# Save the best model and vectorizer for API.py
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(best_vectorizer, 'best_vectorizer.pkl')