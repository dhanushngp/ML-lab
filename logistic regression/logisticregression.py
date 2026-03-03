import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
#  STEP 1: Sigmoid Function
# ─────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ─────────────────────────────────────────
#  STEP 2: Log Loss (Binary Cross-Entropy)
# ─────────────────────────────────────────
def compute_loss(y, y_pred):
    n = len(y)
    loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    return loss

# ─────────────────────────────────────────
#  STEP 3: Training using Gradient Descent
# ─────────────────────────────────────────
def train(X, y, lr=0.1, epochs=1000):
    n_samples = len(X)
    weight = 0.0   # w
    bias   = 0.0   # b
    losses = []

    for epoch in range(epochs):
        # Forward pass
        z      = weight * X + bias
        y_pred = sigmoid(z)

        # Loss
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        # Gradients
        dw = np.mean((y_pred - y) * X)
        db = np.mean(y_pred - y)

        # Update weights
        weight -= lr * dw
        bias   -= lr * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Weight: {weight:.4f} | Bias: {bias:.4f}")

    return weight, bias, losses

# ─────────────────────────────────────────
#  STEP 4: Predict
# ─────────────────────────────────────────
def predict(X, weight, bias, threshold=0.5):
    z          = weight * X + bias
    probability = sigmoid(z)
    prediction  = (probability >= threshold).astype(int)
    return probability, prediction

# ─────────────────────────────────────────
#  DATASET — Hours Studied vs Pass/Fail
#  (Just like our discussion example!)
# ─────────────────────────────────────────
X = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y = np.array([0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1  ])
#              Fail ←────────────────────────────────────────────→ Pass

print("=" * 55)
print("       LOGISTIC REGRESSION — STUDENT PASS/FAIL")
print("=" * 55)
print()

# ─────────────────────────────────────────
#  TRAIN THE MODEL
# ─────────────────────────────────────────
weight, bias, losses = train(X, y, lr=0.1, epochs=1000)

print()
print(f"✅ Final Weight : {weight:.4f}")
print(f"✅ Final Bias   : {bias:.4f}")

# ─────────────────────────────────────────
#  TEST WITH NEW STUDENTS
# ─────────────────────────────────────────
print()
print("=" * 55)
print("             PREDICTIONS ON NEW STUDENTS")
print("=" * 55)

test_hours = np.array([1.0, 3.0, 5.5, 7.0, 9.0])

for hours in test_hours:
    prob, pred = predict(np.array([hours]), weight, bias)
    result = "✅ PASS" if pred[0] == 1 else "❌ FAIL"
    print(f"  Hours Studied: {hours:.1f}  |  Probability: {prob[0]:.2f}  |  {result}")

# ─────────────────────────────────────────
#  PLOT 1 — Sigmoid Curve + Data Points
# ─────────────────────────────────────────
X_plot = np.linspace(0, 11, 300)
z_plot = weight * X_plot + bias
prob_plot = sigmoid(z_plot)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(X_plot, prob_plot, color='royalblue', linewidth=2.5, label='Sigmoid Curve')
plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Decision Boundary (0.5)')
plt.scatter(X[y == 0], y[y == 0], color='tomato',      s=80, zorder=5, label='Fail (0)')
plt.scatter(X[y == 1], y[y == 1], color='limegreen',   s=80, zorder=5, label='Pass (1)')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression — Sigmoid Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# ─────────────────────────────────────────
#  PLOT 2 — Loss over Epochs
# ─────────────────────────────────────────
plt.subplot(1, 2, 2)
plt.plot(losses, color='darkorange', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Loss Curve (should decrease over time)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression_plot.png', dpi=150)
plt.show()
print()
print("📊 Plot saved as logistic_regression_plot.png")