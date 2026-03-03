import matplotlib.pyplot as plt

# 1. Simple SVM with Linear Kernel
class SVM40:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.w = [0.0, 0.0]
        self.b = 0.0

    def fit(self, X, y):
        # We only practice 40 times (iterations)
        for _ in range(40):
            for i in range(len(X)):
                # Linear Kernel: (w1*x1 + w2*x2) - b
                dot_prod = (self.w[0] * X[i][0] + self.w[1] * X[i][1]) - self.b
                
                # Check if point is on the wrong side or too close
                if y[i] * dot_prod < 1:
                    # Move the fence quickly!
                    self.w[0] += self.lr * (y[i] * X[i][0])
                    self.w[1] += self.lr * (y[i] * X[i][1])
                    self.b -= self.lr * y[i]

# 2. Data (Red Crabs vs Blue Turtles)
X = [[1, 2], [2, 1], [2, 3], [6, 8], [8, 7], [9, 9]]
y = [-1, -1, -1, 1, 1, 1]

# 3. Train
model = SVM40()
model.fit(X, y)

# 4. Graph the result
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=('red' if y[i] == -1 else 'blue'), s=100)

# Draw the line: y = (b - w1*x) / w2
x_pts = [0, 10]
y_pts = [(model.b - model.w[0] * x) / model.w[1] for x in x_pts]
plt.plot(x_pts, y_pts, 'k--', label="40-Iteration Fence")

plt.legend()
plt.title("SVM Trained in 40 Iterations")
plt.show()