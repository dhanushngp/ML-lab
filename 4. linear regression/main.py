X = []
Y = []
xSubxBar = []
ySubyBar = []
xSubxBarsq = 0
numerator = 0

size = int(input("Enter size of data:"))
print()

for i in range(size):
    X.append(int(input(f"Enter x{i+1}:")))
    
print()
    
for i in range(size):
    Y.append(int(input(f"Enter y{i+1}:")))
    
print()
    
xBar = sum(X) / len(X)
yBar = sum(Y) / len(Y)

for i in X:
    xSubxBar.append(i - xBar)
for i in Y:
    ySubyBar.append(i - yBar)
for i in X:
    xSubxBarsq += (i - xBar)**2
    
for i in range(len(X)):
    numerator += (xSubxBar[i] * ySubyBar[i])
    
m = numerator / xSubxBarsq

# y = mx + c
c = yBar - (m * xBar)

x = int(input("Enter value to predict: "))
yCap = m * x + c

print()

print("Predicted value:",round(yCap, 2))

print(m)
print(c)