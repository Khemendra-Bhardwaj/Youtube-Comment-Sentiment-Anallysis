import matplotlib.pyplot as plt
from classify.py import subjectivity_scoress, polarity_scores 

plt.figure(figsize=(8, 6))
plt.scatter(polarity_scores, subjectivity_values, c='blue', alpha=0.5)
plt.title('Polarity vs. Subjectivity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')

# Add grid lines
plt.grid(True)

# Fit a line through the data points using linear regression
coefficients = np.polyfit(polarity_scores, subjectivity_values, 1)
line = np.poly1d(coefficients)

# Create data points for the regression line
x_values = np.linspace(min(polarity_scores), max(polarity_scores), 100)
y_values = line(x_values)

# Plot the regression line
plt.plot(x_values, y_values, color='red', linestyle='--', label='Regression Line')

# Display the legend
plt.legend()

# Display the plot
plt.show()