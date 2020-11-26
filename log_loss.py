# A simple log loss function with examples on a set of data 'passed_exams' and pre-calculated probabilities and probabilities_2

import numpy as np
from exam import passed_exam, probabilities, probabilities_2

# Function to calculate log-loss
def log_loss(probabilities,actual_class):
  return np.sum(-(1/actual_class.shape[0])*(actual_class*np.log(probabilities) + (1-actual_class)*np.log(1-probabilities)))

# Print passed_exam:

print(passed_exam)

# Calculate and print loss_1:

loss_1 = log_loss(probabilities, passed_exam)
print(loss_1)

# Calculate and print loss_2:

loss_2 = log_loss(probabilities_2, passed_exam)
print(loss_2)
