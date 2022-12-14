#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Compute the dot product of two vectors
double dot(vector<double> x, vector<double> y) {
  double sum = 0;
  for (int i = 0; i < x.size(); i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

// Compute the prediction of the SVM model
// given input features x and model weights w
double predict(vector<double> x, vector<double> w) {
  return dot(x, w);
}

// Compute the gradient of the hinge loss with respect to the model weights
vector<double> gradient(vector<vector<double>> X, vector<double> y, vector<double> w) {
  vector<double> grad(w.size());
  for (int i = 0; i < X.size(); i++) {
    double y_pred = predict(X[i], w);
    if (1 - y[i] * y_pred > 0) {
      // The point is on the wrong side of the margin,
      // so we need to update the model weights
      for (int j = 0; j < w.size(); j++) {
        grad[j] += -y[i] * X[i][j];
      }
    }
  }
  return grad;
}

// Perform gradient descent to learn the model weights
void gradient_descent(vector<vector<double>> X, vector<double> y, vector<double>& w,
                      double learning_rate, int num_iters) {
  for (int i = 0; i < num_iters; i++) {
    vector<double> grad = gradient(X, y,
