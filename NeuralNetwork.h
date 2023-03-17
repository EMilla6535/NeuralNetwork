#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>
using std::string;
#include <valarray>
using std::valarray;
using std::slice;
#include <tuple>
using std::tuple;
#include <vector>
using std::vector;
#include <thread>
using std::thread;
#include <mutex>
using std::mutex;
using std::lock_guard;

typedef double net_type;
typedef valarray<net_type> net_array;
typedef valarray<net_array> net_multi_array;
typedef tuple<net_multi_array, net_multi_array, int> net_tuple;
typedef valarray<int> int_array;
typedef valarray<int_array> multi_int_array;

class NeuralNetwork
{
    public:
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;
        NeuralNetwork(NeuralNetwork &&obj);
        NeuralNetwork& operator=(NeuralNetwork &&obj);
        NeuralNetwork();
        NeuralNetwork(string shape, string cost_function, string activation);
        ~NeuralNetwork();
        void setParameters(string shape, string cost_function, string activation);
        void SGD(const net_tuple &training_data, int epochs, int mini_batch_size, double eta, double lambda, const net_tuple &test_data);
        void multiSGD(const net_tuple &, int, int, double, double, const net_tuple &, int);
        void saveNetwork();
        void saveParameters(int epochs, int mini_batch_size, double eta, double lambda);
        void saveStatistics(const valarray<double> &accuracy, const valarray<double> &all_cost);
        void loadNetwork(string path, int &epochs, int &mini_batch_size, double &eta, double &lambda);
    private:
        net_multi_array weights, grad_w, zeta, delta;
        net_multi_array bias, grad_b, activations;

        valarray<int> neurons;
        /** Cost
          * Quadratic     = 0
          * Cross-entropy = 1
          */
        /** Activation
         *  Sigmoid = 0
         *  Softmax = 1
         */
        int layers, cost, act_fn;
        vector<thread> th_vector;
        mutex mtex;

        void create();
        void feedforward(const net_array &x_input);
        void feedforward(const net_array &x_input, net_multi_array &act, net_multi_array &z);
        void backpropagation(const net_array &y_output, const net_array &x_input);
        void multiBackpropagation(const net_array &y_output, const net_array &x_input);
        void multiCaller(const int_array &idx, const net_multi_array &t_i, const net_multi_array &t_o);
        void monitor(int epochs, int e, double lambda, const net_tuple &test_data, valarray<double> &accuracy, valarray<double> &all_cost);
        net_array deltaLastLayer(const net_array &z, const net_array &a, const net_array &y);
        void update(double eta, double lambda, int n_data_set, int n_mini_batch);
        int evaluateAccuracy(const net_tuple &test_data);
        bool compareOutput(const net_array &net_output, const net_array &y_output);
        double evaluateCost(const net_tuple &data, double lambda);
        double costFunction(const net_array &a, const net_array &y);
};

#endif // NEURALNETWORK_H
