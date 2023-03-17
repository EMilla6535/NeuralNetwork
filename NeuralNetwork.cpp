#include "NeuralNetwork.h"
#include <iostream>
using std::cout;
using std::endl;
using std::cin;
using std::ios;
#include <cmath>
#include <algorithm>
using std::fill;
using std::transform;
using std::max_element;
using std::min_element;
using std::shuffle;
#include <random>
using std::default_random_engine;
using std::mt19937;
using std::normal_distribution;
#include <chrono>
using namespace std::chrono;
#include <fstream>
using std::ofstream;
using std::ifstream;
#include <stdexcept>
using std::runtime_error;

net_type sigmoid(net_type z)
{
    return (1.0 / (1.0 + exp(-z)));
}
net_type sigmoidPrime(net_type z)
{
    return (sigmoid(z) * (1.0 - sigmoid(z)));
}
net_array softmax(const net_array &z)
{
    net_array result(z.size());
    for(size_t i = 0; i < result.size(); i++)
    {
        result[i] = exp(z[i])/exp(z).sum();
    }
    return result;
}
class updateWeights
{
    public:
        updateWeights(double e, double l, int n_d_s, int n_m_b)
        : eta(e), lambda(l), n_data_set(n_d_s), n_mini_batch(n_m_b){}
        net_type operator()(net_type w, net_type g_w)
        {
            return ((1.0 - eta * (lambda / n_data_set)) * w - (eta / n_mini_batch) * g_w);
        }
    private:
        double eta, lambda;
        int n_data_set, n_mini_batch;
};
class updateBiases
{
    public:
        updateBiases(double e, int n_m_b)
        : eta(e), n_mini_batch(n_m_b){}
        net_type operator()(net_type b, net_type g_b)
        {
            return (b - (eta / n_mini_batch) * g_b);
        }
    private:
        double eta;
        int n_mini_batch;
};
int strToInt(string s)
{
    int i, res = 0, esc = 1;
    for(i = s.length() - 1; i > -1; i--)
    {
        res += (s.at(i) - '0') * esc;
        esc *= 10;
    }
    return res;
}
int getNLayers(string shape)
{
    unsigned int i, n = 0;
    for(i = 0; i < shape.length(); i++)
    {
        if(shape.at(i) == ',')
        {
            n++;
        }
    }
    n++;
    return n;
}
void getNNeurons(valarray<int> &n_neurons, string shape)
{
    string buffer = "";
    int d_count = 0;
    for(unsigned int i = 0; i < shape.length(); i++)
    {
        switch(shape.at(i))
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                buffer.append(shape, i, 1);
                break;
            case ']':
            case ',':
                n_neurons[d_count] = strToInt(buffer);
                buffer = "";
                d_count++;
                break;
            default:
                break;
        }
    }
}
net_array transpose(const net_array &matrix, int rows, int cols)
{
    net_array t_matriz(matrix.size());
    for(int i = 0; i < cols; i++)
    {
        t_matriz[slice(rows * i, rows, 1)] = matrix[slice(i, rows, cols)];
    }
    return t_matriz;
}
net_array randomGenerator(int v_size)
{
    static unsigned long seed = 0;
    seed++;
    mt19937 gen(seed);
    normal_distribution<net_type> dist(0.0, 1.0);
    net_array buffer(v_size);

    for(auto it = begin(buffer); it != end(buffer); ++it)
    {
        *it = dist(gen);
    }
    return buffer;
}

NeuralNetwork::NeuralNetwork(NeuralNetwork &&obj) : weights(std::move(obj.weights)),
                                     grad_w(std::move(obj.grad_w)),
                                     zeta(std::move(obj.zeta)),
                                     delta(std::move(obj.delta)),
                                     bias(std::move(obj.bias)),
                                     grad_b(std::move(obj.grad_b)),
                                     activations(std::move(obj.activations)),
                                     neurons(std::move(obj.neurons)),
                                     th_vector(std::move(obj.th_vector))
{
    layers = obj.layers;
    cost = obj.cost;
    act_fn = obj.act_fn;
}
NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork &&obj)
{
    for(size_t i = 0; i < th_vector.size(); i++)
    {
        if(th_vector.at(i).joinable())
        {
            th_vector.at(i).join();
        }
    }
    th_vector = std::move(obj.th_vector);
    weights = std::move(obj.weights);
    grad_w = std::move(obj.grad_w);
    zeta = std::move(obj.zeta);
    delta = std::move(obj.delta);
    bias = std::move(obj.bias);
    grad_b = std::move(obj.grad_b);
    activations = std::move(obj.activations);
    neurons = std::move(obj.neurons);
    layers = obj.layers;
    cost = obj.cost;
    act_fn = obj.act_fn;

    return *this;
}
NeuralNetwork::NeuralNetwork()
{
    layers = 2;
    neurons.resize(layers, 1);
    create();
    cost = 0;
    act_fn = 0;
}
NeuralNetwork::NeuralNetwork(string shape, string cost_function, string activation)
{
    setParameters(shape, cost_function, activation);
}
NeuralNetwork::~NeuralNetwork()
{
    for(size_t i = 0; i < th_vector.size(); i++)
    {
        if(th_vector.at(i).joinable())
        {
            th_vector.at(i).join();
        }
    }
}
void NeuralNetwork::setParameters(string shape, string cost_function, string activation)
{
    layers = getNLayers(shape);
    if(layers < 2)
    {
        layers = 2;
        neurons.resize(layers, 1);
    }
    else
    {
        neurons.resize(layers);
        getNNeurons(neurons, shape);
    }
    create();
    cost = 0;
    if(cost_function == "quadratic_cost")
    {
        cost = 0;
    }
    if(cost_function == "cross_entropy_cost")
    {
        cost = 1;
    }
    act_fn = 0;
    if(activation == "sigmoid")
    {
        act_fn = 0;
    }
    if(activation == "softmax")
    {
        act_fn = 1;
    }
}
void NeuralNetwork::SGD(const net_tuple &training_data, int epochs, int mini_batch_size, double eta, double lambda, const net_tuple &test_data)
{
    net_multi_array train_input = std::get<0>(training_data);
    net_multi_array train_output = std::get<1>(training_data);
    int train_size = std::get<2>(training_data);
    int i, j, e;
    int_array index(train_size);
    int_array mini_batch;
    for(i = 0; i < train_size; i++)
    {
        index[i] = i;
    }
    int batch_size = 0;
    valarray<double> accuracy(epochs);
    valarray<double> all_cost(epochs);
    unsigned seed = system_clock::now().time_since_epoch().count();
    cout << "=== Start SGD ===" << endl;
    steady_clock::time_point t1 = steady_clock::now();
    for(e = 0; e < epochs; e++)
    {
        shuffle(begin(index), end(index), default_random_engine(seed));
        for(i = 0; i < train_size; i += mini_batch_size)
        {
            batch_size = ((i + mini_batch_size) > train_size) ? (train_size - i) : mini_batch_size;
            mini_batch.resize(batch_size);
            mini_batch = index[slice(i, batch_size, 1)];
            for(j = 0; j < batch_size; j++)
            {
                backpropagation(train_output[mini_batch[j]], train_input[mini_batch[j]]);
            }
            update(eta, lambda, train_size, batch_size);
        }
        cout << "Epoch " << e + 1 << " of " << epochs << " completed." << endl;
        monitor(epochs, e, lambda, test_data, accuracy, all_cost);
    }
    steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double> >(t2 - t1);
    cout << "Epochs + test time = " << time_span.count() << endl;

    cout << "====== Variables used ====" << endl;
    cout << "Epochs          : " << epochs << endl;
    cout << "Mini batch size : " << mini_batch_size << endl;
    cout << "L. R. Eta       : " << eta << endl;
    cout << "Lambda          : " << lambda << endl;

    cout << "====== Final result ======" << endl;
    cout << "Max accuracy    : " << *max_element(begin(accuracy), end(accuracy)) << "%" << endl;
    cout << "At epoch        : " << max_element(begin(accuracy), end(accuracy)) - begin(accuracy) << endl;
    cout << "Min accuracy    : " << *min_element(begin(accuracy), end(accuracy)) << "%" << endl;
    cout << "At epoch        : " << min_element(begin(accuracy), end(accuracy)) - begin(accuracy) << endl;
    cout << "Average accuracy: " << accuracy.sum() / epochs << "%" << endl;
    cout << "Max cost        : " << *max_element(begin(all_cost), end(all_cost)) << endl;
    cout << "At epoch        : " << max_element(begin(all_cost), end(all_cost)) - begin(all_cost) << endl;
    cout << "Min cost        : " << *min_element(begin(all_cost), end(all_cost)) << endl;
    cout << "At epoch        : " << min_element(begin(all_cost), end(all_cost)) - begin(all_cost) << endl;
    cout << "Average cost    : " << all_cost.sum() / all_cost.size() << endl;
    cout << "==========================" << endl;

    char r;
    cout << "Save results into files (y/n) ? -> ";
    while((r != 'y') && (r != 'n'))
    {
        cin >> r;
    }
    switch(r)
    {
        case 'y':
            try
            {
                saveNetwork();
                saveParameters(epochs, mini_batch_size, eta, lambda);
                saveStatistics(accuracy, all_cost);
            }
            catch(runtime_error &e)
            {
                cout << "Error saving data -> " << e.what() << endl;
            }
            break;
        case 'n':
        default:
            break;
    }
}
void NeuralNetwork::multiSGD(const net_tuple &training_data, int epochs, int mini_batch_size, double eta, double lambda, const net_tuple &test_data, int threads)
{
    net_multi_array train_input = std::get<0>(training_data);
    net_multi_array train_output = std::get<1>(training_data);
    int train_size = std::get<2>(training_data);
    int i, j, e, act_threads;
    int_array index(train_size);
    multi_int_array mini_batch(threads);
    bool b_flag = true;
    for(i = 0; i < train_size; i++)
    {
        index[i] = i;
    }
    int batch_size = 0;
    valarray<double> accuracy(epochs);
    valarray<double> all_cost(epochs);
    unsigned seed = system_clock::now().time_since_epoch().count();
    cout << "=== Start SGD ===" << endl;
    steady_clock::time_point t1 = steady_clock::now();
    for(e = 0; e < epochs; e++)
    {
        shuffle(begin(index), end(index), default_random_engine(seed));
        b_flag = true;
        act_threads = threads;
        for(i = 0; i < train_size; i += (mini_batch_size * threads))
        {
            for(j = 0; j < threads && b_flag; j++)
            {
                batch_size = (i + (mini_batch_size * (j + 1))) > train_size ? (train_size - (i + (mini_batch_size * j))) : mini_batch_size;
                b_flag = (batch_size == mini_batch_size) ? true : false;
                if(!b_flag)
                {
                    act_threads = (threads - (j + 1));
                }
                mini_batch[j].resize(batch_size);
                mini_batch[j] = index[slice(i + (mini_batch_size * j), batch_size, 1)];
            }
            th_vector.resize(act_threads);
            for(j = 0; j < act_threads; j++)
            {
                th_vector[j] = thread(&NeuralNetwork::multiCaller, this, std::cref(mini_batch[j]), std::cref(train_input), std::cref(train_output));
            }
            for(size_t k = 0; k < th_vector.size(); k++)
            {
                if(th_vector.at(k).joinable())
                {
                    th_vector.at(k).join();
                }
            }
            update(eta, lambda, train_size,((mini_batch_size * threads - 1) + batch_size)/threads);
        }
        cout << "Epoch " << e + 1 << " of " << epochs << " completed." << endl;
        monitor(epochs, e, lambda, test_data, accuracy, all_cost);
    }
    steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double> >(t2 - t1);
    cout << "Epochs + test time = " << time_span.count() << endl;

    cout << "====== Variables used ====" << endl;
    cout << "Epochs          : " << epochs << endl;
    cout << "Mini batch size : " << mini_batch_size << endl;
    cout << "L. R. Eta       : " << eta << endl;
    cout << "Lambda          : " << lambda << endl;

    cout << "====== Final result ======" << endl;
    cout << "Max accuracy    : " << *max_element(begin(accuracy), end(accuracy)) << "%" << endl;
    cout << "At epoch        : " << max_element(begin(accuracy), end(accuracy)) - begin(accuracy) << endl;
    cout << "Min accuracy    : " << *min_element(begin(accuracy), end(accuracy)) << "%" << endl;
    cout << "At epoch        : " << min_element(begin(accuracy), end(accuracy)) - begin(accuracy) << endl;
    cout << "Average accuracy: " << accuracy.sum() / epochs << "%" << endl;
    cout << "Max cost        : " << *max_element(begin(all_cost), end(all_cost)) << endl;
    cout << "At epoch        : " << max_element(begin(all_cost), end(all_cost)) - begin(all_cost) << endl;
    cout << "Min cost        : " << *min_element(begin(all_cost), end(all_cost)) << endl;
    cout << "At epoch        : " << min_element(begin(all_cost), end(all_cost)) - begin(all_cost) << endl;
    cout << "Average cost    : " << all_cost.sum() / all_cost.size() << endl;
    cout << "==========================" << endl;

    char r;
    cout << "Save results into files (y/n) ? -> ";
    while((r != 'y') && (r != 'n'))
    {
        cin >> r;
    }
    switch(r)
    {
        case 'y':
            try
            {
                saveNetwork();
                saveParameters(epochs, mini_batch_size, eta, lambda);
                saveStatistics(accuracy, all_cost);
            }
            catch(runtime_error &e)
            {
                cout << "Error saving data -> " << e.what() << endl;
            }
            break;
        case 'n':
        default:
            break;
    }
}
void NeuralNetwork::saveNetwork()
{
    ofstream f_weights("save\\weights.txt");
    ofstream f_biases("save\\biases.txt");

    if(!f_weights)
    {
        throw runtime_error("Can't open file : weights.txt");
    }
    if(!f_biases)
    {
        throw runtime_error("Can't open file : biases.txt");
    }
    for(int i = 0; i < layers - 1; i++)
    {
        f_weights << weights[i].size() << endl;
        for(size_t j = 0; j < weights[i].size(); j++)
        {
            f_weights << weights[i][j] << endl;
        }
        f_biases << bias[i].size() << endl;
        for(size_t j = 0; j < bias[i].size(); j++)
        {
            f_biases << bias[i][j] << endl;
        }
    }
    f_weights.close();
    f_biases.close();
    cout << "Weights and Biases saved." << endl;
}
void NeuralNetwork::saveParameters(int epochs, int mini_batch_size, double eta, double lambda)
{
    ofstream f_params("save\\parameters.txt");
    if(!f_params)
    {
        throw runtime_error("Can't open file : parameters.txt");
    }
    f_params << "Layers_: " << layers << endl;
    for(int i = 0; i < layers; i++)
    {
        f_params << "Neurons_at_layer_" << i + 1 << "_: " << neurons[i] << endl;
    }
    f_params << "Cost_function_: ";
    switch(cost)
    {
        case 0:
            f_params << "quadratic_cost" << endl;
            break;
        case 1:
            f_params << "cross_entropy_cost" << endl;
            break;
        default:
            f_params << "quadratic_cost" << endl;
            break;
    }
    f_params << "Activation_function_: ";
    switch(act_fn)
    {
        case 0:
            f_params << "sigmoid" << endl;
            break;
        case 1:
            f_params << "softmax" << endl;
            break;
        default:
            f_params << "sigmoid" << endl;
            break;
    }
    f_params << "Epochs_: " << epochs <<endl;
    f_params << "Mini_batch_size_: " << mini_batch_size << endl;
    f_params << "Eta_: " << eta << endl;
    f_params << "Lambda_: " << lambda << endl;
    f_params.close();
    cout << "Parameters saved." << endl;
}
void NeuralNetwork::saveStatistics(const valarray<double> &accuracy, const valarray<double> &all_cost)
{
    ofstream f_stats("save\\statistics.txt");
    ofstream f_accuracy("save\\accuracy.txt");
    ofstream f_cost("save\\costs.txt");
    if(!f_stats)
    {
        throw runtime_error("Can't open file : statistics.txt");
    }
    if(!f_accuracy)
    {
        throw runtime_error("Can't open file : accuracy.txt");
    }
    if(!f_cost)
    {
        throw runtime_error("Can't open file : costs.txt");
    }
    f_stats << "Max accuracy     : " << *max_element(begin(accuracy), end(accuracy)) << "%" << endl;
    f_stats << "At epoch         : " << max_element(begin(accuracy), end(accuracy)) - begin(accuracy) << endl;
    f_stats << "Min accuracy     : " << *min_element(begin(accuracy), end(accuracy)) << "%" << endl;
    f_stats << "At epoch         : " << min_element(begin(accuracy), end(accuracy)) - begin(accuracy) << endl;
    f_stats << "Average accuracy : " << accuracy.sum() / accuracy.size() << "%" << endl;
    f_stats << "Max cost         : " << *max_element(begin(all_cost), end(all_cost)) << endl,
    f_stats << "At epoch         : " << max_element(begin(all_cost), end(all_cost)) - begin(all_cost) << endl;
    f_stats << "Min cost         : " << *min_element(begin(all_cost), end(all_cost)) << endl;
    f_stats << "At epoch         : " << min_element(begin(all_cost), end(all_cost)) - begin(all_cost) << endl;
    f_stats << "Average cost     : " << all_cost.sum() / all_cost.size();
    f_stats.close();
    cout << "Statistics saved." << endl;
    for(size_t i = 0; i < accuracy.size(); i++)
    {
        f_accuracy << accuracy[i] << endl;
    }
    f_accuracy.close();
    cout << "Accuracy saved." << endl;
    for(size_t i = 0; i < all_cost.size(); i++)
    {
        f_cost << all_cost[i] << endl;
    }
    f_cost.close();
    cout << "Cost saved." << endl;
}
void NeuralNetwork::loadNetwork(string path, int &epochs, int &mini_batch_size, double &eta, double &lambda)
{
    ifstream f_weights(path + "weights.txt");
    ifstream f_biases(path + "biases.txt");
    ifstream f_parameters(path + "parameters.txt");
    if(!f_weights)
    {
        throw runtime_error("Can't open file : " + path + "weights.txt");
    }
    if(!f_biases)
    {
        throw runtime_error("Can't open file : " + path + "biases.txt");
    }
    if(!f_parameters)
    {
        throw runtime_error("Can't open file : " + path + "parameters.txt");
    }
    string buffer, c_function, a_function;
    f_parameters >> buffer >> layers;
    neurons.resize(layers);
    for(int i = 0; i < layers; i++)
    {
        f_parameters >> buffer >> neurons[i];
    }
    f_parameters >> buffer >> c_function;
    cost = 0;
    if(c_function == "quadratic_cost")
    {
        cost = 0;
    }
    if(c_function == "cross_entropy_cost")
    {
        cost = 1;
    }
    f_parameters >> buffer >> a_function;
    act_fn = 0;
    if(a_function == "sigmoid")
    {
        act_fn = 0;
    }
    if(a_function == "softmax")
    {
        act_fn = 1;
    }
    f_parameters >> buffer >> epochs;
    f_parameters >> buffer >> mini_batch_size;
    f_parameters >> buffer >> eta;
    f_parameters >> buffer >> lambda;
    cout << "Epochs : " << epochs << endl;
    cout << "MBS    : " << mini_batch_size << endl;
    cout << "Eta    : " << eta << endl;
    cout << "Lambda : " << lambda << endl;
    f_parameters .close();
    cout << "Parameters loaded." << endl;
    int w_size = 0;
    weights.resize(layers - 1);
    grad_w.resize(layers - 1);
    zeta.resize(layers - 1);
    delta.resize(layers - 1);
    for(int i = 0; i < layers - 1; i++)
    {
        f_weights >> w_size;
        weights[i].resize(w_size);
        grad_w[i].resize(w_size, 0.0);
        zeta[i].resize(neurons[i + 1], 0.0);
        delta[i].resize(neurons[i + 1], 0.0);
        for(int j = 0; j < w_size; j++)
        {
            f_weights >> weights[i][j];
        }
    }
    f_weights.close();
    cout << "Weights loaded." << endl;
    int b_size = 0;
    bias.resize(layers - 1);
    grad_b.resize(layers - 1);
    activations.resize(layers - 1);
    for(int i = 0; i < layers - 1; i++)
    {
        f_biases >> b_size;
        bias[i].resize(b_size);
        grad_b[i].resize(b_size, 0.0);
        activations[i].resize(neurons[i + 1], 0.0);
        for(int j = 0; j < b_size; j++)
        {
            f_biases >> bias[i][j];
        }
    }
    f_biases.close();
    cout << "Biases loaded." << endl;
}
void NeuralNetwork::create()
{
    weights.resize(layers - 1);
    grad_w.resize(layers - 1);
    zeta.resize(layers - 1);
    delta.resize(layers - 1);

    bias.resize(layers - 1);
    grad_b.resize(layers - 1);
    activations.resize(layers - 1);

    for(int i = 0; i < layers - 1; i++)
    {
        weights[i] = randomGenerator(neurons[i + 1] * neurons[i]);
        grad_w[i].resize(neurons[i + 1] * neurons[i], 0.0);
        zeta[i].resize(neurons[i + 1], 0.0);
        delta[i].resize(neurons[i + 1], 0.0);

        bias[i] = randomGenerator(neurons[i + 1]);
        grad_b[i].resize(neurons[i + 1], 0.0);
        activations[i].resize(neurons[i + 1], 0.0);
    }
}
void NeuralNetwork::feedforward(const net_array &x_input)
{
    int cols, rows;
    net_array buffer;
    for(int i = 1; i < layers; i++)
    {
        cols = neurons[i - 1];
        rows = neurons[i];
        buffer.resize(cols);
        for(int j = 0; j < rows; j++)
        {
            buffer = weights[i - 1][slice(cols * j, cols, 1)];
            if(i == 1)
            {
                zeta[i - 1][j] = ((x_input * buffer).sum() + bias[i - 1][j]);
            }
            else
            {
                zeta[i - 1][j] = ((activations[i - 2] * buffer).sum() + bias[i - 1][j]);
            }
            if((act_fn == 0) || (i != layers - 1))
            {
                activations[i - 1][j] = sigmoid(zeta[i - 1][j]);
            }
        }
        if((i == layers - 1) && (act_fn != 0))
        {
            activations[i - 1] = softmax(zeta[i - 1]);
        }
    }
}
void NeuralNetwork::feedforward(const net_array &x_input, net_multi_array &act, net_multi_array &z)
{
    int cols, rows;
    net_array buffer;
    for(int i = 1; i < layers; i++)
    {
        cols = neurons[i - 1];
        rows = neurons[i];
        buffer.resize(cols);
        for(int j = 0; j < rows; j++)
        {
            buffer = weights[i - 1][slice(cols * j, cols, 1)];
            if(i == 1)
            {
                z[i - 1][j] = ((x_input * buffer).sum() + bias[i - 1][j]);
            }
            else
            {
                z[i - 1][j] = ((act[i - 2] * buffer).sum() + bias[i - 1][j]);
            }
            if((act_fn == 0) || (i != layers - 1))
            {
                act[i - 1][j] = sigmoid(z[i - 1][j]);
            }
        }
        if((i == layers - 1) && (act_fn != 0))
        {
            act[i - 1] = softmax(z[i - 1]);
        }
    }
}
void NeuralNetwork::backpropagation(const net_array &y_output, const net_array &x_input)
{
    int rows, cols;
    net_array buffer, t_weight;

    feedforward(x_input);

    for(int i = layers - 1; i > 0; i--)
    {
        if(i == layers - 1)
        {
            delta[i - 1] = deltaLastLayer(zeta[i - 1], activations[i - 1], y_output);
        }
        else
        {
            rows = neurons[i + 1];
            cols = neurons[i];
            t_weight.resize(rows * cols);
            t_weight = transpose(weights[i], rows, cols);
            buffer.resize(rows);
            for(int j = 0; j < cols; j++)
            {
                buffer = t_weight[slice(j * rows, rows, 1)];
                delta[i - 1][j] = (buffer * delta[i]).sum() * sigmoidPrime(zeta[i - 1][j]);
            }
        }
        grad_b[i - 1] += delta[i - 1];
        rows = neurons[i];
        cols = neurons[i - 1];
        for(int j = 0; j < rows; j++)
        {
            buffer.resize(cols, delta[i - 1][j]);
            if(i == 1)
            {
                grad_w[i - 1][slice(j * cols, cols, 1)] += (buffer * x_input);
            }
            else
            {
                grad_w[i - 1][slice(j * cols, cols, 1)] += (buffer * activations[i - 2]);
            }
        }
    }
}
void NeuralNetwork::multiBackpropagation(const net_array &y_output, const net_array &x_input)
{
    int rows, cols;
    net_array buffer, t_weight;
    net_multi_array act(layers - 1);
    net_multi_array z(layers - 1);
    net_multi_array d(layers - 1);

    for(int i = 0; i < layers - 1; i++)
    {
        act[i].resize(neurons[i + 1], 0.0);
        z[i].resize(neurons[i + 1], 0.0);
        d[i].resize(neurons[i + 1], 0.0);
    }
    feedforward(x_input, act, z);

    for(int i = layers - 1; i > 0; i--)
    {
        if(i == layers - 1)
        {
            d[i - 1] = deltaLastLayer(z[i - 1], act[i - 1], y_output);
        }
        else
        {
            rows = neurons[i + 1];
            cols = neurons[i];
            t_weight.resize(rows * cols);
            t_weight = transpose(weights[i], rows, cols);
            buffer.resize(rows);
            for(int j = 0; j < cols; j++)
            {
                buffer = t_weight[slice(j * rows, rows, 1)];
                d[i - 1][j] = (buffer * d[i]).sum() * sigmoidPrime(z[i - 1][j]);
            }
        }
        lock_guard<mutex> lock(mtex);
        grad_b[i - 1] += d[i - 1];
        rows = neurons[i];
        cols = neurons[i - 1];
        for(int j = 0; j < rows; j++)
        {
            buffer.resize(cols, d[i - 1][j]);
            if(i == 1)
            {
                grad_w[i - 1][slice(j * cols, cols, 1)] += (buffer * x_input);
            }
            else
            {
                grad_w[i - 1][slice(j * cols, cols, 1)] += (buffer * act[i - 2]);
            }
        }
    }
}
void NeuralNetwork::multiCaller(const int_array &idx, const net_multi_array &t_i, const net_multi_array &t_o)
{
    //int_array &index = const_cast<int_array&>(idx);
    for(size_t i = 0; i < idx.size(); i++)
    {
        multiBackpropagation(t_o[idx[i]], t_i[idx[i]]);
    }
}
void NeuralNetwork::monitor(int epochs, int e, double lambda, const net_tuple &test_data, valarray<double> &accuracy, valarray<double> &all_cost)
{
    int test_result = 0;
    test_result = evaluateAccuracy(test_data);
    cout << "Test results : " << test_result << " of " << std::get<2>(test_data) << "." << endl;
    accuracy[e] = (test_result * 100.0) / std::get<2>(test_data);
    cout << "Accuracy     : " << accuracy[e] << "%" << endl;
    all_cost[e] = evaluateCost(test_data, lambda);
    cout << "Cost         : " << all_cost[e] << endl;
}
net_array NeuralNetwork::deltaLastLayer(const net_array &z, const net_array &a, const net_array &y)
{
    switch(cost)
    {
        case 0:
            return (a - y) * z.apply(sigmoidPrime);
            break;
        case 1:
            return (a - y);
            break;
        default:
            break;
    }
    return (a - y) * z.apply(sigmoidPrime);
}
void NeuralNetwork::update(double eta, double lambda, int n_data_set, int n_mini_batch)
{
    updateWeights u_w(eta, lambda, n_data_set, n_mini_batch);
    updateBiases u_b(eta, n_mini_batch);
    for(int i = 0; i < layers - 1; i++)
    {
        transform(begin(weights[i]), end(weights[i]), begin(grad_w[i]), begin(weights[i]), u_w);
        fill(begin(grad_w[i]), end(grad_w[i]), 0.0);

        transform(begin(bias[i]), end(bias[i]), begin(grad_b[i]), begin(bias[i]), u_b);
        fill(begin(grad_b[i]), end(grad_b[i]), 0.0);
    }
}
int NeuralNetwork::evaluateAccuracy(const net_tuple &test_data)
{
    int result = 0;
    net_multi_array test_input = std::get<0>(test_data);
    net_multi_array test_output = std::get<1>(test_data);
    for(int i = 0; i < std::get<2>(test_data); i++)
    {
        feedforward(test_input[i]);
        result = compareOutput(activations[layers - 2], test_output[i]) ? result + 1 : result;
    }
    return result;
}
bool NeuralNetwork::compareOutput(const net_array &net_output, const net_array &y_output)
{
    auto net_pos = max_element(begin(net_output), end(net_output));
    auto y_pos = max_element(begin(y_output), end(y_output));
    return ((net_pos - begin(net_output)) == (y_pos - begin(y_output)));
}
double NeuralNetwork::evaluateCost(const net_tuple &data, double lambda)
{
    net_multi_array data_input = std::get<0>(data);
    net_multi_array data_output = std::get<1>(data);
    int data_size = std::get<2>(data);
    double total_cost = 0.0, total_w = 0.0;

    for(int i = 0; i < data_size; i++)
    {
        feedforward(data_input[i]);
        total_cost += costFunction(activations[layers - 2], data_output[i]) / data_size;
    }
    for(int i = 0; i < layers - 1; i++)
    {
        total_w += (pow(weights[i], 2.0)).sum();
    }
    total_cost += 0.5 * (lambda / data_size) * total_w;
    return total_cost;
}
double NeuralNetwork::costFunction(const net_array &a, const net_array &y)
{
    net_array result(a.size());
    switch(cost)
    {
        case 1:
            for(size_t i = 0; i < a.size(); i++)
            {
                result[i] = ((a[i] == 1.0) && (y[i] == 1.0)) ? 0.0 : (-y[i] * log(a[i])) - (1.0 - y[i]) * log(1.0 - a[i]);
            }
            break;
        case 0:
        default:
            for(size_t i = 0; i < a.size(); i++)
            {
                result[i] = 0.5 * pow(a[i] - y[i], 2.0);
            }
            break;
    }
    return result.sum();
}
