#include <iostream>
using std::cin;
using std::cout;
using std::endl;
using std::ios;
#include <string>
using std::string;
#include <stdexcept>
using std::runtime_error;
#include <fstream>
using std::ifstream;
#include "NeuralNetwork.h"

typedef unsigned char uchar;
uchar** readMNISTImages(string full_path, int &number_of_images, int &image_size)
{
    auto reverseInt = [] (int i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open())
    {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++)
        {
            _dataset[i] = new uchar[image_size];
            file.read((char*)_dataset[i], image_size);
        }
        return _dataset;
    }
    else
    {
        throw runtime_error("Cannot open file: '" + full_path + "'!");
    }
}
uchar*  readMNISTLabels(string full_path, int &number_of_labels)
{
    auto reverseInt = [](int i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open())
    {
        int magic_number = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *) &number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar *_dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++)
        {
            file.read((char *)&_dataset[i], 1);
        }
        return _dataset;
    }
    else
    {
        throw runtime_error("Unable to open file '" + full_path + "'!");
    }
}

int main()
{
    int train_n_img = 0, train_img_size = 0, train_n_labels = 0;
    int test_n_img = 0, test_img_size = 0, test_n_labels = 0;

    string path("path\\to\\MNIST\\files");
    string train_img_file(path + "train-images.idx3-ubyte");
    string train_lbl_file(path + "train-labels.idx1-ubyte");

    string test_img_file(path + "t10k-images.idx3-ubyte");
    string test_lbl_file(path + "t10k-labels.idx1-ubyte");

    uchar **train_images;
    uchar *train_labels;
    uchar **test_images;
    uchar *test_labels;

    try
    {
        train_images = readMNISTImages(train_img_file, train_n_img, train_img_size);
        test_images = readMNISTImages(test_img_file, test_n_img, test_img_size);
        train_labels = readMNISTLabels(train_lbl_file, train_n_labels);
        test_labels = readMNISTLabels(test_lbl_file, test_n_labels);
        cout << "Train images read : " << train_n_img << endl;
        cout << "Train image size  : " << train_img_size << endl;
        cout << "Train labels read : " << train_n_labels << endl;
        cout << "Test images read  : " << test_n_img << endl;
        cout << "Test image size   : " << test_img_size << endl;
        cout << "Test labels read  : " << test_n_labels << endl;
    }
    catch(runtime_error &e)
    {
        cout << "Runtime error : " << e.what() << endl;
        return -1;
    }

    int data_size = 0;
    cout << "Set train data size : ";
    cin >> data_size;

    train_n_img = data_size;
    train_n_labels = data_size;

    net_multi_array train_input(train_n_img);
    net_multi_array train_output(train_n_labels);

    net_multi_array test_input(test_n_img);
    net_multi_array test_output(test_n_labels);

    int i, j;

    cout << "Setting data..." << endl;
    for(i = 0; i < train_n_img; i++)
    {
        train_input[i].resize(train_img_size);
        for(j = 0; j < train_img_size; j++)
        {
            //train_input[i][j] = (((int)train_images[i][j] / 127.5) - 1.0) * (-1.0);
            train_input[i][j] = ((int)train_images[i][j] / 127.5) / (2.0);
        }
        train_output[i].resize(10);
        for(j = 0; j < 10; j++)
        {
            train_output[i][j] = (j == (int)train_labels[i]) ? 1.0 : 0.0;
        }
    }
    for(i = 0; i < test_n_img; i++)
    {
        test_input[i].resize(test_img_size);
        for(j = 0; j < test_img_size; j++)
        {
            //test_input[i][j] = (((int)test_images[i][j] / 127.5) - 1.0) * (-1.0);
            test_input[i][j] = ((int)test_images[i][j] / 127.5) / (2.0);
        }
        test_output[i].resize(10);
        for(j = 0; j < 10; j++)
        {
            test_output[i][j] = (j == (int)test_labels[i]) ? 1.0 : 0.0;
        }
    }
    cout << "Data formated." << endl;
    delete [] train_images;
    delete [] train_labels;
    delete [] test_images;
    delete [] test_labels;

    /** Put data into tuples */

    net_tuple training_data(train_input, train_output, train_n_img);
    net_tuple test_data(test_input, test_output, test_n_img);

    /**  */
    int epochs, mini_batch_size, n_threads;
    double eta, lambda = 0.0;
    int option = 0;
    string shape;
    string cost;
    string activation;

    string net_path;

    NeuralNetwork net;

    cout << "Enter an option (1 = New Network; 2 = Load a Network; 3 = Multi thread Network; 4 = Close) -> ";
    cin >> option;
    cin.seekg(0, ios::end);
    cin.clear();

    while((option >= 1) && (option <= 3))
    {
        switch(option)
        {
            case 1:
                cout << "Enter Network parameters:" << endl;
                cout << "-> Shape: ";
                getline(cin, shape);
                cout << "-> Cost function: ";
                getline(cin, cost);
                cout << "-> Activation: ";
                getline(cin, activation);
                cout << "-> Epochs: ";
                cin >> epochs;
                cout << "-> Mini batch size: ";
                cin >> mini_batch_size;
                cout << "-> L. R. Eta: ";
                cin >> eta;
                cout << "-> Lambda: ";
                cin >> lambda;
                cout << "Start training." << endl;
                net.setParameters(shape, cost, activation);
                net.SGD(training_data, epochs, mini_batch_size, eta, lambda, test_data);
                break;
            case 2:
                cout << "Enter path to Network parameter: ";
                getline(cin, net_path);
                cout << endl;
                try
                {
                    net.loadNetwork(net_path, epochs, mini_batch_size, eta, lambda);
                    cout << "Start training." << endl;
                    net.SGD(training_data, epochs, mini_batch_size, eta, lambda, test_data);
                }
                catch(runtime_error &e)
                {
                    cout << "Error loading Network -> " << e.what() << endl;
                }
                break;
            case 3:
                cout << "Enter Network parameters:" << endl;
                cout << "-> Shape: ";
                getline(cin, shape);
                cout << "-> Cost function: ";
                getline(cin, cost);
                cout << "-> Activation: ";
                getline(cin, activation);
                cout << "-> Epochs: ";
                cin >> epochs;
                cout << "-> Mini batch size: ";
                cin >> mini_batch_size;
                cout << "-> L. R. Eta: ";
                cin >> eta;
                cout << "-> Lambda: ";
                cin >> lambda;
                cout << "-> Threads: ";
                cin >> n_threads;
                cout << "Start training." << endl;
                net.setParameters(shape, cost, activation);
                net.multiSGD(training_data, epochs, mini_batch_size, eta, lambda, test_data, n_threads);
            default:
                break;
        }
        cout << "Enter an option (1 = New Network; 2 = Load a Network; 3 = Multi thread Network; 4 = Close) -> ";
        cin >> option;
    }

    cout << "Finish!" << endl;

    return 0;
}
