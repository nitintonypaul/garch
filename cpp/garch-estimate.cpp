//Using math defines to utilize PI
//Including modules
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

//Namespace for pybind11
namespace py = pybind11;

//Function declarations
double estimate_volatility(int no_of_days, double past_vol, const std::vector<double> &shock_array);
double L(double a, double b, double w, const std::vector<double> &arr);

//Global Variables to be accessible anywhere
int days;
double vol;
double expected_volatility;

//Estimating volatility
double estimate_volatility(int no_of_days, double past_vol, const std::vector<double> &shock_array) {

    //Arguments to be recieved
    days = no_of_days;
    vol = past_vol;

    //Initial parameter Guesses
    double w = 0.000001;
    double a = 0.1;
    double b = 0.8;

    //Step values
    double learning_rate = 0.001;
    double step = 0.00001;

    //Optimisation loop
    for (int i = 0; i < 1000; i++) {

        //Computing base likelihood
        double likelihood = L(a, b, w,shock_array);

        //Computing shifted likelihoods
        double L_a = L(a+step, b, w, shock_array);
        double L_b = L(a, b+step, w, shock_array);
        double L_w = L(a, b, w+step, shock_array);

        //Obtaining compared result for readibility (1 for true, 0 for false)
        int a_shift = L_a > likelihood;
        int b_shift = L_b > likelihood;
        int w_shift = L_w > likelihood;

        //Computing partial derivatives
        double da = (L_a - likelihood) / step;
        double db = (L_b - likelihood) / step;
        double dw = (L_w - likelihood) / step;

        //Computing potential shift
        double potential_shift_a = learning_rate * da;
        double potential_shift_b = learning_rate * db;
        double potential_shift_w = learning_rate * dw;

        //Optimizing alpha with contraint
        if (a_shift && a+potential_shift_a >= 0) {
            a += potential_shift_a;
        }
        else if (!a_shift && a+potential_shift_a >= 0){
            a -= potential_shift_a;
        }

        //Optimizing beta with contraint
        if (b_shift && b+potential_shift_b >= 0) {
            b += potential_shift_b;
        }
        else if (!b_shift && b+potential_shift_b >= 0){
            b -= potential_shift_b;
        }

        //Optimizing Omega with contraint
        if (w_shift && w+potential_shift_w >= 0) {
            w += potential_shift_w;
        }
        else if (!w_shift && w+potential_shift_w >= 0){
            w -= potential_shift_w;
        }

        //Scaling constraint (alpha + beta must be less than 1)
        if (a + b >= 1) {

            //Scaling alpha and beta proportionally
            double scale = 0.99 / (a + b);
            a *= scale;
            b *= scale;
        }

        //Upper bound parameter for omega to avoid drifting off to INFINITY
        if (w > 0.00001) w = 0.00001;

        //Convergence check (To break if result is minimal and uneffective)
        if (abs(da) + abs(db) + abs(dw) < 1e-8) break;
    }

    //Running optimization once more to obtain expected volatility
    L(a, b, w, shock_array);

    //Returning expected volatility
    return expected_volatility;
}

//Logarithmic Likelihood function
double L(double a, double b, double w, const std::vector<double> &arr) {
    
    //Assigning base values
    double prev_variance = vol*vol;
    double total_likelihood = 0.000;

    //Tiny epsilon (Adds to logarithm to prevent 0)
    double tiny_eps = 0.0000000001;

    //Computing total likelhood
    for (int i = 1; i < days; i++) {
        double sig_t_square = w + (a * std::pow(arr[i-1],2)) + (b * prev_variance);
        total_likelihood += -0.5 * (std::log(2 * M_PI) + std::log(sig_t_square + tiny_eps) + (std::pow(arr[i], 2) / sig_t_square));
        prev_variance = sig_t_square;
    }

    //Changing global variable to expected variance for easier computation
    expected_volatility = std::sqrt(prev_variance);

    //Returning total likelihood
    return total_likelihood;
}


//Pybind11 declaration
PYBIND11_MODULE(garch_est, m) {
    m.def("estimate_vol", &estimate_volatility);
}