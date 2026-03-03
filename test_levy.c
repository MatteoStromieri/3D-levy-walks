#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "func.h"

int main(){
    
    double mu[] = {1.0};//{1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0};
    int side = 512;
    int lmax = side / 2;
    int n_mu = sizeof(mu)/sizeof(mu[0]);
    double norm[n_mu];
    double average[n_mu];
    for (int i = 0; i < n_mu; ++i) {
        norm[i] = get_normalization_constant(mu[i], lmax);
        printf("mu = %.1f, norm = %.5f\n", mu[i], norm[i]);
    }

    int n_iters = 100000;
    for (int i = 0; i < n_mu; ++i) {
        average[i] = 0.0;
        for (int j = 0; j < n_iters; ++j) {
            double val = Levy(mu[i], lmax, norm[i]);
            //printf("Value = %.5f\n", val);
            average[i] += val;
        }
        average[i] /= n_iters;
    }

    printf("Computing theoretical average...\n");
    double theoretical_average[n_mu];
    for (int i = 0; i < n_mu; ++i) {
        if (mu[i] == 2.0) {
            theoretical_average[i] = norm[i]/2 + norm[i]*log((double)lmax);
        } else {
            theoretical_average[i] = norm[i]/2 + norm[i]/(2-mu[i])*(pow((double)lmax, 2-mu[i]) - 1);;
        }
    }

    printf("Empirical verage values for mu:\n");
    for (int i = 0; i < n_mu; ++i) {
        printf("mu = %.1f: Emp. average = %.5f | Th. average = %.5f\n", mu[i], average[i], theoretical_average[i]);
    }

    for( int i = 0; i < 20; ++i){
        printf("Levy %d: %0.1f \n",i, Levy(3.0, lmax, norm[10]));
    }
    
    return 0;
}