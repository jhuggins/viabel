// weighted regression model used to estimate constants that required to 
// compute error between for a given learning-rate variational approximation
// and the optimal variational approximation for Mean-field Gaussian family
// with stochastic gradient descent (SGD) or modified adaptive SGD.
data {
   int<lower=0> N;
   vector[N] y; // log(SKL)
   vector[N] x; // log(\gamma) 
   real<lower=0> rho;
   vector[N] w; //weights
}

parameters {
    real log_c; // log(c)
    real<lower=0> sigma; // \sigma
}

//transformed parameters {
 //   real<lower=0> c=exp(log_c);
//}

model {
    real mu;
    
    log_c ~ cauchy(0,10);
    sigma ~ cauchy(0,10);

    for (n in 1:N) {
        mu = log_c + 2*log((1/rho)-1) + 2*x[n];
        target += normal_lpdf(y[n] | mu, sigma) * w[n];  
    }     
}