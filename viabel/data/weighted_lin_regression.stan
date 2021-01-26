data {
   int<lower=0> N;
   vector[N] y; // log(SKL)
   vector[N] x; // log(\gamma) 
   real<lower=0> rho;
   vector[N] w; //weights
}

parameters {
    real<lower=0,upper=1>  kappa; //power
    real log_c; // log(c)
    real<lower=0> sigma; // \sigma
}

model {
    real mu;
        
    kappa ~ uniform(0,1);
    log_c ~ cauchy(0,10);
    sigma ~ cauchy(0,10);

    for (n in 1:N) {
        mu = log_c + 2*log((1/rho^kappa)-1) + 2*kappa*x[n];
        target += normal_lpdf(y[n] | mu, sigma) * w[n];  
    }     
}