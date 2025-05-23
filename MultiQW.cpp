#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include "vcl2/vectorclass.h"
#include <math.h>
#include <complex>
#include <iostream>
#include <random>
#include <fstream>
#include <cstdio> // for sprintf

#define PI 3.1415926535897932384626

using namespace Eigen;

//Get least significant bit
unsigned int LSB(int n){
  return n & (-n);
}

//Convert value to grey code
unsigned int grey(unsigned int n){
  return n ^ (n >> 1);
}

//Get position of the only set bit
//Taken from http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
unsigned int log2(unsigned int v){
  static const int MultiplyDeBruijnBitPosition2[32] = 
  {
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
  };
  return MultiplyDeBruijnBitPosition2[(uint32_t)(v * 0x077CB531U) >> 27];
}


//Taken from https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
//and https://stackoverflow.com/questions/60472139/computing-the-inverse-of-the-complementary-error-function-erfcinv-in-c
//Credit goes to the author, njuffa, for this and the next function.
float my_erfcinvf (float a)
{
    float r;

    if ((a >= 2.1875e-3f) && (a <= 1.998125f)) { // max. ulp err. = 2.77667
        float p, t;
        t = fmaf (-a, a, a + a);
        t = logf (t);
        p =              5.43877832e-9f;  //  0x1.75c000p-28 
        p = fmaf (p, t,  1.43286059e-7f); //  0x1.33b458p-23 
        p = fmaf (p, t,  1.22775396e-6f); //  0x1.49929cp-20 
        p = fmaf (p, t,  1.12962631e-7f); //  0x1.e52bbap-24 
        p = fmaf (p, t, -5.61531961e-5f); // -0x1.d70c12p-15 
        p = fmaf (p, t, -1.47697705e-4f); // -0x1.35be9ap-13 
        p = fmaf (p, t,  2.31468701e-3f); //  0x1.2f6402p-9 
        p = fmaf (p, t,  1.15392562e-2f); //  0x1.7a1e4cp-7 
        p = fmaf (p, t, -2.32015476e-1f); // -0x1.db2aeep-3 
        t = fmaf (p, t,  8.86226892e-1f); //  0x1.c5bf88p-1 
        r = fmaf (t, -a, t);
    } else {
        float p, q, s, t;
        t = (a >= 1.0f) ? (2.0f - a) : a;
        t = 0.0f - logf (t);

        s = sqrtf (1.0f / t);
        p =              2.23100796e+1f;  //  0x1.64f616p+4
        p = fmaf (p, s, -5.23008537e+1f); // -0x1.a26826p+5
        p = fmaf (p, s,  5.44409714e+1f); //  0x1.b3871cp+5
        p = fmaf (p, s, -3.35030403e+1f); // -0x1.0c063ap+5
        p = fmaf (p, s,  1.38580027e+1f); //  0x1.bb74c2p+3
        p = fmaf (p, s, -4.37277269e+0f); // -0x1.17db82p+2
        p = fmaf (p, s,  1.53075826e+0f); //  0x1.87dfc6p+0
        p = fmaf (p, s,  2.97993328e-2f); //  0x1.e83b76p-6
        p = fmaf (p, s, -3.71997419e-4f); // -0x1.86114cp-12
        p = fmaf (p, s, s);
        r = 1.0f / p;
        if (a >= 1.0f) r = 0.0f - r;
    }
    return -r;
}

/* Compute inverse of the CDF of the standard normal distribution.
   max ulp err = 4.08385
*/
float my_normcdfinvf (float a)
{
    return fmaf (-1.41421356f, my_erfcinvf (a + a), 0.0f);
}

void H_G(float* psi, float* psi2, unsigned int n){
  //Calculates psi2 += H_G @ psi
  //Take into account that complex vectors are twice as long
  n += 1;
  unsigned int N = (1 << n);

  Vec16f a;
  Vec16f b;
  int h = 16;

  //Use permutations for h<16 cases
  for (int i = 0; i<N; i+=16){
    a.load(psi+i);
    b.load(psi2+i);
    //Uncomment next line if using real arrays rather than complex
    //b += permute16<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(a);

    b += permute16<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(a);
    b += permute16<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>(a);
    b += permute16<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(a);
    b.store(psi2+i);
  }

  //The vector can be kept in cache for these sizes
  //For now, this needs changing manually
  if (n > 4){
    int end = N < 1 << 18 ? N : 1 << 18;
    for (int i = 0; i<N; i+=16){
      b.load(psi2+i);
      for (h = 16; h < end; h*=2){
        int temp = h&i ? -h : h;
        a.load(psi + i + temp);
        b += a;
      }
    b.store(psi2+i);
    }
  }

  //Cache can't save us so we have to suffer
  if (n > 18){
    for (int i = 0; i<N; i+=16){
      b.load(psi2+i);
      for (h = 1 << 18; h < N; h*=2){
        int temp = h&i ? -h : h;
        a.load(psi + i + temp);
        b += a;
      }
    b.store(psi2+i);
    }
  }
  return;
}

ArrayXcf Cheb(unsigned int n, float scale){
  //Returns n'th order Chebyshev expansion of exp(i*x*scale) on [-1,1]
  //Uses an FFT to calculate expansions for sin and cos separately, and sum them
  const std::complex<float> I(0.0,1.0);
  Eigen::FFT<float> fft;
  VectorXcf c(2*n);
  VectorXcf s(2*n);

  VectorXcf res1(2*n);
  VectorXcf res2(2*n);

  VectorXcf out(n);
  for (int i = 0; i < n; i++){
    float temp = (2*i + 1)*PI/2/n;
    c[i] = cos(cos(temp)*scale);
    s[i] = sin(cos(temp)*scale);
    c[i + n] = 0;
    s[i + n] = 0;
  }
  fft.inv(res1,c);
  fft.inv(res2,s);

  for (int i = 0; i < n; i++){
    float temp = PI*i/2/n;
    out[i] = 4*real(exp(I*temp)*res1[i]) + 4.0f*I*real(exp(I*temp)*res2[i]);
  }
  out[0] /= 2.0f;
  return out;
}

ArrayXcf Clenshaw(VectorXcf coeffs, ArrayXcf psi, ArrayXf H_P, float gamma, float scale){
  //Calculate exp(-i*H*t)@psi using the Clenshaw algorithm to evaluate a polynomial approximation
  //Uses only matrix-vector multiplications
  int n = coeffs.size();
  ArrayXcf b1 = coeffs[n-1]*psi;
  std::complex<float>* b1p = b1.data();

  ArrayXcf b2 = b1*0;
  std::complex<float>* b2p = b2.data();

  for (int r = n-2; r > 0; r--){
    while (abs(coeffs[r]) < 1e-6) r--;
    //The performance of this section seems very sensitive to the order of operations
    //We do some trickery to calculate the next iteration in b2 without a temporary array
    b2 *= (0.5*scale/gamma);
    H_G((float*)b1p, (float*)b2p, log2(b1.size()));
    b2 = (2/scale)*(H_P*b1 - gamma*b2);
    b2 += psi*coeffs[r];

    //We essentially swap the labels b1 and b2 
    Eigen::ArrayXcf &ref1 = b1;
    Eigen::ArrayXcf &ref2 = b2;
    std::swap(ref1, ref2);
    b1p = b1.data();
    b2p = b2.data();
  }
  b2 *= (scale/gamma);
  H_G((float*)b1p, (float*)b2p, log2(b1.size()));
  b2 = (1/scale)*(H_P*b1 - gamma*b2) + psi*coeffs[0];

  return b2;
}

float heur[20] = {1.2082979794574937, 1.3131560483482256, 1.4100589067449547, 1.5006739957742985, 1.5861415036770354, 1.6672845086916428, 1.7447219000349252, 1.81893401086401, 1.8903031784517639, 1.9591400720339947, 2.025701482876633, 2.090202753583327, 2.1528267101788385, 2.2137302372981815, 2.2730492199794945, 2.330902325676296, 2.3873939451016604, 2.4426165114475493, 2.4966523525078355, 2.5495751865531977};

int main(int argc, char* argv[]){
  //Arguments are number of spins, number of walk stages, filename, start position and number of problems
  //Last two are to allow for easier multi-threading, just start the program multiple times with different starts
  if (argc < 4) return -1;
  unsigned int n = atoi(argv[1]);
  unsigned int N = 1 << n;
  unsigned int m = atoi(argv[2]);
  
  char* filename = argv[3];

  //Which entry to start and how many to do
  unsigned int start = 0;
  unsigned int reps = 2000;

  float* results = (float*)malloc(reps*4);
  float* gaps = (float*)malloc(reps*4);


  //Number of samples for Monte-Carlo integral
  int samples = 100;

  if (argc == 6){
    start = atoi(argv[4]);
    reps = atoi(argv[5]);
  }

  char output[20];
  std::sprintf(output, "output_%d_%d_%d", n, m, start);
  std::ofstream outFile(output, std::ios::binary | std::ios::app);

  std::ifstream file(filename, std::ios::binary);

  //Seek to beginning, each problem has (n-1)*n parameters, and a float has 4 bytes.
  file.seekg(4*(n+1)*n*start, file.beg);

  Eigen::ArrayXXf times(m,samples);
  Eigen::ArrayXf probs(samples);

  ArrayXf gammas(m);
  ArrayXf onenorms(m);

  //Problem energy levels
  ArrayXf H_P(N);

  //This is used for calculating H_P, state(i,j) = sigma_z_i * sigma_z_j
  ArrayXXf state(n,n);

  //Our quantum register
  ArrayXcf psi(N);

  //Ising problem parameters
  ArrayXXf J(n,n);
  ArrayXf h(n);

  //Used for reading in parameters
  char buffer[4*n*(n+1)];
  double temp[n*(n+1)/2];

  for (int rep = 0; rep < reps; rep++){

    //We care about both the ground state and the energy gap
    double E_0 = 0;
    double E_1 = 0;
    double E_max = 0;
    unsigned int E_loc = 0;

    //Read next set of parameters
    file.read(buffer, 4*n*(n+1));
    std::memcpy(temp, buffer, 4*n*(n+1));

    J.setConstant(0);

    //Load J matrix
    int k = 0;
    for (int i = 1; i < n; i++){
      for (int j = 0; j < i; j++){
        J(i,j) = temp[k];
        k++;
      }
    }

    //Absorb 1/2 factor into J to follow paper
    J += J.transpose().eval();
    J /= 2;

    for (int i = 0; i < n; i++){
      h(i) = temp[n*(n-1)/2 + i];
    }

    state.setConstant(1);
    psi.setConstant(1/sqrt(N));

    //The way we calculate E is prone to error so use a double
    //Start with the energy of the all -1s state
    double E = J.sum() - h.sum();
    H_P[0] = E;

    //In case first state is ground
    E_0 = E;
    E_max = E;
  
    //Use a grey code to efficiently evaluate all energies
    for (unsigned int i = 1; i < N; i++){
      unsigned int flip = log2(LSB(i));
      state.row(flip) *= -1;
      state.col(flip) *= -1;
      state(flip,flip) *= -1;
      E += 4*(J.row(flip)*state.row(flip)).sum() - 2*h(flip)*state(flip,flip);
      H_P[grey(i)] = E;

      //keep track of ground state
      if (E < E_0){
        E_1 = E_0;
        E_0 = E;
        E_loc = grey(i);
      }
      else if (E < E_1){
        E_1 = E;
      }

      //keep track of highest state too
      if (E > E_max){
        E_max = E;
      }
    }
    //We want to shift H_P to reduce the spectral radius
    //float kurt = (H_P*H_P*H_P*H_P).mean() / pow((H_P*H_P).mean(),2);
    //float skew = (H_P*H_P*H_P).mean() / pow((H_P*H_P).mean(),1.5);
    H_P -= (E_max + E_0)/2;
    float E_abs = (E_max - E_0)/2;
    //Now H_P has been calculated

    // This formula is from https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables/89147#89147
    //Calculates estimated maximum energy level using the known variance and assuming a normal distribution
    float b = my_normcdfinvf(1/(float)N);
    float e_m = 0.577215664901532860;
    float e = 2.718281828459045;
    float a = (1 - e_m)*b + e_m*my_normcdfinvf(1/(e*N));

    //If the h was 0, we'd just want HP2 = 2*(J*J).sum()
    //If we were to map to an n+1 qubit problem we'd get h/2 in both a row and a column
    //so we need to add 2*2*(h/2).dot(h/2) = h.dot(h) 
    float HP2 = 2*(J*J).sum() + (h*h).sum();

    //std::cout << (E_abs - a*sqrt(HP2))/E_abs << " " << kurt << " " << sqrt(HP2)*(my_normcdfinvf(1/(e*N)) - b)*sqrt(PI*PI/6)/E_abs << "\n";
    std::cout << HP2 << " " << E_0 << "\n";
  
    //Calculate walk time

    //Similar to above, we want 8*(J*J).sum(), but having h != 0 changes the dynamics
    //We have to derive a new formula that works for h != 0, as mapping to n+1 qubits also changes the dynamics
    //The new formula is 8*(J*J).sum() + 2*h.h
    float short_t = 8*(J*J).sum() + 2*(h*h).sum();
    short_t = sqrt(2*n/short_t);

    //Should really use higher quality RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rand_t(short_t, 2*short_t);

    //Calculate gammas, upper bounds on spectral radius and generate evolution times.
    for (int i = 0; i<m; i++){
      gammas[i] = a*sqrt(HP2)/tan(PI*(i+1)/(2*m + 2))/n;

      //Old heuristic
      //gammas[i] = heur[n - 5]/tan(PI*(i+1)/(2*m + 2));

      for(int j = 0; j < samples; j++) {
        times(i,j) = rand_t(gen);
      }

    }

    //Optimisation for single-stage
    if (m == 1){
      std::sort(times.row(0).begin(), times.row(0).end());
      //Have to calculate backwards to avoid modifying data that's still needed
      times(0,seq(last,1,-1)) -= times(0,seq(last-1,0,-1));
    }

    for (int j = 0; j < samples; j++){
      //Uncomment this section to randomly sample gamma values
      /*for (int i = 0; i<m; i++){
        gammas[i] = heur[n - 5]/tan(rand_g(gen));
      }
      std::sort(gammas.begin(), gammas.end(), std::greater<float>());
      */

      //Loop through all the times and calculate the success probability
      for (int i = 0; i < m; i++){
        float gamma = gammas[i];
        float onenorm = (E_abs + gamma*n);
        //H_P + gamma*H_G is conserved, then a change in H_G of 2n/gamma causes a change in H_P of -2n
        //So for stages with large gamma, H_P is predicted to saturate first.
	if (1.0/tan(PI*(i+1)/(2*m + 2)) > 1) times(i,j) /= sqrt(1.0/tan(PI*(i+1)/(2*m + 2)));
        int terms = 32 + (int)onenorm*times(i,j);
        ArrayXcf coeffs = Cheb(terms,onenorm*times(i,j));
        while (abs(coeffs[terms - 1]) > 1e-6){
          terms*=2;
          coeffs = Cheb(terms, onenorm*times(i,j));
        }

        psi = Clenshaw(coeffs, psi, H_P, gamma, onenorm);
      }
      probs(j) = abs(psi[E_loc])*abs(psi[E_loc]);
      if (m != 1) psi.setConstant(1/sqrt(N));

    }
  results[rep] = probs.sum()/samples;
  std::cout << results[rep] << "\n\n";
  //gaps[rep] = E_1 - E_0;
  }
  outFile.write(reinterpret_cast<const char*>(results), reps * sizeof(float));
  //outFile.write(reinterpret_cast<const char*>(gaps), reps * sizeof(float));
}