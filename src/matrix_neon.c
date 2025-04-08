#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "topological_entropy.h"

// NEON-optimized matrix-vector multiplication for improved performance
// on ARM processors (e.g., Apple M1/M2, Raspberry Pi, etc.)

// Define NEON availability based on compiler flags and architecture
#if defined(DISABLE_NEON)
    // NEON explicitly disabled via compiler flag
    #define HAS_NEON 0
#elif defined(USE_NEON)
    // NEON explicitly enabled via compiler flag
    #if defined(__ARM_NEON) || defined(__ARM_NEON__) || (defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__)))
        #include <arm_neon.h>
        #define HAS_NEON 1
    #else
        #warning "NEON requested but not available on this architecture"
        #define HAS_NEON 0
    #endif
#elif defined(USE_NEON_IF_AVAILABLE)
    // Runtime detection (decide at runtime)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__) || (defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__)))
        #include <arm_neon.h>
        #define HAS_NEON 1
    #else
        #define HAS_NEON 0
    #endif
#else
    // Default behavior (no specific flag set)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__) || (defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__)))
        #include <arm_neon.h>
        #define HAS_NEON 1
    #else
        #define HAS_NEON 0
    #endif
#endif

// Runtime check for NEON capabilities
int check_neon_available() {
#if defined(HAS_NEON) && HAS_NEON
    // On Apple Silicon (M1/M2/etc), NEON is always available
    #if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
        return 1;  // Apple Silicon - NEON is always available
    #else
        // For other ARM processors, we rely on the compile-time detection
        return 1;  // NEON is compiled in and available
    #endif
#else
    // Check if we're on Apple Silicon even if not detected at compile time
    #if defined(__APPLE__)
        // Try to detect Apple Silicon at runtime
        #if defined(__arm64__) || defined(__aarch64__)
            return 1;  // Apple Silicon detected at runtime
        #endif
    #endif
    return 0;  // NEON is not available
#endif
}

// Matrix-vector multiplication with appropriate implementation
void matrix_vector_multiply_neon(double _Complex *matrix, double _Complex *vector, 
                                double _Complex *result, int size) {
#if HAS_NEON
    // NEON-optimized implementation for ARM processors
    for (int i = 0; i < size; i++) {
        float64x2_t sum_real = vdupq_n_f64(0.0);
        float64x2_t sum_imag = vdupq_n_f64(0.0);
        
        // Process blocks of 2 complex numbers
        for (int j = 0; j < size - 1; j += 2) {
            // Load matrix elements (2 complex numbers)
            double real1 = creal(matrix[i * size + j]);
            double imag1 = cimag(matrix[i * size + j]);
            double real2 = creal(matrix[i * size + j + 1]);
            double imag2 = cimag(matrix[i * size + j + 1]);
            
            // Load vector elements (2 complex numbers)
            double vec_real1 = creal(vector[j]);
            double vec_imag1 = cimag(vector[j]);
            double vec_real2 = creal(vector[j + 1]);
            double vec_imag2 = cimag(vector[j + 1]);
            
            // Create NEON vectors
            float64x2_t mat_real = vsetq_lane_f64(real1, vdupq_n_f64(0.0), 0);
            mat_real = vsetq_lane_f64(real2, mat_real, 1);
            
            float64x2_t mat_imag = vsetq_lane_f64(imag1, vdupq_n_f64(0.0), 0);
            mat_imag = vsetq_lane_f64(imag2, mat_imag, 1);
            
            float64x2_t vec_real = vsetq_lane_f64(vec_real1, vdupq_n_f64(0.0), 0);
            vec_real = vsetq_lane_f64(vec_real2, vec_real, 1);
            
            float64x2_t vec_imag = vsetq_lane_f64(vec_imag1, vdupq_n_f64(0.0), 0);
            vec_imag = vsetq_lane_f64(vec_imag2, vec_imag, 1);
            
            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            // Real part: ac - bd
            float64x2_t temp1 = vmulq_f64(mat_real, vec_real);
            float64x2_t temp2 = vmulq_f64(mat_imag, vec_imag);
            float64x2_t real_part = vsubq_f64(temp1, temp2);
            
            // Imaginary part: ad + bc
            float64x2_t temp3 = vmulq_f64(mat_real, vec_imag);
            float64x2_t temp4 = vmulq_f64(mat_imag, vec_real);
            float64x2_t imag_part = vaddq_f64(temp3, temp4);
            
            // Accumulate
            sum_real = vaddq_f64(sum_real, real_part);
            sum_imag = vaddq_f64(sum_imag, imag_part);
        }
        
        // Extract and sum the results
        double real_sum = vgetq_lane_f64(sum_real, 0) + vgetq_lane_f64(sum_real, 1);
        double imag_sum = vgetq_lane_f64(sum_imag, 0) + vgetq_lane_f64(sum_imag, 1);
        
        // Handle leftover element if size is odd
        if (size % 2 != 0) {
            int j = size - 1;
            double real = creal(matrix[i * size + j]);
            double imag = cimag(matrix[i * size + j]);
            double vec_real = creal(vector[j]);
            double vec_imag = cimag(vector[j]);
            
            // Complex multiplication
            real_sum += real * vec_real - imag * vec_imag;
            imag_sum += real * vec_imag + imag * vec_real;
        }
        
        result[i] = real_sum + imag_sum * I;
    }
#else
    // Standard implementation for non-ARM processors
    for (int i = 0; i < size; i++) {
        double _Complex sum = 0.0;
        for (int j = 0; j < size; j++) {
            sum += matrix[i * size + j] * vector[j];
        }
        result[i] = sum;
    }
#endif
}

// Calculate eigenvalues with appropriate implementation
void calculate_eigenvalues_neon(double _Complex *matrix, double *eigenvalues, int size) {
#if HAS_NEON
    // NEON-optimized implementation
    // Allocate memory for work matrices and vectors
    double _Complex *work_matrix = (double _Complex *)malloc(size * size * sizeof(double _Complex));
    double _Complex *eigenvector = (double _Complex *)malloc(size * sizeof(double _Complex));
    double _Complex *temp_vector = (double _Complex *)malloc(size * sizeof(double _Complex));
    
    if (!work_matrix || !eigenvector || !temp_vector) {
        fprintf(stderr, "Error: Memory allocation failed in eigenvalue calculation\n");
        if (work_matrix) free(work_matrix);
        if (eigenvector) free(eigenvector);
        if (temp_vector) free(temp_vector);
        
        // Set eigenvalues to equal probabilities as fallback
        for (int i = 0; i < size; i++) {
            eigenvalues[i] = 1.0 / size;
        }
        return;
    }
    
    // Copy the original matrix to work matrix
    for (int i = 0; i < size * size; i++) {
        work_matrix[i] = matrix[i];
    }
    
    // Initialize eigenvalues array
    for (int i = 0; i < size; i++) {
        eigenvalues[i] = 0.0;
    }
    
    // Number of largest eigenvalues to extract (limited by matrix size)
    int num_eigen = (size > 16) ? 16 : size;
    
    // Extract eigenvalues using deflation and NEON-accelerated matrix operations
    for (int k = 0; k < num_eigen; k++) {
        // Initialize eigenvector with random values
        double norm = 0.0;
        for (int i = 0; i < size; i++) {
            double real = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double imag = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            eigenvector[i] = real + imag * _Complex_I;
            norm += cabs(eigenvector[i]) * cabs(eigenvector[i]);
        }
        
        // Normalize the initial vector
        norm = sqrt(norm);
        for (int i = 0; i < size; i++) {
            eigenvector[i] /= norm;
        }
        
        // Power iteration to find dominant eigenvalue and eigenvector
        double lambda = 0.0;
        double prev_lambda = -1.0;
        int max_iter = 100;
        double tolerance = 1e-6;
        
        for (int iter = 0; iter < max_iter && fabs(lambda - prev_lambda) > tolerance; iter++) {
            prev_lambda = lambda;
            
            // Use NEON-optimized matrix-vector multiplication
            matrix_vector_multiply_neon(work_matrix, eigenvector, temp_vector, size);
            
            // Calculate the Rayleigh quotient (eigenvalue estimate) using NEON
            double _Complex rq_num = 0.0;
            double _Complex rq_denom = 0.0;
            
            // Process 2 elements at a time using NEON
            float64x2_t num_real_sum = vdupq_n_f64(0.0);
            float64x2_t num_imag_sum = vdupq_n_f64(0.0);
            float64x2_t denom_sum = vdupq_n_f64(0.0);
            
            for (int i = 0; i < size - 1; i += 2) {
                // Load eigenvector pairs
                double eigen_real1 = creal(eigenvector[i]);
                double eigen_imag1 = cimag(eigenvector[i]);
                double eigen_real2 = creal(eigenvector[i+1]);
                double eigen_imag2 = cimag(eigenvector[i+1]);
                
                // Load temp_vector pairs
                double temp_real1 = creal(temp_vector[i]);
                double temp_imag1 = cimag(temp_vector[i]);
                double temp_real2 = creal(temp_vector[i+1]);
                double temp_imag2 = cimag(temp_vector[i+1]);
                
                // Create NEON vectors
                float64x2_t eigen_real = vsetq_lane_f64(eigen_real1, vdupq_n_f64(0.0), 0);
                eigen_real = vsetq_lane_f64(eigen_real2, eigen_real, 1);
                
                float64x2_t eigen_imag = vsetq_lane_f64(eigen_imag1, vdupq_n_f64(0.0), 0);
                eigen_imag = vsetq_lane_f64(eigen_imag2, eigen_imag, 1);
                
                float64x2_t temp_real = vsetq_lane_f64(temp_real1, vdupq_n_f64(0.0), 0);
                temp_real = vsetq_lane_f64(temp_real2, temp_real, 1);
                
                float64x2_t temp_imag = vsetq_lane_f64(temp_imag1, vdupq_n_f64(0.0), 0);
                temp_imag = vsetq_lane_f64(temp_imag2, temp_imag, 1);
                
                // Complex conjugate of eigenvector
                float64x2_t conj_eigen_real = eigen_real;
                float64x2_t conj_eigen_imag = vnegq_f64(eigen_imag);
                
                // Numerator: conj(eigen) * temp
                // Real part: conj_real * temp_real + conj_imag * temp_imag
                float64x2_t num_real = vmulq_f64(conj_eigen_real, temp_real);
                float64x2_t num_imag_temp = vmulq_f64(conj_eigen_imag, temp_imag);
                num_real = vaddq_f64(num_real, num_imag_temp);
                
                // Imaginary part: conj_real * temp_imag - conj_imag * temp_real
                float64x2_t num_imag = vmulq_f64(conj_eigen_real, temp_imag);
                float64x2_t num_real_temp = vmulq_f64(conj_eigen_imag, temp_real);
                num_imag = vsubq_f64(num_imag, num_real_temp);
                
                // Denominator: conj(eigen) * eigen = |eigen|^2 (real only)
                float64x2_t denom_real = vmulq_f64(eigen_real, eigen_real);
                float64x2_t denom_imag = vmulq_f64(eigen_imag, eigen_imag);
                float64x2_t denom = vaddq_f64(denom_real, denom_imag);
                
                // Accumulate
                num_real_sum = vaddq_f64(num_real_sum, num_real);
                num_imag_sum = vaddq_f64(num_imag_sum, num_imag);
                denom_sum = vaddq_f64(denom_sum, denom);
            }
            
            // Extract and sum the results
            double num_real = vgetq_lane_f64(num_real_sum, 0) + vgetq_lane_f64(num_real_sum, 1);
            double num_imag = vgetq_lane_f64(num_imag_sum, 0) + vgetq_lane_f64(num_imag_sum, 1);
            double denom = vgetq_lane_f64(denom_sum, 0) + vgetq_lane_f64(denom_sum, 1);
            
            // Handle leftover element if size is odd
            if (size % 2 != 0) {
                int i = size - 1;
                double eigen_real = creal(eigenvector[i]);
                double eigen_imag = cimag(eigenvector[i]);
                double temp_real = creal(temp_vector[i]);
                double temp_imag = cimag(temp_vector[i]);
                
                // Complex conjugate of eigenvector
                double conj_eigen_real = eigen_real;
                double conj_eigen_imag = -eigen_imag;
                
                // Numerator: conj(eigen) * temp
                num_real += conj_eigen_real * temp_real + conj_eigen_imag * temp_imag;
                num_imag += conj_eigen_real * temp_imag - conj_eigen_imag * temp_real;
                
                // Denominator: |eigen|^2
                denom += eigen_real * eigen_real + eigen_imag * eigen_imag;
            }
            
            // Combine real and imaginary parts
            rq_num = num_real + num_imag * _Complex_I;
            rq_denom = denom;
            
            // Extract eigenvalue (should be nearly real for Hermitian matrices)
            lambda = creal(rq_num / rq_denom);
            
            // Normalize the new eigenvector with NEON
            norm = 0.0;
            
            // Copy temp_vector to eigenvector and calculate norm
            float64x2_t norm_sum = vdupq_n_f64(0.0);
            
            for (int i = 0; i < size - 1; i += 2) {
                // Load two complex numbers from temp_vector
                double real1 = creal(temp_vector[i]);
                double imag1 = cimag(temp_vector[i]);
                double real2 = creal(temp_vector[i+1]);
                double imag2 = cimag(temp_vector[i+1]);
                
                // Create NEON vectors
                float64x2_t real_vec = vsetq_lane_f64(real1, vdupq_n_f64(0.0), 0);
                real_vec = vsetq_lane_f64(real2, real_vec, 1);
                
                float64x2_t imag_vec = vsetq_lane_f64(imag1, vdupq_n_f64(0.0), 0);
                imag_vec = vsetq_lane_f64(imag2, imag_vec, 1);
                
                // Calculate |z|^2 = real^2 + imag^2 for each complex number
                float64x2_t real_squared = vmulq_f64(real_vec, real_vec);
                float64x2_t imag_squared = vmulq_f64(imag_vec, imag_vec);
                float64x2_t abs_squared = vaddq_f64(real_squared, imag_squared);
                
                // Accumulate norm
                norm_sum = vaddq_f64(norm_sum, abs_squared);
                
                // Copy to eigenvector
                eigenvector[i] = temp_vector[i];
                eigenvector[i+1] = temp_vector[i+1];
            }
            
            // Handle odd size case
            if (size % 2 != 0) {
                int i = size - 1;
                eigenvector[i] = temp_vector[i];
                double real = creal(temp_vector[i]);
                double imag = cimag(temp_vector[i]);
                norm += real * real + imag * imag;
            }
            
            // Finalize norm calculation
            norm += vgetq_lane_f64(norm_sum, 0) + vgetq_lane_f64(norm_sum, 1);
            norm = sqrt(norm);
            
            if (norm < 1e-10) break;  // Avoid division by near-zero
            
            // Normalize eigenvector
            double norm_inv = 1.0 / norm;
            float64x2_t norm_inv_vec = vdupq_n_f64(norm_inv);
            
            for (int i = 0; i < size - 1; i += 2) {
                // Load two complex numbers
                double real1 = creal(eigenvector[i]);
                double imag1 = cimag(eigenvector[i]);
                double real2 = creal(eigenvector[i+1]);
                double imag2 = cimag(eigenvector[i+1]);
                
                // Create NEON vectors
                float64x2_t real_vec = vsetq_lane_f64(real1, vdupq_n_f64(0.0), 0);
                real_vec = vsetq_lane_f64(real2, real_vec, 1);
                
                float64x2_t imag_vec = vsetq_lane_f64(imag1, vdupq_n_f64(0.0), 0);
                imag_vec = vsetq_lane_f64(imag2, imag_vec, 1);
                
                // Normalize
                float64x2_t norm_real = vmulq_f64(real_vec, norm_inv_vec);
                float64x2_t norm_imag = vmulq_f64(imag_vec, norm_inv_vec);
                
                // Store back normalized values
                eigenvector[i] = vgetq_lane_f64(norm_real, 0) + vgetq_lane_f64(norm_imag, 0) * _Complex_I;
                eigenvector[i+1] = vgetq_lane_f64(norm_real, 1) + vgetq_lane_f64(norm_imag, 1) * _Complex_I;
            }
            
            // Handle odd size case
            if (size % 2 != 0) {
                int i = size - 1;
                eigenvector[i] /= norm;
            }
        }
        
        // Store the eigenvalue
        eigenvalues[k] = lambda;
        
        // Matrix deflation: Remove the extracted component from the matrix
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                work_matrix[i * size + j] -= lambda * eigenvector[i] * conj(eigenvector[j]);
            }
        }
    }
    
    // Ensure eigenvalues sum to 1 (for density matrices)
    double sum = 0.0;
    for (int i = 0; i < num_eigen; i++) {
        sum += eigenvalues[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < num_eigen; i++) {
            eigenvalues[i] /= sum;
        }
    } else {
        // Fallback to equal eigenvalues if calculation fails
        for (int i = 0; i < size; i++) {
            eigenvalues[i] = 1.0 / size;
        }
    }
    
    // Free allocated memory
    free(work_matrix);
    free(eigenvector);
    free(temp_vector);
#else
    // Standard implementation for non-ARM processors
    // Allocate memory for work matrices and vectors
    double _Complex *work_matrix = (double _Complex *)malloc(size * size * sizeof(double _Complex));
    double _Complex *eigenvector = (double _Complex *)malloc(size * sizeof(double _Complex));
    double _Complex *temp_vector = (double _Complex *)malloc(size * sizeof(double _Complex));
    
    if (!work_matrix || !eigenvector || !temp_vector) {
        fprintf(stderr, "Error: Memory allocation failed in eigenvalue calculation\n");
        if (work_matrix) free(work_matrix);
        if (eigenvector) free(eigenvector);
        if (temp_vector) free(temp_vector);
        
        // Set eigenvalues to equal probabilities as fallback
        for (int i = 0; i < size; i++) {
            eigenvalues[i] = 1.0 / size;
        }
        return;
    }
    
    // Copy the original matrix to work matrix
    for (int i = 0; i < size * size; i++) {
        work_matrix[i] = matrix[i];
    }
    
    // Initialize eigenvalues array
    for (int i = 0; i < size; i++) {
        eigenvalues[i] = 0.0;
    }
    
    // Number of eigenvalues to extract
    int num_eigen = (size > 16) ? 16 : size;
    
    // Extract eigenvalues using power iteration
    for (int k = 0; k < num_eigen; k++) {
        // Initialize eigenvector with random values
        double norm = 0.0;
        for (int i = 0; i < size; i++) {
            double real = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double imag = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            eigenvector[i] = real + imag * _Complex_I;
            norm += cabs(eigenvector[i]) * cabs(eigenvector[i]);
        }
        
        // Normalize the initial vector
        norm = sqrt(norm);
        for (int i = 0; i < size; i++) {
            eigenvector[i] /= norm;
        }
        
        // Power iteration
        double lambda = 0.0;
        double prev_lambda = -1.0;
        int max_iter = 100;
        double tolerance = 1e-6;
        
        for (int iter = 0; iter < max_iter && fabs(lambda - prev_lambda) > tolerance; iter++) {
            prev_lambda = lambda;
            
            // Matrix-vector multiplication
            for (int i = 0; i < size; i++) {
                temp_vector[i] = 0.0;
                for (int j = 0; j < size; j++) {
                    temp_vector[i] += work_matrix[i * size + j] * eigenvector[j];
                }
            }
            
            // Calculate Rayleigh quotient
            double _Complex num = 0.0;
            double denom = 0.0;
            for (int i = 0; i < size; i++) {
                num += conj(eigenvector[i]) * temp_vector[i];
                denom += cabs(eigenvector[i]) * cabs(eigenvector[i]);
            }
            
            lambda = creal(num / denom);
            
            // Normalize eigenvector
            norm = 0.0;
            for (int i = 0; i < size; i++) {
                eigenvector[i] = temp_vector[i];
                norm += cabs(eigenvector[i]) * cabs(eigenvector[i]);
            }
            
            norm = sqrt(norm);
            if (norm < 1e-10) break;
            
            for (int i = 0; i < size; i++) {
                eigenvector[i] /= norm;
            }
        }
        
        // Store eigenvalue
        eigenvalues[k] = lambda;
        
        // Matrix deflation
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                work_matrix[i * size + j] -= lambda * eigenvector[i] * conj(eigenvector[j]);
            }
        }
    }
    
    // Ensure eigenvalues sum to 1 (for density matrices)
    double sum = 0.0;
    for (int i = 0; i < num_eigen; i++) {
        sum += eigenvalues[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < num_eigen; i++) {
            eigenvalues[i] /= sum;
        }
    } else {
        // Fallback to equal eigenvalues
        for (int i = 0; i < size; i++) {
            eigenvalues[i] = 1.0 / size;
        }
    }
    
    // Free allocated memory
    free(work_matrix);
    free(eigenvector);
    free(temp_vector);
#endif
}

// Calculate von Neumann entropy with appropriate implementation
double von_neumann_entropy_neon(double _Complex *density_matrix, int size) {
#if HAS_NEON
    // NEON-optimized implementation
    if (!density_matrix) return 0.0;
    
    // First, ensure the density matrix is properly normalized (trace = 1)
    double trace = 0.0;
    for (int i = 0; i < size; i++) {
        trace += creal(density_matrix[i * size + i]);
    }
    
    // Create a normalized copy of the density matrix if needed
    double _Complex *normalized_matrix = NULL;
    double _Complex *matrix_to_use = density_matrix;
    
    if (fabs(trace - 1.0) > 1e-6) {
        if (getenv("DEBUG_ENTROPY")) {
            printf("DEBUG: Normalizing density matrix with trace = %f for NEON calculation\n", trace);
        }
        
        // If trace is significant, normalize; otherwise use maximally mixed state
        if (trace > 1e-10) {
            normalized_matrix = (double _Complex *)malloc(size * size * sizeof(double _Complex));
            if (!normalized_matrix) {
                fprintf(stderr, "Error: Memory allocation failed for normalized matrix in NEON calculation\n");
                return 0.0;
            }
            
            // Normalize the matrix
            for (int i = 0; i < size * size; i++) {
                normalized_matrix[i] = density_matrix[i] / trace;
            }
            matrix_to_use = normalized_matrix;
        } else {
            // For nearly zero trace, use maximally mixed state
            if (getenv("DEBUG_ENTROPY")) {
                printf("DEBUG: Trace too small (%e) in NEON calculation, using maximally mixed state\n", trace);
            }
            
            normalized_matrix = (double _Complex *)malloc(size * size * sizeof(double _Complex));
            if (!normalized_matrix) {
                fprintf(stderr, "Error: Memory allocation failed for normalized matrix in NEON calculation\n");
                return 0.0;
            }
            
            // Set up maximally mixed state (identity/size)
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    normalized_matrix[i * size + j] = (i == j) ? 1.0 / size : 0.0;
                }
            }
            matrix_to_use = normalized_matrix;
        }
    }
    
    // Allocate memory for eigenvalues
    double *eigenvalues = (double *)malloc(size * sizeof(double));
    if (!eigenvalues) {
        fprintf(stderr, "Error: Memory allocation failed for eigenvalues\n");
        if (normalized_matrix) free(normalized_matrix);
        return 0.0;
    }
    
    // Calculate eigenvalues using NEON-optimized power iteration method
    calculate_eigenvalues_neon(matrix_to_use, eigenvalues, size);
    
    // Calculate sum of eigenvalues (may include negative ones for quantum systems)
    double eigensum = 0.0;
    for (int i = 0; i < size; i++) {
        eigensum += eigenvalues[i];
    }
    
    // Normalize eigenvalues if their sum is significantly different from 1
    if (fabs(eigensum - 1.0) > 1e-6 && eigensum > 1e-10) {
        for (int i = 0; i < size; i++) {
            eigenvalues[i] /= eigensum;
        }
    } else if (eigensum <= 1e-10) {
        // If eigenvalues sum to nearly zero, use uniform distribution
        for (int i = 0; i < size; i++) {
            eigenvalues[i] = 1.0 / size;
        }
    }
    
    // Calculate entropy from eigenvalues: S = -Σλ_i log(λ_i)
    double entropy = 0.0;
    for (int i = 0; i < size; i++) {
        if (eigenvalues[i] > 1e-10) {  // Avoid log(0)
            entropy -= eigenvalues[i] * log(eigenvalues[i]);
        }
    }
    
    // Free allocated memory
    if (normalized_matrix) free(normalized_matrix);
    free(eigenvalues);
    
    // Allow negative entropy values for quantum systems
    if (entropy < 0.0 && getenv("DEBUG_ENTROPY")) {
        printf("DEBUG: NEON calculation produced negative entropy (%f)\n", entropy);
    }
    
    return entropy;
#else
    // Standard implementation for non-ARM processors
    if (!density_matrix) return 0.0;
    
    // Ensure the density matrix is properly normalized
    double trace = 0.0;
    for (int i = 0; i < size; i++) {
        trace += creal(density_matrix[i * size + i]);
    }
    
    // Create a normalized copy if needed
    double _Complex *normalized_matrix = NULL;
    double _Complex *matrix_to_use = density_matrix;
    
    if (fabs(trace - 1.0) > 1e-6) {
        if (getenv("DEBUG_ENTROPY")) {
            printf("DEBUG: Normalizing density matrix with trace = %f\n", trace);
        }
        
        if (trace > 1e-10) {
            normalized_matrix = (double _Complex *)malloc(size * size * sizeof(double _Complex));
            if (!normalized_matrix) {
                fprintf(stderr, "Error: Memory allocation failed for normalized matrix\n");
                return 0.0;
            }
            
            for (int i = 0; i < size * size; i++) {
                normalized_matrix[i] = density_matrix[i] / trace;
            }
            matrix_to_use = normalized_matrix;
        } else {
            normalized_matrix = (double _Complex *)malloc(size * size * sizeof(double _Complex));
            if (!normalized_matrix) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                return 0.0;
            }
            
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    normalized_matrix[i * size + j] = (i == j) ? 1.0 / size : 0.0;
                }
            }
            matrix_to_use = normalized_matrix;
        }
    }
    
    // Calculate eigenvalues
    double *eigenvalues = (double *)malloc(size * sizeof(double));
    if (!eigenvalues) {
        if (normalized_matrix) free(normalized_matrix);
        return 0.0;
    }
    
    calculate_eigenvalues_neon(matrix_to_use, eigenvalues, size);
    
    // Calculate entropy
    double entropy = 0.0;
    for (int i = 0; i < size; i++) {
        if (eigenvalues[i] > 1e-10) {
            entropy -= eigenvalues[i] * log(eigenvalues[i]);
        }
    }
    
    if (normalized_matrix) free(normalized_matrix);
    free(eigenvalues);
    
    return entropy;
#endif
}
