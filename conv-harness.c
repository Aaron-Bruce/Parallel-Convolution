/* Test and timing harness program for developing a multichannel
   multikernel convolution (as used in deep learning networks)

   Note there are some simplifications around this implementation,
   in particular with respect to computing the convolution at edge
   pixels of the image.

   Author: David Gregg
   Date:   March 2022

   Version 1.7 : Adjusted types for mixed-type computation

   Version 1.6 : Modified the code so that the input tensor is float

   Version 1.5 : Modified the code so that the input and kernel
                 are tensors of 16-bit integer values

   Version 1.4 : Modified the random generator to reduce the range
                 of generated values;

   Version 1.3 : Fixed which loop variables were being incremented
                 in write_out();
                 Fixed dimensions of output and control_output 
                 matrices in main function

   Version 1.2 : Changed distribution of test data to (hopefully) 
                 eliminate random walk of floating point error;
                 Also introduced checks to restrict kernel-order to
                 a small set of values

   Version 1.1 : Fixed bug in code to create 4d matrix
*/

#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

/* write 3d matrix to stdout */
void write_out(int16_t ***a, int dim0, int dim1, int dim2)
{
    int i, j, k;

    for (i = 0; i < dim0; i++)
    {
        printf("Outer dimension number %d\n", i);
        for (j = 0; j < dim1; j++)
        {
            for (k = 0; k < dim2 - 1; k++)
            {
                printf("%d, ", a[i][j][k]);
            }
            // print end of line
            printf("%d\n", a[i][j][dim2 - 1]);
        }
    }
}

/* create new empty 4d float matrix */
double ****new_empty_4d_matrix_float(int dim0, int dim1, int dim2, int dim3)
{
    double ****result = malloc(dim0 * sizeof(double ***));
    double ***mat1 = malloc(dim0 * dim1 * sizeof(double **));
    double **mat2 = malloc(dim0 * dim1 * dim2 * sizeof(double *));
    double *mat3 = malloc(dim0 * dim1 * dim2 * dim3 * sizeof(double));
    int i, j, k;

    for (i = 0; i < dim0; i++)
    {
        result[i] = &(mat1[i * dim1]);
        for (j = 0; j < dim1; j++)
        {
            result[i][j] = &(mat2[i * dim1 * dim2 + j * dim2]);
            for (k = 0; k < dim2; k++)
            {
                result[i][j][k] = &(mat3[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3]);
            }
        }
    }

    return result;
}

/* create new empty 3d matrix */
double ***new_empty_3d_matrix_float(int dim0, int dim1, int dim2)
{
    double ****mat4d;
    double ***mat3d;

    // create a 4d matrix with single first dimension
    mat4d = new_empty_4d_matrix_float(1, dim0, dim1, dim2);
    // now throw away out first dimension
    mat3d = mat4d[0];
    free(mat4d);
    return mat3d;
}

/* create new empty 4d int16_t matrix */
double ****new_empty_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
    double ****result = malloc(dim0 * sizeof(double ***));
    double ***mat1 = malloc(dim0 * dim1 * sizeof(double **));
    double **mat2 = malloc(dim0 * dim1 * dim2 * sizeof(double *));
    double *mat3 = malloc(dim0 * dim1 * dim2 * dim3 * sizeof(double));
    int i, j, k;

    for (i = 0; i < dim0; i++)
    {
        result[i] = &(mat1[i * dim1]);
        for (j = 0; j < dim1; j++)
        {
            result[i][j] = &(mat2[i * dim1 * dim2 + j * dim2]);
            for (k = 0; k < dim2; k++)
            {
                result[i][j][k] = &(mat3[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3]);
            }
        }
    }

    return result;
}

/* create new empty 3d matrix */
double ***new_empty_3d_matrix_int16(int dim0, int dim1, int dim2)
{
    double ****mat4d;
    double ***mat3d;

    // create a 4d matrix with single first dimension
    mat4d = new_empty_4d_matrix_int16(1, dim0, dim1, dim2);
    // now throw away out first dimension
    mat3d = mat4d[0];
    free(mat4d);
    return mat3d;
}

/* take a copy of the matrix and return in a newly allocated matrix */
double ****copy_4d_matrix(double ****source_matrix, int dim0,
                           int dim1, int dim2, int dim3)
{
    int i, j, k, l;
    double ****result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

    for (i = 0; i < dim0; i++)
    {
        for (j = 0; j < dim1; j++)
        {
            for (k = 0; k < dim2; k++)
            {
                for (l = 0; l < dim3; l++)
                {
                    result[i][j][k][l] = source_matrix[i][j][k][l];
                }
            }
        }
    }
    return result;
}

/* create a matrix and fill it with random numbers */
double ****gen_random_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
    double ****result;
    int i, j, k, l;
    struct timeval seedtime;
    int seed;

    result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

    /* use the microsecond part of the current time as a pseudorandom seed */
    gettimeofday(&seedtime, NULL);
    seed = seedtime.tv_usec;
    srandom(seed);

    /* fill the matrix with random numbers */
    const int range = 1 << 10; // 2^10
    //const int bias = 1 << 16; // 2^16
    int16_t offset = 0.0;
    for (i = 0; i < dim0; i++)
    {
        for (j = 0; j < dim1; j++)
        {
            for (k = 0; k < dim2; k++)
            {
                for (l = 0; l < dim3; l++)
                {
                    // generate uniform random integer with mean of zero
                    long long rand = random();
                    // now cut down the range and bias the mean to reduce
                    // the likelihood of large floating point round-off errors
                    int reduced_range = (rand % range);
                    result[i][j][k][l] = (double) reduced_range;
                }
            }
        }
    }

    return result;
}

/* create a matrix and fill it with random numbers */
double ****gen_random_4d_matrix_float(int dim0, int dim1, int dim2, int dim3)
{
    double ****result;
    int i, j, k, l;
    struct timeval seedtime;
    int seed;

    result = new_empty_4d_matrix_float(dim0, dim1, dim2, dim3);

    /* use the microsecond part of the current time as a pseudorandom seed */
    gettimeofday(&seedtime, NULL);
    seed = seedtime.tv_usec;
    srandom(seed);

    /* fill the matrix with random numbers */
    const int range = 1 << 12; // 2^12
    const int bias = 1 << 10;  // 2^16
    int16_t offset = 0.0;
    for (i = 0; i < dim0; i++)
    {
        for (j = 0; j < dim1; j++)
        {
            for (k = 0; k < dim2; k++)
            {
                for (l = 0; l < dim3; l++)
                {
                    // generate uniform random integer with mean of zero
                    long long rand = random();
                    // now cut down the range and bias the mean to reduce
                    // the likelihood of large floating point round-off errors
                    int reduced_range = (rand % range);
                    result[i][j][k][l] = (double) reduced_range + bias;
                }
            }
        }
    }

    return result;
}

/* create a matrix and fill it with random numbers */
double ***gen_random_3d_matrix_float(int dim0, int dim1, int dim2)
{
    double ****mat4d;
    double ***mat3d;

    // create a 4d matrix with single first dimension
    mat4d = gen_random_4d_matrix_float(1, dim0, dim1, dim2);
    // now throw away out first dimension
    mat3d = mat4d[0];
    free(mat4d);
    return mat3d;
}

/* create a matrix and fill it with random numbers */
double ***gen_random_3d_matrix_int16(int dim0, int dim1, int dim2)
{
    double ****mat4d;
    double ***mat3d;

    // create a 4d matrix with single first dimension
    mat4d = gen_random_4d_matrix_int16(1, dim0, dim1, dim2);
    // now throw away out first dimension
    mat3d = mat4d[0];
    free(mat4d);
    return mat3d;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(double ***result, double ***control,
                  int dim0, int dim1, int dim2)
{
    int i, j, k;
    double sum_abs_diff = 0.0;
    const double EPSILON = 0.0625;
    
    int count = 0;
    int total = 0;

    //printf("SAD\n");

    for (i = 0; i < dim0; i++)
    {
        for (j = 0; j < dim1; j++)
        {
            for (k = 0; k < dim2; k++)
            {
                if (control[i][j][k] != result[i][j][k])
                {
                    //printf("Wrong: %f %f %d %d %d\n", control[i][j][k], result[i][j][k], i, j, k);
                }
                else
                {
                  //printf("Right: %f %f %d %d %d\n", control[i][j][k], result[i][j][k], i, j, k);
                }
                double diff = fabs(control[i][j][k] - result[i][j][k]);
                assert(diff >= 0.0);
                sum_abs_diff = sum_abs_diff + diff;
            }
        }
    }
    if (sum_abs_diff > EPSILON)
    {
        fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
                sum_abs_diff, EPSILON);
    }
    else
    {
        printf("COMMENT: sum of absolute differences (%f)  within acceptable range (%f)\n", sum_abs_diff, EPSILON);
    }
}

/* the slow but correct version of matmul written by David */
void multichannel_conv(double ***image, double ****kernels,
                       double ***output, int width, int height,
                       int nchannels, int nkernels, int kernel_order)
{
    int h, w, x, y, c, m;
    //printf("%d\n", kernels[0][0][0][0]);
    //printf("%d\n", kernels[0][1][0][0]);
    for (m = 0; m < nkernels; m++)
    {
        for (w = 0; w < width; w++)
        {
            for (h = 0; h < height; h++)
            {
                double sum = 0.0;
                for (c = 0; c < nchannels; c++)
                {
                    //printf("%d\n", kernels[m][c][0][0]);
                    for (x = 0; x < kernel_order; x++)
                    {
                        for (y = 0; y < kernel_order; y++)
                        {
                            sum += image[w + x][h + y][c] * kernels[m][c][x][y];
                        }
                    }
                    output[m][w][h] = sum;
                }
            }
        }
    }
    // printf("Greggs output: %f\n", output[9][11][10]);
}

/* the fast version of matmul written by the student */
void student_conv(double ***image,double ****kernels, double ***output,int width, int height, int nchannels, int nkernels,int kernel_order)
{
    int h, w, x, y, c, m,off,other;
    double temp[2];   
    for (m = 0; m < nkernels; m++)
    {
        for (w = 0; w < width; w++)
        {
            for (h = 0; h < height; h++)
            {
                __m128d sum4;
                sum4 = _mm_setzero_pd();
                for (c = 0; c < nchannels; c++)
                {
                    for(x = 0; x<kernel_order; x++)
                    {
#pragma omp parallel for
                        for(y = 0; y<kernel_order; y += 2)
                        {
                          __m128d x4, k4,product4;
                          if(y == kernel_order-1)
                          {
                            k4 = _mm_loadu_pd(&(kernels[m][c][x][y]));
                            x4 = _mm_set_pd(0,image[w+x][h+y][c]);
                          }
                          else
                          {
                            //off = y+1;
                            k4 = _mm_loadu_pd(&(kernels[m][c][x][y]));
                            x4 = _mm_set_pd(image[w+x][h+y+1][c],image[w+x][h+y][c]);
                          }
                          product4 = _mm_mul_pd(k4,x4);
#pragma omp critical
{
                          sum4 = _mm_add_pd(sum4,product4);
                          }
                        }
                    }
                }
                _mm_storeu_pd(temp, sum4);
                //printf("%f %f", temp[0], temp[1]);
                output[m][w][h] = temp[0] + temp[1];
            }
        }
    }
    //printf("My output: %f\n", output[9][11][10]);
    //multichannel_conv(image, kernels, output, width,
    //                  height, nchannels, nkernels, kernel_order);
    printf("%d %d\n", off, other);
}

int main(int argc, char **argv)
{
    //float image[W][H][C];
    //float kernels[M][C][K][K];
    //float output[M][W][H];

    double ***image;
    double ****kernels;
    double ***control_output, ***output;
    double speedup,mul_time,gregg_mul_time;
    int width, height, kernel_order, nchannels, nkernels;
    struct timeval start_time;
    struct timeval stop_time;
    struct timeval gregg_start_time;
    struct timeval gregg_stop_time;

    if (argc != 6)
    {
        fprintf(stderr, "Usage: conv-harness <image_width> <image_height> <kernel_order> <number of channels> <number of kernels>\n");
        exit(1);
    }
    else
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        kernel_order = atoi(argv[3]);
        nchannels = atoi(argv[4]);
        nkernels = atoi(argv[5]);
    }
    switch (kernel_order)
    {
    case 1:
    case 3:
    case 5:
    case 7:
        break;
    default:
        fprintf(stderr, "FATAL: kernel_order must be 1, 3, 5 or 7, not %d\n",
                kernel_order);
        exit(1);
    }

    /* allocate the matrices */
    image = gen_random_3d_matrix_float(width + kernel_order, height + kernel_order,
                                       nchannels);
    kernels = gen_random_4d_matrix_int16(nkernels, nchannels, kernel_order, kernel_order);
    output = new_empty_3d_matrix_float(nkernels, width, height);
    control_output = new_empty_3d_matrix_float(nkernels, width, height);

    //DEBUGGING(write_out(A, a_dim1, a_dim2));
  
    gettimeofday(&gregg_start_time, NULL);
    
    /* use a simple multichannel convolution routine to produce control result */
    multichannel_conv(image, kernels, control_output, width,
                      height, nchannels, nkernels, kernel_order);
                      
    gettimeofday(&gregg_stop_time, NULL);
    gregg_mul_time = (gregg_stop_time.tv_sec - gregg_start_time.tv_sec) * 1000000L +
               (gregg_stop_time.tv_usec - gregg_start_time.tv_usec);
    printf("Gregg conv time: %f microseconds\n", gregg_mul_time);  
      
   
    /*double imageD[width+(kernel_order-1)][height+(kernel_order-1)][nchannels];
    double kernelsD[nkernels][nchannels][kernel_order][kernel_order];
    
    for(int i=0; i<width+(kernel_order-1); i++)
    {
      for(int j=0; j<height+(kernel_order-1); j++)
      {
        for(int k=0; k<nchannels; k++)
        {
            imageD[i][j][k] = (double) image[i][j][k];
        }
      }
    }
    
                      
    for(int i=0; i<nkernels; i++)
    {
      for(int j=0; j<nchannels; j++)
      {
        for(int k=0; k<kernel_order; k++)
        {
          for(int l=0; l<kernel_order; l++)
          {
            kernelsD[i][j][k][l] = (float) kernels[i][j][k][l];
          }
        }
      }
    }
    
    free(image);
    free(kernels);
    */

    /* record starting time of student's code*/
    gettimeofday(&start_time, NULL);

    /* perform student's multichannel convolution */
    student_conv(image,kernels,output, width,height, nchannels, nkernels, kernel_order);

    /* record finishing time */
    gettimeofday(&stop_time, NULL);
    mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
               (stop_time.tv_usec - start_time.tv_usec);
    printf("Student conv time: %f microseconds\n", mul_time);
    speedup = gregg_mul_time/mul_time;
    printf("Speedup: %f\n", speedup);
    

    DEBUGGING(write_out(output, nkernels, width, height));

    /* now check that the student's multichannel convolution routine
     gives the same answer as the known working version */
    check_result(output, control_output, nkernels, width, height);

    return 0;
}
