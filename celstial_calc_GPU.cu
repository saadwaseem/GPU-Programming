#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
// unsigned int *d_histogram;
float *d_histogram;

const float PI = 3.141592;
const int ARR_SIZE = 720;

/**
 * Performs calculations on celestial object data and updates a histogram array.
 *
 * This CUDA kernel function takes in two sets of celestial object data, consisting of right ascension and declination values. 
 * The function then performs calculations based on this data and increments histogram bins.
 *
 * @param float *ra - A pointer to an array containing right ascension.
 * @param float *decl - A pointer to an array containing declination.
 * @param int size - The size of the `ra` and `decl` arrays.
 * @param unsigned *hist - A pointer to an array of unsigned integers that stores the histogram values.
 * @param float *second_ra - A pointer to an array containing right ascension values for the second set of celestial objects.
 * @param float *second_decl - A pointer to an array containing declination values for the second set of celestial objects.
 * @param int second_size - The size of the `second_ra` and `second_decl` arrays.
 * @param int mode - An optional integer parameter that indicates the mode of operation for the kernel function.
 *                    If set to 0, the function calculates the histogram values for DD or RR.
 *                    If set to 1, the function calculates the histogram values for DR.
 */
__global__ void calandcount(float *ra, float *decl, int size, unsigned *hist, float *second_ra, float *second_decl, int second_size, int mode){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        float theta;
        if( mode == 0 ){ // hist_DR
            for (int j = 0; j < size; j++) {
                theta = sin(decl[tid])*sin(second_decl[j]) + cos(decl[tid])*cos(second_decl[j])*cos(ra[tid]-second_ra[j]);
                if (theta>1.0) theta = 1.0;
                if (theta<-1.0) theta = -1.0;
                theta = acos( theta ) * (180 / PI);
                unsigned bin_index = theta * binsperdegree;
                if (bin_index < ARR_SIZE){
                    atomicAdd(&hist[bin_index], 1);
                }
                else
                    printf("index out of bound %d\n", bin_index);
            }
        }else{ // hist RR and hist_DD
            for (int j = tid+1; j < size; j++) {                                        
                theta = sin(decl[tid])*sin(decl[j]) + cos(decl[tid])*cos(decl[j])*cos(ra[tid]-ra[j]);
                if (theta>1.0) theta = 1.0;
                if (theta<-1.0) theta = -1.0;
                theta = acos( theta ) * (180 / PI);
                unsigned bin_index = theta * binsperdegree;
                if (bin_index < ARR_SIZE){
                    atomicAdd(&hist[bin_index], 2);
                }
                else
                    printf("index out of bound %d\n", bin_index);
            }
            atomicAdd(&hist[0],1);                                             
        }

    }
}

/**
 * Calculate Omega values
 * @param float *omega_diffs - A pointer to an array to contain omega values.
 * omega = (DD[i] - 2*DR[i] + RR[i]) / RR[i]
*/
void calcOmega(float *omega_diffs){
    
    for (int i = 0; i < ARR_SIZE ; i++)
    {
        if(histogramRR[i] > 0){
            omega_diffs[i] = (float) (histogramDD[i] - (2 * histogramDR[i]) + histogramRR[i]) / histogramRR[i];
        }
        else
        omega_diffs[i] = 0.0;
    }
}

/**
 * Write omega, hist_DD, hist_DR, hist_RR to results_file
 * @param float *omega_diffs - A pointer to omega values.
 * @param char *results_file - results file name.
 */
void resultsToFile(float *omega_diffs, char *results_file) {
    FILE *file_ptr;

    // use appropriate location if you are using MacOS or Linux
    file_ptr = fopen(results_file,"w");

    if (file_ptr == NULL)
    {
        printf("Error opening file!");
        exit(1);
    }

    fprintf(file_ptr, "bin_start\t omega\t hist_DD\t hist_DR\t hist_RR\n");
    
    float bin;

    for (int i = 0; i < ARR_SIZE; i++) {
        bin = (float) i / binsperdegree;
        fprintf(file_ptr, "%.3f\t %.6f\t %d\t %d\t %d \n",bin,omega_diffs[i],histogramDD[i],histogramDR[i],histogramRR[i]);
    }
   
    fclose(file_ptr);
}

/**
 * Convert angle from Arc Mints to Radian
 * @param float arcmints - angle in arcmints.
 * @return float rad - return angle in radian
*/
float arcmintorad(float arcmints){
    float rad = arcmints * (PI/(60*180));
    return rad;
}

int main(int argc, char *argv[])
{
   int    noofblocks;
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   long int histogramDRsum=0, histogramDDsum=0, histogramRRsum=0;
   
   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   if ( getDevice(0) != 0 ) return(-1);

   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);
   
   // Vectors for holding the host-side (CPU-side) data
   histogramDD   = (unsigned *)calloc(ARR_SIZE ,sizeof(unsigned));
   histogramRR   = (unsigned *)calloc(ARR_SIZE ,sizeof(unsigned));
   histogramDR   = (unsigned *)calloc(ARR_SIZE ,sizeof(unsigned));
   d_histogram   = (float *)calloc(ARR_SIZE ,sizeof(float));

   // allocate mameory on the GPU
   unsigned int *g_histogramDD, *g_histogramRR, *g_histogramDR;
   int hist_size = ARR_SIZE * sizeof(unsigned);
   cudaMalloc(&g_histogramDD, hist_size);
   cudaMalloc(&g_histogramRR, hist_size);
   cudaMalloc(&g_histogramDR, hist_size);
   
   float *g_ra_real, *g_decl_real, *g_ra_random, *g_decl_random;
   int real_mem_size = NoofReal * sizeof(float);
   int random_mem_size = NoofSim * sizeof(float);
   cudaMalloc(&g_ra_real, real_mem_size);
   cudaMalloc(&g_decl_real, real_mem_size);
   cudaMalloc(&g_ra_random, random_mem_size);
   cudaMalloc(&g_decl_random, random_mem_size);

   // copy data to the GPU
   cudaMemcpy(g_histogramDD, histogramDD, hist_size, cudaMemcpyHostToDevice);
   cudaMemcpy(g_histogramRR, histogramRR, hist_size, cudaMemcpyHostToDevice);
   cudaMemcpy(g_histogramDR, histogramDR, hist_size, cudaMemcpyHostToDevice);
   cudaMemcpy(g_ra_real, ra_real, real_mem_size, cudaMemcpyHostToDevice);
   cudaMemcpy(g_decl_real, decl_real, real_mem_size, cudaMemcpyHostToDevice);
   cudaMemcpy(g_ra_random, ra_sim, random_mem_size, cudaMemcpyHostToDevice);
   cudaMemcpy(g_decl_random, decl_sim, random_mem_size, cudaMemcpyHostToDevice);
   
   //64-bit integer to hold a value of 10 billion
   unsigned long long problem_size = NoofReal * NoofReal;
   //Warp size is also 32. I tried different size for NUM_THREADS but 
   //there was no further decrease in time. So I decided to go with 32.
   int NUM_THREADS = 32;
   //calculate number of blocks according to problem size and number of threads
   noofblocks = (problem_size + NUM_THREADS - 1) / NUM_THREADS;
   
   cudaEvent_t start, stop;
   float elapsed_time;
   
   // Create CUDA events
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // Record the start event
   cudaEventRecord(start, 0);

   // run the kernels on the GPU
   calandcount<<<noofblocks, NUM_THREADS>>>(g_ra_real, g_decl_real, NoofReal, g_histogramDD, g_ra_random, g_decl_random, NoofSim,1);
   calandcount<<<noofblocks, NUM_THREADS>>>(g_ra_random, g_decl_random, NoofSim, g_histogramRR, g_ra_random, g_decl_random, NoofSim,1);
   calandcount<<<noofblocks, NUM_THREADS>>>(g_ra_real, g_decl_real, NoofReal, g_histogramDR, g_ra_random, g_decl_random, NoofSim, 0);
   cudaDeviceSynchronize();

   // Record the stop event
   cudaEventRecord(stop, 0);

   // Wait for the stop event to complete
   cudaEventSynchronize(stop);

   // Calculate the elapsed time
   cudaEventElapsedTime(&elapsed_time, start, stop);

   // Print the elapsed time
   printf("Kernel execution time: %f ms\n", elapsed_time);

   // Destroy the CUDA events
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   // copy the results back to the CPU
   cudaMemcpy(histogramDD, g_histogramDD, hist_size, cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramRR, g_histogramRR, hist_size, cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramDR, g_histogramDR, hist_size, cudaMemcpyDeviceToHost);

   // calculate omega values on the CPU
   calcOmega(d_histogram);

   //Write results to file
   resultsToFile(d_histogram, argv[3]); 
    
   for ( int i = 0; i < ARR_SIZE; i++ ){
    histogramDDsum += histogramDD[i];
   }
   
   //Check points to verify calculations are done right.
   printf("histogramDDsum = %ld\n",histogramDDsum);
   if ( histogramDDsum != 10000000000 ) {
        printf("Incorrect sum, exiting....\n");
        return(0);
    }
    
   for ( int i = 0; i < ARR_SIZE; i++ ){
    histogramRRsum += histogramRR[i];
   }

   printf("histogramRRsum = %ld\n",histogramRRsum);
   if ( histogramRRsum != 10000000000 ) {
        printf("Incorrect sum, exiting....\n");
        return(0);
    }
    
   for ( int i = 0; i < ARR_SIZE; i++ ){
    histogramDRsum += histogramDR[i];
   }

   printf("histogramDRsum = %ld\n",histogramDRsum);
   if ( histogramDRsum != 10000000000 ) {
        printf("Incorrect sum, exiting....\n");
        return(0);
    }

   cudaFree(g_histogramDD);
   cudaFree(g_histogramRR);
   cudaFree(g_histogramDR);
   cudaFree(g_ra_real);
   cudaFree(g_decl_real);
   cudaFree(g_ra_random);
   cudaFree(g_decl_random);
   
   return(0);
}

int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = arcmintorad(ra);
      decl_real[i] = arcmintorad(dec);
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}
  announcednumber = -1;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = arcmintorad(ra);
      decl_sim[i] = arcmintorad(dec);
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}

int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}