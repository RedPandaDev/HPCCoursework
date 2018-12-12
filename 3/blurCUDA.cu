#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#include <math.h>

__global__ 
void doBlur(int *R,int *h_R,int *G,int *h_G,int *B,int *h_B,int colsize,int rowsize){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;


		if(col<colsize && row<rowsize){
				if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)){

					R[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[(row-1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])/4;
					G[row * colsize + col] = (h_G[(row+1) * colsize + col]+h_G[(row-1) * colsize + col]+h_G[row * colsize + (col+1)]+h_G[row * colsize + (col-1)])/4;
					B[row * colsize + col] = (h_B[(row+1) * colsize + col]+h_B[(row-1) * colsize + col]+h_B[row * colsize + (col+1)]+h_B[row * colsize + (col-1)])/4;
					// R[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[(row-1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])*0;
					// G[row * colsize + col] = (h_G[(row+1) * colsize + col]+h_G[(row-1) * colsize + col]+h_G[row * colsize + (col+1)]+h_G[row * colsize + (col-1)])*0;
					// B[row * colsize + col] = (h_B[(row+1) * colsize + col]+h_B[(row-1) * colsize + col]+h_B[row * colsize + (col+1)]+h_B[row * colsize + (col-1)])*0;

				}
				else if (row == 0 && col != 0 && col != (colsize-1)){
					R[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])/3;
					G[row * colsize + col] = (h_G[(row+1) * colsize + col]+h_G[row * colsize + (col+1)]+h_G[row * colsize + (col-1)])/3;
					B[row * colsize + col] = (h_B[(row+1) * colsize + col]+h_B[row * colsize + (col+1)]+h_B[row * colsize + (col-1)])/3;
					// R[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])*0;
					// G[row * colsize + col] = (h_G[(row+1) * colsize + col]+h_G[row * colsize + (col+1)]+h_G[row * colsize + (col-1)])*0;
					// B[row * colsize + col] = (h_B[(row+1) * colsize + col]+h_B[row * colsize + (col+1)]+h_B[row * colsize + (col-1)])*0;
					
				}
				else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
					R[row * colsize + col] = (h_R[(row-1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])/3;
					G[row * colsize + col] = (h_G[(row-1) * colsize + col]+h_G[row * colsize + (col+1)]+h_G[row * colsize + (col-1)])/3;
					B[row * colsize + col] = (h_B[(row-1) * colsize + col]+h_B[row * colsize + (col+1)]+h_B[row * colsize + (col-1)])/3;
					
				}
				else if (col == 0 && row != 0 && row != (rowsize-1)){
					R[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[(row-1) * colsize + col]+h_R[row * colsize + (col+1)])/3;
					G[row * colsize + col] = (h_G[(row+1) * colsize + col]+h_G[(row-1) * colsize + col]+h_G[row * colsize + (col+1)])/3;
					B[row * colsize + col] = (h_B[(row+1) * colsize + col]+h_B[(row-1) * colsize + col]+h_B[row * colsize + (col+1)])/3;
					
				}
				else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
					R[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[(row-1) * colsize + col]+h_R[row * colsize + (col-1)])/3;
					G[row * colsize + col] = (h_G[(row+1) * colsize + col]+h_G[(row-1) * colsize + col]+h_G[row * colsize + (col-1)])/3;
					B[row * colsize + col] = (h_B[(row+1) * colsize + col]+h_B[(row-1) * colsize + col]+h_B[row * colsize + (col-1)])/3;
					
				}
				else if (row==0 &&col==0){
					R[row * colsize + col] = (h_R[row * colsize + (col+1)]+h_R[(row+1) * colsize + col])/2;
					G[row * colsize + col] = (h_G[row * colsize + (col+1)]+h_G[(row+1) * colsize + col])/2;
					B[row * colsize + col] = (h_B[row * colsize + (col+1)]+h_B[(row+1) * colsize + col])/2;
					
				}
				else if (row==0 &&col==(colsize-1)){
					R[row * colsize + col] = (h_R[row * colsize + (col-1)]+h_R[(row+1) * colsize + col])/2;
					G[row * colsize + col] = (h_G[row * colsize + (col-1)]+h_G[(row+1) * colsize + col])/2;
					B[row * colsize + col] = (h_B[row * colsize + (col-1)]+h_B[(row+1) * colsize + col])/2;
					
				}
				else if (row==(rowsize-1) &&col==0){
					R[row * colsize + col] = (h_R[row * colsize + (col+1)]+h_R[(row-1) * colsize + col])/2;
					G[row * colsize + col] = (h_G[row * colsize + (col+1)]+h_G[(row-1) * colsize + col])/2;
					B[row * colsize + col] = (h_B[row * colsize + (col+1)]+h_B[(row-1) * colsize + col])/2;
					
				}
				else if (row==(rowsize-1) &&col==(colsize-1)){
					R[row * colsize + col] = (h_R[row * colsize + (col-1)]+h_R[(row-1) * colsize + col])/2;
					G[row * colsize + col] = (h_G[row * colsize + (col-1)]+h_G[(row-1) * colsize + col])/2;
					B[row * colsize + col] = (h_B[row * colsize + (col-1)]+h_B[(row-1) * colsize + col])/2;
					
				}	




			}
		
}
__global__ 
void doCopy(int *R,int *h_R,int *G,int *h_G,int *B,int *h_B,int colsize,int rowsize){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

	if(col<colsize && row<rowsize){
		h_R[row * colsize + col] = R[row * colsize + col];
		h_G[row * colsize + col] = G[row * colsize + col];
		h_B[row * colsize + col] = B[row * colsize + col];

	}
	

}

int main (int argc, const char * argv[]) {
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	int row = 0, col = 0, nblurs, lineno=0, k;
	struct timeval tim;
	int *R, *B, *G;
   	int sizei; 
   	sizei = sizeof(int)*colsize*rowsize;

   	R = (int*)malloc(sizei);
   	G = (int*)malloc(sizei);
   	B = (int*)malloc(sizei);
	
	fp = fopen("David.ps", "r");
 
	while(! feof(fp))
	{
		fscanf(fp, "\n%[^\n]", str);
		if (nlines < 5) {strcpy((char *)lines[nlines++],(char *)str);}

		else{
			for (sptr=&str[0];*sptr != '\0';sptr+=6){
				sscanf(sptr,"%2x",&h1);
				sscanf(sptr+2,"%2x",&h2);
				sscanf(sptr+4,"%2x",&h3);

				if (col==colsize){
					col = 0;
					row++;
				}
				if (row < rowsize && col < colsize) {

					R[row * colsize + col] = h1;

					G[row * colsize + col] = h2;

					B[row * colsize + col] = h3;
				}
				col++;
			}
		}

	}
	fclose(fp);
	
	nblurs = 160;
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	int *Rnew, *Bnew, *Gnew;
	int *h_R, *h_G, *h_B;

	h_R = (int*)malloc(sizei);

	h_G = (int*)malloc(sizei);
		
	h_B = (int*)malloc(sizei);

	// memset(h_R, 0, sizeof h_R);
	// memset(R, 0, sizeof R);

	// memset(h_G, 0, sizeof h_G);
	// memset(G, 0, sizeof G);

	// memset(h_B, 0, sizeof h_B);
	// memset(B, 0, sizeof B);


	Rnew = (int*)malloc(sizei);
   	Gnew = (int*)malloc(sizei);
   	Bnew = (int*)malloc(sizei);

   	int *d_R, *d_G, *d_B;


	cudaMalloc((void **)&h_R,sizei);
	cudaMalloc((void **)&h_G,sizei);
	cudaMalloc((void **)&h_B,sizei);

	cudaMalloc((void **)&d_R,sizei);
	cudaMalloc((void **)&d_G,sizei);
	cudaMalloc((void **)&d_B,sizei);

	cudaMemcpy(h_R,R,sizei,cudaMemcpyHostToDevice);
	cudaMemcpy(h_G,G,sizei,cudaMemcpyHostToDevice);
	cudaMemcpy(h_B,B,sizei,cudaMemcpyHostToDevice);


	dim3 dimGrid(ceil(colsize/(float)32),ceil(rowsize/(float)32),1);
    dim3 dimBlock(32,32,1);


	for(k=0;k<nblurs;k++){

		doBlur<<<dimGrid,dimBlock>>>(d_R,h_R,d_G,h_G,d_B,h_B,colsize,rowsize);
        doCopy<<<dimGrid,dimBlock>>>(d_R,h_R,d_G,h_G,d_B,h_B,colsize,rowsize);	
	}



	cudaMemcpy(Rnew,h_R,sizei,cudaMemcpyDeviceToHost);
	cudaMemcpy(Gnew,h_G,sizei,cudaMemcpyDeviceToHost);
	cudaMemcpy(Bnew,h_B,sizei,cudaMemcpyDeviceToHost);

	cudaFree(h_R);
	cudaFree(h_G);
	cudaFree(h_B);

	cudaFree(d_R);
	cudaFree(d_G);
	cudaFree(d_B);

	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed\n", t2-t1);


	fout= fopen("DavidBlur.ps", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%02x%02x%02x",Rnew[row*colsize+col],Gnew[row*colsize+col],Bnew[row*colsize+col]);
			lineno++;
			if (lineno==linelen){
				fprintf(fout,"\n");
				lineno = 0;
			}
		}
	}
	fclose(fout);
    return 0;
}
