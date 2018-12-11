#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>

__global__ 
void performUpdatesKernel(int *h_Rnew, int *h_R,int *h_Gnew, int *h_G,int *h_Bnew, int *h_B,int colsize, int rowsize)
{
	int row = 0, col = 0;
    for(row=0;row<rowsize;row++){
			for (col=0;col<colsize;col++){	
				if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)){
					h_Rnew[row][col] = (h_R[row+1][col]+h_R[row-1][col]+h_R[row][col+1]+h_R[row][col-1])/4;
					h_Gnew[row][col] = (h_G[row+1][col]+h_G[row-1][col]+h_G[row][col+1]+h_G[row][col-1])/4;
					h_Bnew[row][col] = (h_B[row+1][col]+h_B[row-1][col]+h_B[row][col+1]+h_B[row][col-1])/4;
				}
				else if (row == 0 && col != 0 && col != (colsize-1)){
					h_Rnew[row][col] = (h_R[row+1][col]+h_R[row][col+1]+h_R[row][col-1])/3;
					h_Gnew[row][col] = (h_G[row+1][col]+h_G[row][col+1]+h_G[row][col-1])/3;
					h_Bnew[row][col] = (h_B[row+1][col]+h_B[row][col+1]+h_B[row][col-1])/3;
				}
				else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
					h_Rnew[row][col] = (h_R[row-1][col]+h_R[row][col+1]+h_R[row][col-1])/3;
					h_Gnew[row][col] = (h_G[row-1][col]+h_G[row][col+1]+h_G[row][col-1])/3;
					h_Bnew[row][col] = (h_B[row-1][col]+h_B[row][col+1]+h_B[row][col-1])/3;
				}
				else if (col == 0 && row != 0 && row != (rowsize-1)){
					h_Rnew[row][col] = (h_R[row+1][col]+h_R[row-1][col]+h_R[row][col+1])/3;
					h_Gnew[row][col] = (h_G[row+1][col]+h_G[row-1][col]+h_G[row][col+1])/3;
					h_Bnew[row][col] = (h_B[row+1][col]+h_B[row-1][col]+h_B[row][col+1])/3;
				}
				else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
					h_Rnew[row][col] = (h_R[row+1][col]+h_R[row-1][col]+h_R[row][col-1])/3;
					h_Gnew[row][col] = (h_G[row+1][col]+h_G[row-1][col]+h_G[row][col-1])/3;
					h_Bnew[row][col] = (h_B[row+1][col]+h_B[row-1][col]+h_B[row][col-1])/3;
				}
				else if (row==0 &&col==0){
					h_Rnew[row][col] = (h_R[row][col+1]+h_R[row+1][col])/2;
					h_Gnew[row][col] = (h_G[row][col+1]+h_G[row+1][col])/2;
					h_Bnew[row][col] = (h_B[row][col+1]+h_B[row+1][col])/2;
				}
				else if (row==0 &&col==(colsize-1)){
					h_Rnew[row][col] = (h_R[row][col-1]+h_R[row+1][col])/2;
					h_Gnew[row][col] = (h_G[row][col-1]+h_G[row+1][col])/2;
					h_Bnew[row][col] = (h_B[row][col-1]+h_B[row+1][col])/2;
				}
				else if (row==(rowsize-1) &&col==0){
					h_Rnew[row][col] = (h_R[row][col+1]+h_R[row-1][col])/2;
					h_Gnew[row][col] = (h_G[row][col+1]+h_G[row-1][col])/2;
					h_Bnew[row][col] = (h_B[row][col+1]+h_B[row-1][col])/2;
				}
				else if (row==(rowsize-1) &&col==(colsize-1)){
					h_Rnew[row][col] = (h_R[row][col-1]+h_R[row-1][col])/2;
					h_Gnew[row][col] = (h_G[row][col-1]+h_G[row-1][col])/2;
					h_Bnew[row][col] = (h_B[row][col-1]+h_B[row-1][col])/2;
				}		
			}
		}
}
__global__
void doCopyKernel(int *h_Rnew, int *h_R,int *h_Gnew, int *h_G,int *h_Bnew, int *h_B,int colsize, int rowsize)
{
	int row = 0, col = 0;
    for(row=0;row<rowsize;row++){
			for (col=0;col<colsize;col++){
			    h_R[row][col] = h_Rnew[row][col];
			    h_G[row][col] = h_Gnew[row][col];
			    h_B[row][col] = h_Bnew[row][col];
			}
		}
}


int main (int argc, const char * argv[]) {
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	int R[rowsize][colsize], G[rowsize][colsize], B[rowsize][colsize];
	int Rnew[rowsize][colsize], Gnew[rowsize][colsize], Bnew[rowsize][colsize];
	int row = 0, col = 0, nblurs, lineno=0, k;
	struct timeval tim;
	
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
				if (row < rowsize) {
					R[row][col] = h1;
					G[row][col] = h2;
					B[row][col] = h3;
				}
				col++;
			}
		}
	}
	fclose(fp);


	fout= fopen("DavidBlur.ps", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%02x%02x%02x",R[row][col],G[row][col],B[row][col]);
			lineno++;
			if (lineno==linelen){
				fprintf(fout,"\n");
				lineno = 0;
			}
		}
	}
	fclose(fout);
	doBlur (Rnew, R, Gnew, G,Bnew, B,colsize,rowsize);
    return 0;
}
	
	int doBlur (int *Rnew, int *R,int *Gnew, int *G,int *Bnew, int *B,int colsize, int rowsize) {
	nblurs = 10;
	printf("%i\n",nblurs);
	gettimeofday(&tim, NULL);

	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	int *h_R[rowsize][colsize], *h_G[rowsize][colsize], *h_B[rowsize][colsize];
	int *h_Rnew[rowsize][colsize], *h_Gnew[rowsize][colsize], *h_Bnew[rowsize][colsize];

	h_R = (int **)malloc((sizeof(int*)*(myrowsize)));
	h_G = (int **)malloc((sizeof(int*)*(myrowsize)));
	h_B = (int **)malloc((sizeof(int*)*(myrowsize)));


	// // memset(h_R, 0, sizeof h_R);
	// // memset(h_G, 0, sizeof h_G);
	// // memset(h_B, 0, sizeof h_B);


	h_Rnew = (int **)malloc((sizeof(int*)*(myrowsize)));
	h_Gnew = (int **)malloc((sizeof(int*)*(myrowsize)));
	h_Bnew = (int **)malloc((sizeof(int*)*(myrowsize)));

    
 //    memset(h_R, 0, sizeof h_R);
	// memset(h_G, 0, sizeof h_G);
	// memset(h_B, 0, sizeof h_B);

	int sizef = sizeof(int)*colsize*rowsize;
    cudaMalloc((void **)&h_Rnew,sizef);
    cudaMalloc((void **)&h_R,sizef);

	cudaMalloc((void **)&h_Gnew,sizef);
    cudaMalloc((void **)&h_G,sizef);

	cudaMalloc((void **)&h_Bnew,sizef);
    cudaMalloc((void **)&h_B,sizef);

    cudaMemcpy(h_R,R,sizef,cudaMemcpyHostToDevice);
    cudaMemcpy(h_G,G,sizef,cudaMemcpyHostToDevice);
    cudaMemcpy(h_B,B,sizef,cudaMemcpyHostToDevice);


    for(k=0;k<nblurs;k++){
		performUpdatesKernel(h_Rnew,h_R,h_Gnew,h_G,h_Bnew,h_B,colsize,rowsize);
        doCopyKernel(h_Rnew,h_R,h_Gnew,h_G,h_Bnew,h_B,colsize,rowsize);
		
	}
    cudaMemcpy(Rnew,h_R,sizef,cudaMemcpyDeviceToHost);
    cudaMemcpy(Gnew,h_G,sizef,cudaMemcpyDeviceToHost);
    cudaMemcpy(Bnew,h_B,sizef,cudaMemcpyDeviceToHost);
    cudaFree(h_Rnew); cudaFree(h_R);
    cudaFree(h_Gnew); cudaFree(h_G);
    cudaFree(h_Bnew); cudaFree(h_B);

	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed\n", t2-t1);
}

