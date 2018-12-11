#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#define NSTEPS 500
#define TX 16
#define TY 32
#define NPTSX 200
#define NPTSY 200

__global__ 
void performUpdatesKernel(float *d_phi, float *d_oldphi, int *d_mask, int nptsx, int nptsy)
{
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    int x = Row*nptsx+Col;
    int xm = x-nptsx;
    int xp = x+nptsx;

    if(Col<nptsx && Row<nptsy)
        if (d_mask[x]) d_phi[x] = 0.25f*(d_oldphi[x+1]+d_oldphi[x-1]+d_oldphi[xp]+d_oldphi[xm]);
}
__global__
void doCopyKernel(float *d_phi, float *d_oldphi, int *d_mask, int nptsx, int nptsy)
{
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    int x = Row*nptsx+Col;

    if(Col<nptsx && Row<nptsy)
        if (d_mask[x]) d_oldphi[x] = d_phi[x];
}

void performUpdates(float *h_phi, float * h_oldphi, int *h_mask, int nptsx, int nptsy, int nsteps)
{
    float *d_phi, *d_oldphi;
    int *d_mask;
    int k;
    int sizef = sizeof(float)*nptsx*nptsy;
    int sizei = sizeof(int)*nptsx*nptsy;
    cudaMalloc((void **)&d_phi,sizef);
    cudaMalloc((void **)&d_oldphi,sizef);
    cudaMalloc((void **)&d_mask,sizei);
    cudaMemcpy(d_oldphi,h_oldphi,sizef,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,h_mask,sizei,cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(nptsx/(float)TX),ceil(nptsy/(float)TY),1);
    dim3 dimBlock(TX,TY,1);
    for(k=0;k<nsteps;++k){
        performUpdatesKernel<<<dimGrid,dimBlock>>>(d_phi,d_oldphi,d_mask,nptsx,nptsy);
        doCopyKernel<<<dimGrid,dimBlock>>>(d_phi,d_oldphi,d_mask,nptsx,nptsy);
    } 
    cudaMemcpy(h_phi,d_oldphi,sizef,cudaMemcpyDeviceToHost);
    cudaFree(d_phi); cudaFree(d_oldphi); cudaFree(d_mask);
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
	
	nblurs = 10;
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	for(k=0;k<nblurs;k++){
		__global__
		for(row=0;row<rowsize;row++){
			for (col=0;col<colsize;col++){	
				if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)){
					Rnew[row][col] = (R[row+1][col]+R[row-1][col]+R[row][col+1]+R[row][col-1])/4;
					Gnew[row][col] = (G[row+1][col]+G[row-1][col]+G[row][col+1]+G[row][col-1])/4;
					Bnew[row][col] = (B[row+1][col]+B[row-1][col]+B[row][col+1]+B[row][col-1])/4;
				}
				else if (row == 0 && col != 0 && col != (colsize-1)){
					Rnew[row][col] = (R[row+1][col]+R[row][col+1]+R[row][col-1])/3;
					Gnew[row][col] = (G[row+1][col]+G[row][col+1]+G[row][col-1])/3;
					Bnew[row][col] = (B[row+1][col]+B[row][col+1]+B[row][col-1])/3;
				}
				else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
					Rnew[row][col] = (R[row-1][col]+R[row][col+1]+R[row][col-1])/3;
					Gnew[row][col] = (G[row-1][col]+G[row][col+1]+G[row][col-1])/3;
					Bnew[row][col] = (B[row-1][col]+B[row][col+1]+B[row][col-1])/3;
				}
				else if (col == 0 && row != 0 && row != (rowsize-1)){
					Rnew[row][col] = (R[row+1][col]+R[row-1][col]+R[row][col+1])/3;
					Gnew[row][col] = (G[row+1][col]+G[row-1][col]+G[row][col+1])/3;
					Bnew[row][col] = (B[row+1][col]+B[row-1][col]+B[row][col+1])/3;
				}
				else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
					Rnew[row][col] = (R[row+1][col]+R[row-1][col]+R[row][col-1])/3;
					Gnew[row][col] = (G[row+1][col]+G[row-1][col]+G[row][col-1])/3;
					Bnew[row][col] = (B[row+1][col]+B[row-1][col]+B[row][col-1])/3;
				}
				else if (row==0 &&col==0){
					Rnew[row][col] = (R[row][col+1]+R[row+1][col])/2;
					Gnew[row][col] = (G[row][col+1]+G[row+1][col])/2;
					Bnew[row][col] = (B[row][col+1]+B[row+1][col])/2;
				}
				else if (row==0 &&col==(colsize-1)){
					Rnew[row][col] = (R[row][col-1]+R[row+1][col])/2;
					Gnew[row][col] = (G[row][col-1]+G[row+1][col])/2;
					Bnew[row][col] = (B[row][col-1]+B[row+1][col])/2;
				}
				else if (row==(rowsize-1) &&col==0){
					Rnew[row][col] = (R[row][col+1]+R[row-1][col])/2;
					Gnew[row][col] = (G[row][col+1]+G[row-1][col])/2;
					Bnew[row][col] = (B[row][col+1]+B[row-1][col])/2;
				}
				else if (row==(rowsize-1) &&col==(colsize-1)){
					Rnew[row][col] = (R[row][col-1]+R[row-1][col])/2;
					Gnew[row][col] = (G[row][col-1]+G[row-1][col])/2;
					Bnew[row][col] = (B[row][col-1]+B[row-1][col])/2;
				}		
			}
		}
		for(row=0;row<rowsize;row++){
			for (col=0;col<colsize;col++){
			    R[row][col] = Rnew[row][col];
			    G[row][col] = Gnew[row][col];
			    B[row][col] = Bnew[row][col];
			}
		}
	}
	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed\n", t2-t1);
	
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
    return 0;
}
