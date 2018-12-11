#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#include <math.h>

void doBlur(int h_Rnew,int h_R,int colsize,int rowsize){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

		if(col<colsize && row<rowsize){
				if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)){
					h_Rnew[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[(row-1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])/4;
				}
				else if (row == 0 && col != 0 && col != (colsize-1)){
					h_Rnew[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])/3;
					
				}
				else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
					h_Rnew[row * colsize + col] = (h_R[(row-1) * colsize + col]+h_R[row * colsize + (col+1)]+h_R[row * colsize + (col-1)])/3;
					
				}
				else if (col == 0 && row != 0 && row != (rowsize-1)){
					h_Rnew[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[(row-1) * colsize + col]+h_R[row * colsize + (col+1)])/3;
					
				}
				else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
					h_Rnew[row * colsize + col] = (h_R[(row+1) * colsize + col]+h_R[(row-1) * colsize + col]+h_R[row * colsize + (col-1)])/3;
					
				}
				else if (row==0 &&col==0){
					h_Rnew[row * colsize + col] = (h_R[row * colsize + (col+1)]+h_R[(row+1) * colsize + col])/2;
					
				}
				else if (row==0 &&col==(colsize-1)){
					h_Rnew[row * colsize + col] = (h_R[row * colsize + (col-1)]+h_R[(row+1) * colsize + col])/2;
					
				}
				else if (row==(rowsize-1) &&col==0){
					h_Rnew[row * colsize + col] = (h_R[row * colsize + (col+1)]+h_R[(row-1) * colsize + col])/2;
					
				}
				else if (row==(rowsize-1) &&col==(colsize-1)){
					h_Rnew[row * colsize + col] = (h_R[row * colsize + (col-1)]+h_R[(row-1) * colsize + col])/2;
					
				}		
			}
		
}

void doCopy(int h_Rnew,int h_R,int colsize,int rowsize){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
	if(col<colsize && row<rowsize){
	    h_R[row * colsize + col] = h_Rnew[row * colsize + col];

	}

	}

int RGBval(int x){

	int r,g,b, pow8 = 256;
    if(x<=0.5){
        b = (int)((1.0-2.0*x)*255.0);
        g = (int)(2.0*x*255.0);
		r = 0; 
    }
    else{
        b = 0;
        g = (int)((2.0-2.0*x)*255.0);
        r = (int)((2.0*x-1.0)*255.0);
    }
    return (b+(g+r*pow8)*pow8);
}

int main (int argc, const char * argv[]) {
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	int *R, *G, *B;
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
	
	nblurs = 10;
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	int *h_R;
   	int *h_Rnew;
   	int sizei; 
   	sizei = sizeof(int)*colsize*rowsize;

	h_R = (int*)malloc(sizei);
	h_Rnew = (int*)malloc(sizei);

	memset(h_R, 0, sizeof h_R);

	cudaMalloc((void **)&h_Rnew,sizei);
	cudaMalloc((void **)&h_R,sizei);
	cudaMemcpy(h_R,R,sizei,cudaMemcpyHostToDevice);
   	
	for(k=0;k<nblurs;k++){

		doBlur(h_Rnew,h_R,colsize,rowsize);
        doCopy(h_Rnew,h_R,colsize,rowsize);
		
	}


    cudaMemcpy(R,h_R,sizei,cudaMemcpyDeviceToHost);
	cudaFree(h_Rnew); cudaFree(h_R);


	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed\n", t2-t1);
	
	fout= fopen("DavidBlur.ps", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%06x",RGBval(h_R[row*colsize+col]));
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
