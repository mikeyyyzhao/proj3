#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    //Flip the kernal here//

    int xsize;
    int ysize;
    int temp;
    for (ysize = 0; ysize<=KERNY; ysize++){
    	for (xsize = 0; xsize<kern_cent_X; xsize++){
    		temp = kernel[xsize][ysize];
    		kernel[xsize][ysize] = kernel[KERNX-1-xsize][ysize];
    		kernel[KERNY-1-xsize][ysize] = temp;
    	}
    }
    for (xsize = 0; xsize<=KERNX; xsize++){
    	for (ysize = 0; ysize<kern_cent_Y; ysize++){
    		temp = kernel[xsize][ysize];
    		kernel[xsize][ysize] = kernel[xsize][KERNY-1-ysize];
    		kernel[xsize][KERNY-1-ysize] = temp;
    	}
    }    
    // main convolution loop
	for(int x = 0; x < data_size_X; x+=KERNX){ // the x coordinate of the output location we're focusing on
		for(int y = 0; y < data_size_Y; y+=KERNY){ // the y coordinate of theoutput location we're focusing on
			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel flipped
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel is now flipped
					// only do the operation if not out of bounds
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
						out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
					}
				}
			}
		}
	}
	return 1;
}
