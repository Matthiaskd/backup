#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "nn2.h"
#include "file.h"
int mat[784];
void convert(SDL_Surface* surface)
{
    Uint32* pixels = surface->pixels;
    int len = surface->w * surface->h;
    SDL_PixelFormat* format = surface->format;

    SDL_LockSurface(surface);
    for (int i = 0; i < len; i++)
    {
	    Uint8 r, g, b;
	    SDL_GetRGB(pixels[i], format, &r, &g, &b);
	    if (0.3*r + 0.59*g + 0.11*b>255*0.31){
            mat[i] = 0;
            // printf("%i\n", 1);
        }
	        
	    else
	        mat[i] = 1;
    }
    SDL_UnlockSurface(surface);
}	
int main(int argc,char** argv){
    if(argc>26)
        return 1;
    char* path=argv[1];
    SDL_Surface* surf=IMG_Load(path);
    
    convert(surf);
    for (int i =0; i<784; i++)
        printf("%i", mat[i]);
    return 0;
}