
/***gcc -o libpycall.so -shared -fPIC pycall.c*/  
#include <stdio.h>  
#include <stdlib.h>  

int foo(int a)  
{  
  printf("you input %d\n", a);  
  return a+a;
}
