#include <stdio.h>

int foo(int);
int get_global();
int set_global(int);

int main(){
    printf("%d\n",foo(10));
    printf("%d\n", get_global());
    printf("%d\n", set_global(100));
    printf("%d\n", get_global());
}