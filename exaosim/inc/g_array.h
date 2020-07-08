#define ATYPE float

typedef struct{
    int dim;
    int *shape;
    int size;
    char type;
    ATYPE *data;
    bool gpu_array;
    ATYPE *gdata;
}  ARRAY;

ARRAY* array_zeros(int s_num, ...);
ATYPE array_get(ARRAY* array, ...);
void array_set(ARRAY* array, ...);
ATYPE* array_ref(ARRAY* array, ...);
ARRAY* array_load(FILE *file);
ARRAY* array_load_file(char const* filename);
void array_save(FILE *file, ARRAY *array);
void array_del(ARRAY *array);
ARRAY* array_zeros_like(ARRAY *array);
void array_multiple_c(ARRAY *A, ARRAY *B, ARRAY *C);
void array2device(ARRAY *array);
void array2host(ARRAY *array);
__global__ void array_multi(float *A, float * B, float *C, int Ax, int Ay, 
int Bx, int By, int Cx, int Cy);