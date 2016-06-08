#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <fftw3.h>

#include "vector.h"

Vector * Vector_create(int length)
{
    Vector * vector = (Vector *)malloc(sizeof(Vector));
    if (!vector) {
        return NULL;
    }
    FFTW_COMPLEX * data = (FFTW_COMPLEX*)FFTW_MALLOC(sizeof(FFTW_COMPLEX) * length);
    if (!data) {
        Vector_destroy(vector);
        return NULL;
    }
    vector->data = data;
    vector->length = length;
    return vector;
}

void Vector_print(Vector * vector)
{
    printf("Vector{");
    for (int i = 0; i < vector->length - 1; ++i) {
        printf("%f+%fI, ", vector->data[i][0], vector->data[i][1]);
    }
    if (vector->length) {
        printf("%f+%fI", vector->data[vector->length - 1][0], vector->data[vector->length - 1][1]);
    }
    printf("}\n");
}

void Vector_zero(Vector * vector)
{
    memset(vector->data, 0, sizeof(FFTW_COMPLEX) * vector->length);
}


void Vector_destroy(Vector * vector)
{
  if (vector) {
      if (vector->data) {
          FFTW_FREE(vector->data);
      }
      free(vector);
  }
}

void Vector_multiply_real(Vector * target, const Vector * operand1, const Vector * operand2)
{
    int length = target->length > operand1->length ? operand1->length : target->length;
    length = length * 2;

    NUMERIC * restrict target_numeric = (NUMERIC *)target->data;
    NUMERIC * restrict operand1_numeric = (NUMERIC *)operand1->data;
    NUMERIC * restrict operand2_numeric = (NUMERIC *)operand2->data;

    // It turns out using -mfpu=neon is good enough here to use
    // the arm8 intrinsics
    for (int i = 0; i < length; ++i) {
        target_numeric[i] = operand1_numeric[i] * operand2_numeric[i];
    }
}

void Vector_multiply(Vector * target, const Vector * operand1, const Vector * operand2)
{
    int length = target->length > operand1->length ? operand1->length : target->length;

    FFTW_COMPLEX * restrict target_numeric = (FFTW_COMPLEX *)target->data;
    FFTW_COMPLEX * restrict operand1_numeric = (FFTW_COMPLEX *)operand1->data;
    FFTW_COMPLEX * restrict operand2_numeric = (FFTW_COMPLEX *)operand2->data;

    register float k1, k2, k3;
    for (int i = 0; i < length; ++i) {
        k1 = operand1_numeric[i][0] * (operand2_numeric[i][0] + operand2_numeric[i][1]);
        k2 = operand2_numeric[i][1] * (operand1_numeric[i][0] + operand1_numeric[i][1]);
        k3 = operand2_numeric[i][0] * (operand1_numeric[i][1] - operand1_numeric[i][0]);

        target_numeric[i][0] = k1 - k2;
        target_numeric[i][1] = k1 + k3;
    }

}


void Vector_set(Vector * target, const NUMERIC * source, int offset, int length)
{
    int end = offset + length;
    NUMERIC * restrict target_numeric = (NUMERIC *)target->data;
    for (int j = 0, i = offset; i < end; ++i, ++j) {
        target_numeric[j] = source[i];
    }
}

void Vector_set_real(Vector * target, const NUMERIC * source, int offset, int length)
{
    int end = offset + length;
    NUMERIC * restrict target_numeric = (NUMERIC *)target->data;
    for (int j = 0, i = offset; i < end; ++i, ++j) {
        target_numeric[j << 1] = source[i];
    }
}


#ifdef VECTOR_MAIN

int main(int argc, char ** argv)
{
  Vector * vector_a = Vector_create(1024);

  for (int i = 0; i < vector_a->length; ++i) {
    vector_a->data[i][0] = i;
    vector_a->data[i][1] = i;
  }

  Vector * vector_b = Vector_create(1024);
  Vector * vector_c = Vector_create(1024);

  for (int i = 0; i < vector_b->length; ++i) {
    vector_b->data[i][0] = i + 1;
  }


  struct timeval stop, start;
  gettimeofday(&start, NULL);
  for (int i = 0; i < 100000; ++i) {
    Vector_multiply_real(vector_c, vector_a, vector_b);
  }
  gettimeofday(&stop, NULL);
  printf("took %lu\n", stop.tv_usec - start.tv_usec);

  Vector * vector_d = Vector_create(2);
  Vector * vector_e = Vector_create(2);
  float d_data[] = {1, 1, 2, 2};
  float e_data[] = {3, -1, 1, 1};

  for (int i = 0; i < 4; ++i) {
      vector_d->data[i / 2][i % 2] = d_data[i];
      vector_e->data[i / 2][i % 2] = e_data[i];
  }

  Vector_print(vector_d);
  Vector_print(vector_e);

  Vector_multiply(vector_d, vector_d, vector_e);

  Vector_print(vector_d);


  Vector_destroy(vector_a);
  Vector_destroy(vector_b);
  Vector_destroy(vector_c);
  Vector_destroy(vector_d);
  Vector_destroy(vector_e);

}

#endif
