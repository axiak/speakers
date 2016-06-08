#pragma once

#include <fftw3.h>

#include "common.h"

typedef struct {
    FFTW_COMPLEX * restrict data;
    int length;
} Vector;

void Vector_print(Vector * vector);
Vector * Vector_create(int length);
void Vector_destroy(Vector * vector);

/* This function assumes that the imaginary part of one of these
   operands is 0.
*/
void Vector_multiply_real(Vector * target, const Vector * operand1, const Vector * operand2);

void Vector_multiply(Vector * target, const Vector * operand1, const Vector * operand2);

void Vector_set_real(Vector * target, const NUMERIC * source, int offset, int length);



void Vector_set(Vector * target, const NUMERIC * source, int offset, int length);

void Vector_zero(Vector * vector);
