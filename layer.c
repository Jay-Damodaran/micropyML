#include "py/obj.h"
#include "py/runtime.h"

#include "ulab.h"
#include "ndarray.h"
#include "layer.h"

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DROPOUT_LOOP(type) \
    type *arr = (type *) arr_pt->array; \
    type *out_arr = (type *) out_arr_pt->array; \
    for (size_t i = 0; i < arr_pt->len; i++) {  \
        out_arr[i] = ((float) rand() / RAND_MAX) < rate ? 0 : arr[i]; \
    }

mp_obj_t dropout(size_t n_args, const mp_obj_t *args){
    // type check to ensure input is ndarray followed by float
    float rate = 0.5;
    if(!mp_obj_is_type(args[0], &ulab_ndarray_type)){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray as first argument"));
    }
    if(n_args == 2 && !mp_obj_is_float(args[1])){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected type float for dropout rate"));
    }
    else if(n_args == 2)
        rate = mp_obj_get_float(args[1]);
    srand(time(NULL));
    ndarray_obj_t *arr_pt = MP_OBJ_TO_PTR(args[0]);
    uint8_t dtype = arr_pt->dtype;
    ndarray_obj_t *out_arr_pt = ndarray_new_dense_ndarray(arr_pt->ndim, arr_pt->shape, dtype);
    switch(dtype){
        case NDARRAY_FLOAT: {
            DROPOUT_LOOP(float)
            break;
        }
        case NDARRAY_INT16: {
            DROPOUT_LOOP(int16_t)
            break;
        }
        case NDARRAY_INT8: {
            DROPOUT_LOOP(int8_t)
            break;
        }
        case NDARRAY_UINT16: {
            DROPOUT_LOOP(uint16_t)
            break;
        }
        case NDARRAY_UINT8: {
            DROPOUT_LOOP(uint8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Unrecognized ndarray dtype"));
            break;
        }
    }
    return MP_OBJ_FROM_PTR(out_arr_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(dropout_obj, 1, 2, dropout);

#define MAXPOOL1D_LOOP(type) \
    type *arr = (type *) arr_pt->array; \
    type *out_arr = (type *) out_arr_pt->array;\
    type max = MIN_FOR_TYPE(type);       \
    uint32_t ref = 0;        \
    for(uint32_t i = 0; i < (arr_pt->len); i++){ \
        if(arr[i] > max) max = arr[i]; \
        if((i+1-ref) % kernel == 0){ \
            out_arr[(i-ref)/kernel] = max; \
            max = MIN_FOR_TYPE(type); \
        } \
        if((i+1-ref) % (result_shape[2] * kernel) == 0){ \
            ref += (uint32_t)((arr_pt->shape)[2] - result_shape[2] * kernel); \
            i += (uint32_t)((arr_pt->shape)[2] - result_shape[2] * kernel); \
        } \
    }

mp_obj_t maxpool1d(size_t n_args, const mp_obj_t *args){
    // preconditions ensuring valid ndarray input
    mp_obj_t x = args[0];
    if(!mp_obj_is_type(x, &ulab_ndarray_type)){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray as first argument"));
    }
    ndarray_obj_t *arr_pt = MP_OBJ_TO_PTR(x);
    if(arr_pt->ndim != 3) mp_raise_ValueError(MP_ERROR_TEXT("Expected 3D array of form (batch, channels, time steps)"));

    //preconditions to ensure valid kernel input
    int32_t kernel;
    if(n_args == 1){
        kernel = 2;
    }
    else if(!mp_obj_is_int(args[1])){
        mp_raise_TypeError(MP_ERROR_TEXT("Expected integer kernel size"));
    }
    else{
        kernel = mp_obj_get_int(args[1]);
    }
    // extract shape from arr_pt
    size_t result_shape[3]; // Batch, channels, time-points
    result_shape[0] = (arr_pt->shape)[0];
    result_shape[1] = (arr_pt->shape)[1];
    if(kernel <= 0 || (size_t)kernel > (arr_pt->shape)[2]) mp_raise_ValueError(MP_ERROR_TEXT("Expected 0 < kernel size <= dim 2"));
    result_shape[2] = (arr_pt->shape)[2] / kernel;

    ndarray_obj_t *out_arr_pt = ndarray_new_dense_ndarray(3, result_shape, arr_pt->dtype);
    switch(arr_pt->dtype){
        case NDARRAY_FLOAT: {
            MAXPOOL1D_LOOP(float)
            break;
        }
        case NDARRAY_INT16: {
            MAXPOOL1D_LOOP(int16_t)
            break;
        }
        case NDARRAY_INT8: {
            MAXPOOL1D_LOOP(int8_t)
            break;
        }
        case NDARRAY_UINT16: {
            MAXPOOL1D_LOOP(uint16_t)
            break;
        }
        case NDARRAY_UINT8: {
            MAXPOOL1D_LOOP(uint8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Unrecognized or unsupported ndarray dtype"));
            break;
        }
    }
    return MP_OBJ_FROM_PTR(out_arr_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(maxpool1d_obj, 1, 2, maxpool1d);


static inline int32_t requantize(int32_t accum, int32_t multiplier, uint8_t shift, int32_t zero_pt){
    int64_t product = (int64_t)accum * (int64_t)multiplier;
    if(product >= 0)
        product += ((int64_t)1 << (30 + shift));
    else
        product = -(-product + ((int64_t)1 << (30 + shift)));
    product >>= (31 + shift);
    return (int32_t)product + zero_pt; // saturating cast into lower precision range is calling function's responsibility
}
#define CONVBODY(type) \
    type *x_arr = (type *)x_pt->array; \
    type *kernel_arr = (type *)kernel_pt->array; \
    int32_t *bias_arr = NULL;                   \
    if(bias_pt){               \
        bias_arr = (int32_t *)bias_pt->array; \
    } \
    type *result_arr = (type *)result_pt->array; \
    int32_t dot_prod = 0;              \
    for(int32_t b = 0; b < x_pt->shape[0]; b++) {   \
        for(int32_t f = 0; f < kernel_pt->shape[0]; f++) {  \
            result_index = (b * kernel_pt->shape[0] + f) * channel_ref; \
            for(int32_t n = 0; n < result_pt->shape[2]; n++){           \
                dot_prod = bias_pt ? bias_arr[f] : 0;   \
                for(int32_t c = 0; c < x_pt->shape[1]; c++){\
                    x_index = b * batch_ref + c * channel_ref;      \
                    kernel_index = f * filter_ref + c * kernel_ref; \
                    for(int32_t k = 0; k < kernel_pt->shape[2]; k++){   \
                        xk = n + k - pad; \
                        if (xk < 0 || xk >= (int32_t)x_pt->shape[2]) continue; \
                        dot_prod += (int32_t)x_arr[x_index + xk] * (int32_t)kernel_arr[kernel_index + k];     \
                    }           \
                }               \
                dot_prod = requantize(dot_prod, m0, shift, output_zp); \
                dot_prod = dot_prod < output_zp ? output_zp : dot_prod; /*applying relu activation*/      \
                dot_prod = dot_prod > MAX_FOR_TYPE(type) ? MAX_FOR_TYPE(type) : dot_prod; \
                dot_prod = dot_prod < MIN_FOR_TYPE(type) ? MIN_FOR_TYPE(type) : dot_prod;            \
                result_arr[result_index + n] = (type)dot_prod;           \
            }                        \
        } \
    }

/* for(int32_t c = 0; c < x_pt->shape[1]; c++) { \
                x_index = b * batch_ref + c * channel_ref; \
                kernel_index = f * filter_ref + c * kernel_ref; \
                for (int32_t k = 0; k < result_pt->shape[2]; k++) { \
                    for (int32_t i = 0; i < kernel_pt->shape[2]; i++) { \
                        if ((i + k - pad) < 0 || (i + k - pad) >= (int32_t)x_pt->shape[2]) continue; \
                        result_arr[result_index + k] += x_arr[x_index + i + k - pad] * kernel_arr[kernel_index + i]; \
                    } \
                } \
            }    put a \ here if you paste it back */
mp_obj_t qconvrelu1d(size_t n_args, const mp_obj_t *args){
    ndarray_obj_t *bias_pt = NULL;
    mp_obj_t m0_obj;
    mp_obj_t shift_obj;
    mp_obj_t output_zp_obj;
    int32_t m0 = 1;
    int32_t output_zp = 0;
    uint8_t shift = 0;

    if(!mp_obj_is_type(args[0], &ulab_ndarray_type) || !mp_obj_is_type(args[1], &ulab_ndarray_type))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarrays for input and kernel arguments"));
    ndarray_obj_t *x_pt = MP_OBJ_TO_PTR(args[0]);
    ndarray_obj_t *kernel_pt = MP_OBJ_TO_PTR(args[1]);
    if(x_pt->ndim != 3 || kernel_pt->ndim != 3)
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3D array for input and kernel"));
    if(n_args == 4){
        if(!mp_obj_is_type(args[2], &ulab_ndarray_type))
            mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray for bias argument"));
        bias_pt = MP_OBJ_TO_PTR(args[2]);
        if(bias_pt->ndim != 1)
            mp_raise_ValueError(MP_ERROR_TEXT("Expected 1D array for bias"));
        m0_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(0), MP_OBJ_SENTINEL);
        shift_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(1), MP_OBJ_SENTINEL);
        output_zp_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(2), MP_OBJ_SENTINEL);
        if(!(mp_obj_is_int(m0_obj) && mp_obj_is_int(shift_obj) && mp_obj_is_int(output_zp_obj)))
            mp_raise_TypeError(MP_ERROR_TEXT("Expected integers for quantization params"));
        m0 = mp_obj_get_int(m0_obj);
        shift = mp_obj_get_int(shift_obj);
        output_zp = mp_obj_get_int(output_zp_obj);
    }
    else if(n_args == 3){
        m0_obj = mp_obj_subscr(args[2], MP_OBJ_NEW_SMALL_INT(0), MP_OBJ_SENTINEL);
        shift_obj = mp_obj_subscr(args[2], MP_OBJ_NEW_SMALL_INT(1), MP_OBJ_SENTINEL);
        output_zp_obj = mp_obj_subscr(args[3], MP_OBJ_NEW_SMALL_INT(2), MP_OBJ_SENTINEL);
        if(!(mp_obj_is_int(m0_obj) && mp_obj_is_int(shift_obj) && mp_obj_is_int(output_zp_obj)))
            mp_raise_TypeError(MP_ERROR_TEXT("Expected integers for quantization params"));
        m0 = mp_obj_get_int(m0_obj);
        shift = mp_obj_get_int(shift_obj);
        output_zp = mp_obj_get_int(output_zp_obj);
    }
    size_t result_shape[3];
    result_shape[0] = x_pt->shape[0];
    result_shape[1] = kernel_pt->shape[0];
    result_shape[2] = x_pt->shape[2];
    ndarray_obj_t *result_pt = ndarray_new_dense_ndarray(3, result_shape, x_pt->dtype);

    int32_t pad = (int32_t)(kernel_pt->shape[2] - 1) / 2; // padding formula for stride 1 and no dilation
    // inner loop for same padding
    int32_t channel_ref = x_pt->shape[2]; // how much index increments with each channel (N)
    int32_t kernel_ref = kernel_pt->shape[2];
    int32_t batch_ref = x_pt->shape[1] * channel_ref; // how much to increment index for each filter or batch (C)
    int32_t filter_ref = kernel_pt->shape[1] * kernel_pt->shape[2];
    int32_t x_index, kernel_index, result_index, xk;

    switch (x_pt->dtype) {
        case NDARRAY_INT16: {
            CONVBODY(int16_t)
            break;
        }
        case NDARRAY_INT8: {
            CONVBODY(int8_t)
            break;
        }
        case NDARRAY_UINT16: {
            CONVBODY(uint16_t)
            break;
        }
        case NDARRAY_UINT8: {
            CONVBODY(uint8_t)
            break;
        }
        default: {
            mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray dtype of 8 or 16 bit uint/int"));
            break;
        }
    }
    // inner loop for valid padding
    //    for(size_t k = 0; k <= (x_pt->shape[2]-kernel_pt->shape[2]); k++) {
    //        for (size_t i = 0; i < kernel_pt->shape[2]; i++) {
    //            result_arr[k] += x_arr[][][i+k] * kernel_arr[][][i];
    //        }
    //    }
    return MP_OBJ_FROM_PTR(result_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(qconvrelu1d_obj, 2, 4, qconvrelu1d);

mp_obj_t conv1d(size_t n_args, const mp_obj_t *args){
    ndarray_obj_t *bias_pt = NULL;

    if(!mp_obj_is_type(args[0], &ulab_ndarray_type) || !mp_obj_is_type(args[1], &ulab_ndarray_type))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarrays for input and kernel arguments"));
    ndarray_obj_t *x_pt = MP_OBJ_TO_PTR(args[0]);
    ndarray_obj_t *kernel_pt = MP_OBJ_TO_PTR(args[1]);
    if(x_pt->dtype != NDARRAY_FLOAT || kernel_pt->dtype != NDARRAY_FLOAT)
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarrays of dtype float"));
    if(x_pt->ndim != 3 || kernel_pt->ndim != 3)
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3D array for input and kernel"));
    if(n_args == 3 && !mp_obj_is_type(args[2], &ulab_ndarray_type))
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ndarray for bias argument"));
    else if(n_args == 3){
        bias_pt = MP_OBJ_TO_PTR(args[2]);
        if(bias_pt->ndim != 1) mp_raise_ValueError(MP_ERROR_TEXT("Expected 1D array for bias"));
        if(bias_pt->dtype != NDARRAY_FLOAT) mp_raise_TypeError(MP_ERROR_TEXT("Expected bias of dtype float"));
    }
    size_t result_shape[3];
    result_shape[0] = x_pt->shape[0];
    result_shape[1] = kernel_pt->shape[0];
    result_shape[2] = x_pt->shape[2];
    ndarray_obj_t *result_pt = ndarray_new_dense_ndarray(3, result_shape, x_pt->dtype);

    int32_t pad = (int32_t)(kernel_pt->shape[2] - 1) / 2; // padding formula for stride 1 and no dilation
    // inner loop for same padding
    int32_t channel_ref = x_pt->shape[2]; // how much index increments with each channel (N)
    int32_t kernel_ref = kernel_pt->shape[2];
    int32_t batch_ref = x_pt->shape[1] * channel_ref; // how much to increment index for each filter or batch (C)
    int32_t filter_ref = kernel_pt->shape[1] * kernel_pt->shape[2];
    int32_t x_index, kernel_index, result_index, xk;

    float *x_arr = (float *)x_pt->array;
    float *kernel_arr = (float *)kernel_pt->array;
    float *bias_arr = NULL;

    if (bias_pt) {
        bias_arr = (float *)bias_pt->array;
    }

    float *result_arr = (float *)result_pt->array;
    float dot_prod = 0;

    for(int32_t b = 0; b < x_pt->shape[0]; b++) {
        for(int32_t f = 0; f < kernel_pt->shape[0]; f++) {
            result_index = (b * kernel_pt->shape[0] + f) * channel_ref;
            for(int32_t n = 0; n < result_pt->shape[2]; n++) {
                dot_prod = bias_pt ? bias_arr[f] : 0;
                for(int32_t c = 0; c < x_pt->shape[1]; c++) {
                    x_index = b * batch_ref + c * channel_ref;
                    kernel_index = f * filter_ref + c * kernel_ref;
                    for(int32_t k = 0; k < kernel_pt->shape[2]; k++) {
                        xk = n + k - pad;
                        if(xk < 0 || xk >= (int32_t)x_pt->shape[2]) {
                            continue;
                        }
                        dot_prod += x_arr[x_index + xk] * kernel_arr[kernel_index + k];
                    }
                }
                result_arr[result_index + n] = dot_prod;
            }
        }
    }
    return MP_OBJ_FROM_PTR(result_pt);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(conv1d_obj, 2, 3, conv1d);