// function that performs relu activation on ndarray
mp_obj_t relu(const mp_obj_t x);

// function that performs reluN activation on ndarray, where N is upper bound
// relu6 is the most widely used, so 6 is the default value for N
mp_obj_t reluN(size_t n_args, const mp_obj_t *args);

// function that performs softmax activation on integer ndarray. It is assumed that max ndarray value <= INT_MAX.
// returns fixed point ndarray with dtype uint16. max value is 10000, which represents 1.
// move decimal over between 0 and 2 places to get confidence percent that is displayed.
mp_obj_t softmax(const mp_obj_t x);

// function that converts softmax output to percent confidence in range [0.00, 100.00]
// Results are returned as a tuple of two integers: (integer part, fraction part)
// Q 7.7 fixed point input is expected
mp_obj_t confidence(const mp_obj_t x);

MP_DECLARE_CONST_FUN_OBJ_1(relu_obj);
MP_DECLARE_CONST_FUN_OBJ_VAR_BETWEEN(reluN_obj);
MP_DECLARE_CONST_FUN_OBJ_1(softmax_obj);
MP_DECLARE_CONST_FUN_OBJ_1(confidence_obj);