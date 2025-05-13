#include "m_pd.h"

/**
 * notes:
 * - phasor_implementation.md
 * - optimization_techniques_for_dsp_code.md
 **/

// only handles relu and linear
// see optimization_techniques_for_dsp_code.md for other approaches
// note: this is a macro; expansion happens in the context within which it's
// called, so it will have access to `leak`
// z will get replaces with the value that's passed to the macro
// note: this should probably be defined within the layer_forward function
#define HANDLE_ACTIVATION(z) \
  (is_relu ? ((z > 0) ? z : z * leak) : z)

typedef enum {
  ACTIVATION_LINEAR,
  ACTIVATION_RELU
} t_activation_type;

typedef struct _layer {
  int l_n;
  int l_n_prev;
  t_activation_type l_activation;

  t_float *l_weights;
  t_float *l_dw;
  t_float *l_v_dw;
  t_float *l_s_dw;
  t_float *l_biases;
  t_float *l_db;
  t_float *l_v_db;
  t_float *l_s_db;
  t_float *l_z_cache;
  t_float *l_dz;
  t_float *l_a_cache;
  t_float *l_da;
} t_layer;

typedef struct _nnlfo {
  t_object x_obj;
} t_nnlfo;
