#include "m_pd.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

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
  ACTIVATION_RELU,
  ACTIVATION_SIGMOID,
  ACTIVATION_TANH
} t_activation_type;

typedef struct _layer {
  int l_n;
  int l_n_prev;
  t_activation_type l_activation;

  t_float *l_weights;
  t_float *l_dw;
  t_float *l_biases;
  t_float *l_db;
  t_float *l_z_cache;
  t_float *l_dz;
  t_float *l_a_cache;
  t_float *l_da;
} t_layer;

typedef struct _nnpulse2 {
  t_object x_obj;

  t_layer *x_layers;
  int *x_layer_dims;
  int x_num_layers;

  // t_float x_conv; // 1.0 / sample rate
  double x_conv;
  double x_features_conv;

  t_float x_example_freq;
  // t_float x_example_phase;
  double x_example_phase;
  t_float x_example_pw;

  t_float x_label_freq;
  // t_float x_label_phase;
  double x_label_phase;
  t_float x_label_pw;
  t_float x_current_label;

  int x_num_features;
  int x_feature_samps; // int for now
  t_float *x_input_features;

  t_float x_leak;
  t_float x_alpha;

  t_inlet *x_example_freq_inlet;
  t_inlet *x_label_freq_inlet;
  t_outlet *x_prediction_outlet;
  // t_outlet *x_label_outlet;
} t_nnpulse2;

static t_class *nnpulse2_class = NULL;
static void initialize_layers(t_nnpulse2 *x);
static t_float apply_activation(t_nnpulse2 *x, t_layer *layer, t_float z);
static t_float activation_derivative(t_activation_type activation, t_float z, t_float a, t_float leak);
static void init_layer_weights(t_nnpulse2 *x, int l);
static void init_layer_biases(t_nnpulse2 *x, int l);

static void *nnpulse2_new(void) {
  t_nnpulse2 *x = (t_nnpulse2 *)pd_new(nnpulse2_class);
  x->x_input_features = NULL;
  x->x_layers = NULL;
  x->x_layer_dims = NULL;

  x->x_num_layers = 6;
  x->x_layer_dims = (int *)getbytes(sizeof(int) * x->x_num_layers + 1);
  if (!x->x_layer_dims) {
    pd_error(x, "nnpulse2~: failed to allocate memory for layer_dims");
    return NULL;
  }
  x->x_num_features = 64;

  // hardcoded for now:
  x->x_layer_dims[0] = x->x_num_features; // input layer
  x->x_layer_dims[1] = 3 * x->x_num_features;
  x->x_layer_dims[2] = x->x_num_features;
  x->x_layer_dims[3] = 32;
  x->x_layer_dims[4] = 32;
  x->x_layer_dims[5] = 16;
  x->x_layer_dims[6] = 1; // output layer

  // x->x_conv = (t_float)0.0;
  x->x_conv = (double)0.0;
  x->x_features_conv = (double)0.0;

  x->x_example_freq = (t_float)220.0;
  // x->x_example_phase = (t_float)0.0;
  x->x_example_phase = (double)0.0;
  x->x_example_pw = (t_float)0.5;

  x->x_label_freq = (t_float)440.0;
  // x->x_label_phase = (t_float)0.0;
  x->x_label_phase = (double)0.0;
  x->x_label_pw = (t_float)0.5;
  x->x_current_label = (t_float)0.0;

  x->x_leak = (t_float)0.001;
  x->x_alpha = (t_float)0.0001;

  x->x_input_features = getbytes(sizeof(t_float) * x->x_num_features);
  if (!x->x_input_features) {
    pd_error(x, "nnpulse2~: failed to allocate memory for input_features");
    return NULL;
  }

  x->x_feature_samps = 0;

  static int seed_initialized = 0;
  if (!seed_initialized) {
    srand((unsigned int)time(NULL));
    seed_initialized = 1;
  }

  initialize_layers(x);

  x->x_example_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  pd_float((t_pd *)x->x_example_freq_inlet, x->x_example_freq);
  x->x_label_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  // I don't think this works
  pd_float((t_pd *)x->x_label_freq_inlet, x->x_label_freq);

  x->x_prediction_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void nnpulse2_free(t_nnpulse2 *x) {
  if (x->x_example_freq_inlet) inlet_free(x->x_example_freq_inlet);
  if (x->x_prediction_outlet) outlet_free(x->x_prediction_outlet);

  if (x->x_layer_dims) {
    freebytes(x->x_layer_dims, sizeof(int) * (x->x_num_layers + 1));
  }

  if (x->x_layers) {
    for (int l = 0; l < x->x_num_layers; l++) {
      t_layer *layer = &x->x_layers[l];
      if (layer->l_weights) freebytes(layer->l_weights, sizeof(t_float) * layer->l_n * layer->l_n_prev);
      if (layer->l_dw) freebytes(layer->l_dw, sizeof(t_float) * layer->l_n * layer->l_n_prev);
      if (layer->l_biases) freebytes(layer->l_biases, sizeof(t_float) * layer->l_n);
      if (layer->l_db) freebytes(layer->l_db, sizeof(t_float) * layer->l_n);
      if (layer->l_z_cache) freebytes(layer->l_z_cache, sizeof(t_float) * layer->l_n);
      if (layer->l_dz) freebytes(layer->l_dz, sizeof(t_float) * layer->l_n);
      if (layer->l_a_cache) freebytes(layer->l_a_cache, sizeof(t_float) * layer->l_n);
      if (layer->l_da) freebytes(layer->l_da, sizeof(t_float) * layer->l_n);
    }
    freebytes(x->x_layers, sizeof(t_layer) * x->x_num_layers);
  }

  if (x->x_input_features) freebytes(x->x_input_features, sizeof(t_float) * x->x_num_features);
}

static inline void populate_features(t_nnpulse2 *x,
                                     t_float example_freq,
                                     double current_phase) {
  double features_conv = x->x_features_conv;
  t_float pw = x->x_example_pw; // hardcoded for now
  int num_features = x->x_num_features;
  t_float *features_buffer = x->x_input_features;

  for (int i = 0; i < num_features; i++) {
    features_buffer[i] = (current_phase < pw) ? (t_float)1.0 : (t_float)-1.0;

    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
  }
}

static void layer_forward(t_nnpulse2 *x, t_layer *layer, t_float *input_buffer) {
  int n = layer->l_n;
  int n_prev = layer->l_n_prev;
  t_float leak = x->x_leak;

  // pre-compute activation function selection
  int is_relu = (layer->l_activation == ACTIVATION_RELU);

  int n_prev_unroll_limit = n_prev - 8;

  for (int i = 0; i < n; i++) {
    t_float z = layer->l_biases[i];
    // prefetch next bias
    // __builtin_prefetch args:
    // 1: pointer to memory address
    // 2: read/write access (0 read, 1 write)
    // 3: locality hint (0 - 3), 3 indicates high locality
    // locality seems to be intended to indicate how often the data will be
    // accessed?
    if (i + 1 < n) __builtin_prefetch(&layer->l_biases[i+1], 0, 3);

    // unroll by 8 (there's a technical reason for 8)
    int k = 0;
    for (; k <= n_prev_unroll_limit; k += 8) {
      __builtin_prefetch(&layer->l_weights[i*n_prev+k+8], 0, 3);
      z += layer->l_weights[i*n_prev+k] * input_buffer[k] +
           layer->l_weights[i*n_prev+k+1] * input_buffer[k+1] +
           layer->l_weights[i*n_prev+k+2] * input_buffer[k+2] +
           layer->l_weights[i*n_prev+k+3] * input_buffer[k+3] +
           layer->l_weights[i*n_prev+k+4] * input_buffer[k+4] +
           layer->l_weights[i*n_prev+k+5] * input_buffer[k+5] +
           layer->l_weights[i*n_prev+k+6] * input_buffer[k+6] +
           layer->l_weights[i*n_prev+k+7] * input_buffer[k+7];

    }

    // handle remaining elements
    for (; k < n_prev; k++) {
      z += layer->l_weights[i*n_prev+k] * input_buffer[k];
    }

    layer->l_z_cache[i] = z;

    layer->l_a_cache[i] = HANDLE_ACTIVATION(z);
  }
}

static void model_forward(t_nnpulse2 *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    t_float *input = (l == 0) ? x->x_input_features : x->x_layers[l-1].l_a_cache;
    layer_forward(x, layer, input);
  }
}

static void calculate_dz(t_nnpulse2 *x, t_layer *layer) {
  int is_relu = layer->l_activation == ACTIVATION_RELU;
  int n = layer->l_n;
  t_float leak = x->x_leak;
  int n_unroll_limit = n - 8;
  
  #define ACTIVATION_DERIVATIVE(z) \
    (is_relu ? ((z > 0) ? 1.0f : leak) : 1.0f)
  
  int i = 0;
  for (; i <= n_unroll_limit; i += 8) {
    if (i + 8 < n) {
      __builtin_prefetch(&layer->l_da[i+8], 0, 3);
      __builtin_prefetch(&layer->l_z_cache[i+8], 0, 3);
    }
    
    layer->l_dz[i] = layer->l_da[i] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i]);
    layer->l_dz[i+1] = layer->l_da[i+1] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i+1]);
    layer->l_dz[i+2] = layer->l_da[i+2] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i+2]);
    layer->l_dz[i+3] = layer->l_da[i+3] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i+3]);
    layer->l_dz[i+4] = layer->l_da[i+4] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i+4]);
    layer->l_dz[i+5] = layer->l_da[i+5] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i+5]);
    layer->l_dz[i+6] = layer->l_da[i+6] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i+6]);
    layer->l_dz[i+7] = layer->l_da[i+7] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i+7]);
  }
  
  // handle remaining elements (or layers where n < 8)
  for (; i < n; i++) {
    layer->l_dz[i] = layer->l_da[i] * ACTIVATION_DERIVATIVE(layer->l_z_cache[i]);
  }
  
  #undef ACTIVATION_DERIVATIVE
}

static void calculate_db(t_nnpulse2 *x, int l, t_layer *layer) {
  int n = layer->l_n;
  
  // special case for output layer
  if (l == x->x_num_layers - 1 && n == 1) {
    // db = dz
    layer->l_db[0] = layer->l_dz[0];
    return;
  }

  int n_unroll_limit = n - 8;

  int i = 0;
  #pragma GCC ivdep
  for (; i <= n_unroll_limit; i += 8) {
    if (i + 8 < n) {
      __builtin_prefetch(&layer->l_dz[i+8], 0, 3);
    }

    layer->l_db[i]   = layer->l_dz[i];
    layer->l_db[i+1] = layer->l_dz[i+1];
    layer->l_db[i+2] = layer->l_dz[i+2];
    layer->l_db[i+3] = layer->l_dz[i+3];
    layer->l_db[i+4] = layer->l_dz[i+4];
    layer->l_db[i+5] = layer->l_dz[i+5];
    layer->l_db[i+6] = layer->l_dz[i+6];
    layer->l_db[i+7] = layer->l_dz[i+7];
  }
  
  for (; i < n; i++) {
    layer->l_db[i] = layer->l_dz[i];
  }
}

static void calculate_da_prev(t_nnpulse2 *x, int l, t_layer *layer) {
  int n = layer->l_n;
  int n_prev = layer->l_n_prev;
  t_layer *prev_layer = &x->x_layers[l-1];


  // probably this could be changed to 8
  int n_unroll_limit = n - 4;

  for (int k = 0; k < n_prev; k++) {
    if (k + 1 < n_prev) {
      __builtin_prefetch(&prev_layer->l_da[k+1], 1, 3);  // Note: write access (1)
      
      // Prefetch the next column of weights (scattered across memory)
      for (int p = 0; p < n; p += 16) {
        if (p + 16 < n) {
          __builtin_prefetch(&layer->l_weights[(p+16)*n_prev+k+1], 0, 1);
        }
      }
    }

    t_float sum = (t_float)0.0;

    int i = 0;
    for (; i <= n_unroll_limit; i += 4) {
      if (i + 4 < n) {
        __builtin_prefetch(&layer->l_weights[(i+4)*n_prev+k], 0, 3);
        __builtin_prefetch(&layer->l_dz[i+4], 0, 3);
      }

      sum += layer->l_weights[i*n_prev+k] * layer->l_dz[i] +
             layer->l_weights[(i+1)*n_prev+k] * layer->l_dz[i+1] +
             layer->l_weights[(i+2)*n_prev+k] * layer->l_dz[i+2] +
             layer->l_weights[(i+3)*n_prev+k] * layer->l_dz[i+3];
    }

    for (; i < n; i++) {
      sum += layer->l_weights[i*n_prev+k] * layer->l_dz[i];
    }

    prev_layer->l_da[k] = sum;
  }
}

static void calculate_dw(t_nnpulse2 *x, int l, t_layer *layer) {
  int n = layer->l_n;
  int n_prev = layer->l_n_prev;
  t_float *prev_activations = (l == 0) ? x->x_input_features : x->x_layers[l-1].l_a_cache;

  int n_prev_unroll_limit = n_prev - 8;

  // clear previous gradients
  memset(layer->l_dw, 0, sizeof(t_float) * n * n_prev);

  // outer loop (output neurons)
  for (int i = 0; i < n; i++) {
    // cache the neuron's dz to avoid repeated access
    t_float dz_i = layer->l_dz[i];

    if (i + 1 < n) {
      __builtin_prefetch(&layer->l_dz[i+1], 0, 3);
    }

    // inner loop (input neurons)
    int j = 0;

    #pragma GCC ivdep  // tells the compiler there are no loop-carried dependencies
    for (; j <= n_prev_unroll_limit; j += 8) {
      if (j + 8 < n_prev) {
        __builtin_prefetch(&prev_activations[j+8], 0, 3);
      }

      int idx = i * n_prev + j;
      layer->l_dw[idx]     = dz_i * prev_activations[j];
      layer->l_dw[idx + 1] = dz_i * prev_activations[j + 1];
      layer->l_dw[idx + 2] = dz_i * prev_activations[j + 2];
      layer->l_dw[idx + 3] = dz_i * prev_activations[j + 3];
      layer->l_dw[idx + 4] = dz_i * prev_activations[j + 4];
      layer->l_dw[idx + 5] = dz_i * prev_activations[j + 5];
      layer->l_dw[idx + 6] = dz_i * prev_activations[j + 6];
      layer->l_dw[idx + 7] = dz_i * prev_activations[j + 7];
    }

    // handle any remaining weights
    for (; j < n_prev; j++) {
      layer->l_dw[i * n_prev + j] = dz_i * prev_activations[j];
    }
  }
}

static void layer_backward(t_nnpulse2 *x, int l, t_layer *layer) {
  if (l == x->x_num_layers - 1) {
    // calculate output layer da
    // since the output layer always has 1 neuron
    layer->l_da[0] = layer->l_a_cache[0] - x->x_current_label;
  }

  calculate_dz(x, layer);
  calculate_dw(x, l, layer);
  calculate_db(x, l, layer);

  if (l > 0) calculate_da_prev(x, l, layer);
}

static void model_backward(t_nnpulse2 *x) {
  for (int l = x->x_num_layers - 1; l >= 0; l--) {
    t_layer *layer = &x->x_layers[l];
    layer_backward(x, l, layer);
  }
}

static void update_parameters(t_nnpulse2 *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    int n = layer->l_n;
    int n_prev = layer->l_n_prev;

    for (int i = 0; i < n * n_prev; i++) {
      layer->l_weights[i] -= x->x_alpha * layer->l_dw[i];
    }

    for (int i = 0; i < n; i++) {
      layer->l_biases[i] -= x->x_alpha * layer->l_db[i];
    }
  }
}

static t_int *nnpulse2_perform(t_int *w) {
  t_nnpulse2 *x = (t_nnpulse2 *)(w[1]);
  t_sample *example_freq_in = (t_sample *)(w[2]);
  t_sample *label_freq_in = (t_sample *)(w[3]);
  t_sample *pred_out = (t_sample *)(w[4]);
  int n = (int)(w[5]);

  double conv = x->x_conv;

  double example_phase = x->x_example_phase;
  double label_phase = x->x_label_phase;
  t_float label_pw = x->x_label_pw;
  t_float example_pw = x->x_example_pw;

  int output_layer = x->x_num_layers - 1;

  t_float label = x->x_current_label;

  while (n--) {
    t_sample example_freq = *example_freq_in++;
    t_sample label_freq = *label_freq_in++;
    t_sample y_hat = (t_sample)0.0;

    populate_features(x, example_freq, example_phase);
    model_forward(x);

    y_hat = x->x_layers[output_layer].l_a_cache[0];

    label = (label_phase < label_pw) ? (t_float)1.0 : (t_float)-1.0;

    *pred_out++ = y_hat;

    example_phase += example_freq * conv;
    example_phase -= floor(example_phase);
    label_phase += label_freq * conv;
    label_phase -= floor(label_phase);
  }

  model_backward(x);
  update_parameters(x);

  x->x_current_label = label;
  x->x_example_phase = example_phase;
  x->x_label_phase = label_phase;
  return (w+6);
}

static void nnpulse2_dsp(t_nnpulse2 *x, t_signal **sp) {
  x->x_conv = (double)1.0 / (double)sp[0]->s_sr;
  x->x_feature_samps = (int)(sp[0]->s_sr / x->x_num_features);
  x->x_features_conv = x->x_conv * x->x_feature_samps;
  dsp_add(nnpulse2_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_length);
}

static void model_reset(t_nnpulse2 *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    int n = layer->l_n;
    int n_prev = layer->l_n_prev;
    int w_size = n * n_prev;

    memset(layer->l_z_cache, 0, sizeof(t_float) * n);
    memset(layer->l_dz, 0, sizeof(t_float) * n);
    memset(layer->l_a_cache, 0, sizeof(t_float) * n);
    memset(layer->l_da, 0, sizeof(t_float) * n);
    memset(layer->l_weights, 0, sizeof(t_float) * w_size);
    memset(layer->l_dw, 0, sizeof(t_float) * w_size);
    memset(layer->l_biases, 0, sizeof(t_float) * n);
    memset(layer->l_db, 0, sizeof(t_float) * n);

    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }
}

static void set_alpha(t_nnpulse2 *x, t_floatarg f) {
  x->x_alpha = f;
}

static void set_leak(t_nnpulse2 *x, t_floatarg f) {
  x->x_leak = f;
}

void nnpulse2_tilde_setup(void) {
  nnpulse2_class = class_new(gensym("nnpulse2~"),
                            (t_newmethod)nnpulse2_new,
                            (t_method)nnpulse2_free,
                            sizeof(t_nnpulse2),
                            CLASS_DEFAULT,
                            0);
  class_addmethod(nnpulse2_class, (t_method)nnpulse2_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(nnpulse2_class, (t_method)model_reset, gensym("reset"), 0);
  class_addmethod(nnpulse2_class, (t_method)set_alpha, gensym("alpha"), A_FLOAT, 0);
  class_addmethod(nnpulse2_class, (t_method)set_leak, gensym("leak"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(nnpulse2_class, t_nnpulse2, x_example_freq);
}

float he_init(int n_prev) {
  t_float u1 = (t_float)rand() / RAND_MAX;
  t_float u2 = (t_float)rand() / RAND_MAX;
  t_float radius = (t_float)(sqrt(-2 * log(u1)));
  t_float theta = 2 * (t_float)M_PI * u2;
  t_float standard_normal = radius * (t_float)(cos(theta));

  return standard_normal * (t_float)(sqrt(2.0 / n_prev));
}

static void init_layer_weights(t_nnpulse2 *x, int l) {
  t_layer *layer = &x->x_layers[l];
  int size = layer->l_n * layer->l_n_prev;
  for (int i = 0; i < size; i++) {
    t_float weight = he_init(layer->l_n_prev);
    layer->l_weights[i] = weight;
  }
}

static void init_layer_biases(t_nnpulse2 *x, int l) {
  t_layer *layer = &x->x_layers[l];
  for (int i = 0; i < layer->l_n; i++) {
    // not technically needed
    layer->l_biases[i] = (t_float)0.0;
  }
}

static void initialize_layers(t_nnpulse2 *x) {
  x->x_layers = (t_layer *)getbytes(sizeof(t_layer) * x->x_num_layers);
  if (!x->x_layers) {
    pd_error(x, "nnpulse2~: failed to allocate memory for layers");
    return;
  }

  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    layer->l_n = x->x_layer_dims[l+1];
    layer->l_n_prev = x->x_layer_dims[l];

    if (l < x->x_num_layers - 1) {
      layer->l_activation = ACTIVATION_RELU;
    } else {
      layer->l_activation = ACTIVATION_LINEAR; // output layer
    }

    layer->l_weights = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_weights) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer weights");
      return;
    }
    layer->l_dw = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_dw) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer dw");
      return;
    }
    layer->l_biases = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_biases) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer biases");
      return;
    }
    layer->l_db = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_db) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer db");
      return;
    }
    layer->l_z_cache = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_z_cache) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer z_cache");
      return;
    }
    layer->l_dz = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_dz) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer dz");
      return;
    }
    layer->l_a_cache = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_a_cache) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer a_cache");
      return;
    }
    layer->l_da = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_da) {
      pd_error(x, "nnpulse2~: failed to allocate memory for layer da");
      return;
    }

    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }
}

static inline float fast_tanh(float x) {
     float x2 = 2.0f * x;
     return x2 / (1.0f + fabsf(x2));
 }

static t_float apply_activation(t_nnpulse2 *x, t_layer *layer, t_float z) {
  switch(layer->l_activation) {
    case ACTIVATION_SIGMOID:
      return 1.0 / (1.0 + exp(-z));
    case ACTIVATION_TANH:
      return fast_tanh(z);
      // return tanh(z);
    case ACTIVATION_RELU:
      return z > 0 ? z : z * x->x_leak;
    case ACTIVATION_LINEAR:
    default:
      return z;
  }
}

static t_float activation_derivative(t_activation_type activation,
                                     t_float z,
                                     t_float a,
                                     t_float leak) {
  switch(activation) {
    case ACTIVATION_SIGMOID:
      return a * (1.0 - a);
    case ACTIVATION_TANH:
      return 1.0 - a * a;
    case ACTIVATION_RELU:
      return z > 0 ? 1.0 : leak;
    case ACTIVATION_LINEAR:
    default:
      return 1.0;
  }
}
