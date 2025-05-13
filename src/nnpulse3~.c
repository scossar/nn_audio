#include "m_pd.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

/**
 * prediction triggered by input edges
 * notes:
 * - phasor_implementation.md
 * - optimization_techniques_for_dsp_code.md
 * - audio_neural_network_optimization_methods.md
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

typedef struct _nnpulse3 {
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
  t_float x_previous_example;

  t_float x_label_freq;
  // t_float x_label_phase;
  double x_label_phase;
  t_float x_label_pw;
  t_float x_current_label;
  t_float x_previous_label;

  t_float x_prev_example_pulse;
  t_float x_prev_label_pulse;

  int x_num_features;
  int x_feature_samps; // int for now
  t_float *x_input_features;
  t_sample x_y_hat;

  t_float x_leak;
  t_float x_alpha;
  t_float x_lambda;
  t_float x_beta_1;
  t_float x_beta_2;

  t_inlet *x_example_freq_inlet;
  t_inlet *x_label_freq_inlet;
  t_outlet *x_prediction_outlet;
  t_outlet *x_example_bang_out, *x_label_bang_out;
  // t_outlet *x_label_outlet;
} t_nnpulse3;

static t_class *nnpulse3_class = NULL;
static void initialize_layers(t_nnpulse3 *x);
// static t_float apply_activation(t_nnpulse3 *x, t_layer *layer, t_float z);
// static t_float activation_derivative(t_activation_type activation, t_float z, t_float a, t_float leak);
static void init_layer_weights(t_nnpulse3 *x, int l);
static void init_layer_biases(t_nnpulse3 *x, int l);

static void *nnpulse3_new(void) {
  t_nnpulse3 *x = (t_nnpulse3 *)pd_new(nnpulse3_class);
  x->x_input_features = NULL;
  x->x_layers = NULL;
  x->x_layer_dims = NULL;

  x->x_num_layers = 7;
  x->x_layer_dims = (int *)getbytes(sizeof(int) * x->x_num_layers + 1);
  if (!x->x_layer_dims) {
    pd_error(x, "nnpulse3~: failed to allocate memory for layer_dims");
    return NULL;
  }
  x->x_num_features = 64;

  // hardcoded for now:
  x->x_layer_dims[0] = x->x_num_features; // input layer
  x->x_layer_dims[1] = 2 * x->x_num_features;
  x->x_layer_dims[2] = 2 * x->x_num_features;
  x->x_layer_dims[3] = 64;
  x->x_layer_dims[4] = 64;
  x->x_layer_dims[5] = 32;
  x->x_layer_dims[6] = 32;
  x->x_layer_dims[7] = 1; // output layer

  // x->x_conv = (t_float)0.0;
  x->x_conv = (double)0.0;
  x->x_features_conv = (double)0.0;

  x->x_example_freq = (t_float)1.0;
  // x->x_example_phase = (t_float)0.0;
  x->x_example_phase = (double)0.0;
  x->x_example_pw = (t_float)0.5;
  x->x_previous_example = (t_float)0.0f;

  x->x_label_freq = (t_float)1.0;
  // x->x_label_phase = (t_float)0.0;
  x->x_label_phase = (double)0.0;
  x->x_label_pw = (t_float)0.5;
  x->x_previous_label = (t_float)0.0f;
  x->x_current_label = (t_float)0.0;

  x->x_prev_example_pulse = (t_float)0.0;
  x->x_prev_label_pulse = (t_float)0.0;

  x->x_y_hat = (t_sample)0.0;

  x->x_leak = (t_float)0.001;
  x->x_alpha = (t_float)0.0001;
  x->x_lambda = (t_float)0.0001;
  x->x_beta_1 = (t_float)0.9;
  x->x_beta_2 = (t_float)0.999;

  x->x_input_features = getbytes(sizeof(t_float) * x->x_num_features);
  if (!x->x_input_features) {
    pd_error(x, "nnpulse3~: failed to allocate memory for input_features");
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
  x->x_example_bang_out = outlet_new(&x->x_obj, &s_bang);
  x->x_label_bang_out = outlet_new(&x->x_obj, &s_bang);

  return (void *)x;
}

static void nnpulse3_free(t_nnpulse3 *x) {
  if (x->x_example_freq_inlet) inlet_free(x->x_example_freq_inlet);
  if (x->x_prediction_outlet) outlet_free(x->x_prediction_outlet);
  if (x->x_example_bang_out) outlet_free(x->x_example_bang_out);
  if (x->x_label_bang_out) outlet_free(x->x_label_bang_out);

  if (x->x_layer_dims) {
    freebytes(x->x_layer_dims, sizeof(int) * (x->x_num_layers + 1));
  }

  if (x->x_layers) {
    for (int l = 0; l < x->x_num_layers; l++) {
      t_layer *layer = &x->x_layers[l];
      if (layer->l_weights) freebytes(layer->l_weights, sizeof(t_float) * layer->l_n * layer->l_n_prev);
      if (layer->l_dw) freebytes(layer->l_dw, sizeof(t_float) * layer->l_n * layer->l_n_prev);
      if (layer->l_v_dw) freebytes(layer->l_v_dw, sizeof(t_float) * layer->l_n * layer->l_n_prev);
      if (layer->l_s_dw) freebytes(layer->l_s_dw, sizeof(t_float) * layer->l_n * layer->l_n_prev);
      if (layer->l_biases) freebytes(layer->l_biases, sizeof(t_float) * layer->l_n);
      if (layer->l_db) freebytes(layer->l_db, sizeof(t_float) * layer->l_n);
      if (layer->l_v_db) freebytes(layer->l_v_db, sizeof(t_float) * layer->l_n);
      if (layer->l_s_db) freebytes(layer->l_s_db, sizeof(t_float) * layer->l_n);
      if (layer->l_z_cache) freebytes(layer->l_z_cache, sizeof(t_float) * layer->l_n);
      if (layer->l_dz) freebytes(layer->l_dz, sizeof(t_float) * layer->l_n);
      if (layer->l_a_cache) freebytes(layer->l_a_cache, sizeof(t_float) * layer->l_n);
      if (layer->l_da) freebytes(layer->l_da, sizeof(t_float) * layer->l_n);
    }
    freebytes(x->x_layers, sizeof(t_layer) * x->x_num_layers);
  }

  if (x->x_input_features) freebytes(x->x_input_features, sizeof(t_float) * x->x_num_features);
}

static inline void populate_features(t_nnpulse3 *x,
                                     t_float example_freq,
                                     double current_phase) {
  double features_conv = x->x_features_conv;
  // t_float pw = x->x_example_pw; // hardcoded for now
  int num_features = x->x_num_features;
  t_float *features_buffer = x->x_input_features;

  for (int i = 0; i < num_features; i++) {
    // features_buffer[i] = (current_phase < pw) ? (t_float)1.0 : (t_float)-1.0;
    features_buffer[i] = current_phase;

    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
  }
}

// see audio_neural_network_optimizations.md for more optimizations
#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void populate_features_optimized_v1(t_nnpulse3 *x,
                                        t_float example_freq,
                                        double current_phase) {
  double features_conv = x->x_features_conv;
  int num_features = x->x_num_features;
  t_float pw = x->x_example_pw;
  int features_unroll_limit = num_features - 8;
  t_float *features_buffer = x->x_input_features;

  int i = 0;
  #pragma GCC ivdep
  for (; i <= features_unroll_limit; i += 8) {
    // features_buffer[i] = current_phase;
    // give every 8th feature current pulse state
    features_buffer[i] = (current_phase < pw) ? (t_float)1.0 : (t_float)0.0;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
    features_buffer[i+1] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
    features_buffer[i+2] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
    features_buffer[i+3] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
    features_buffer[i+4] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
    features_buffer[i+5] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
    features_buffer[i+6] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
    features_buffer[i+7] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
  }
  for (; i < num_features; i++) {
    features_buffer[i] = current_phase;
    current_phase += example_freq * features_conv;
    current_phase -= floor(current_phase);
  }
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void layer_forward(t_nnpulse3 *x, t_layer *layer, t_float *input_buffer) {
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

static void model_forward(t_nnpulse3 *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    t_float *input = (l == 0) ? x->x_input_features : x->x_layers[l-1].l_a_cache;
    layer_forward(x, layer, input);
  }
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_dz(t_nnpulse3 *x, t_layer *layer) {
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

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_db(t_nnpulse3 *x, int l, t_layer *layer) {
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

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_da_prev(t_nnpulse3 *x, int l, t_layer *layer) {
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

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_dw(t_nnpulse3 *x, int l, t_layer *layer) {
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

static void layer_backward(t_nnpulse3 *x, int l, t_layer *layer) {
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

static void model_backward(t_nnpulse3 *x) {
  for (int l = x->x_num_layers - 1; l >= 0; l--) {
    t_layer *layer = &x->x_layers[l];
    layer_backward(x, l, layer);
  }
}

static void update_parameters(t_nnpulse3 *x) {
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

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void update_parameters_optimized(t_nnpulse3 *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    if (l + 1 < x->x_num_layers) {
      __builtin_prefetch(&x->x_layers[l+1], 0, 3);
    }

    int n = layer->l_n;
    int n_prev = layer->l_n_prev;
    int num_weights = n * n_prev;
    int weights_unroll_limit = num_weights - 8;
   int biases_unroll_limit = n - 8;
    t_float alpha = x->x_alpha;
    t_float weight_decay_factor = (t_float)1.0 - alpha * x->x_lambda;

    int j = 0;
    #pragma GCC ivdep
    for (; j <= weights_unroll_limit; j += 8) {
      if (j + 8 < num_weights) {
        __builtin_prefetch(&layer->l_weights[j+8], 1, 3);
        __builtin_prefetch(&layer->l_db[j+8], 0, 3);
      }
      layer->l_weights[j] = weight_decay_factor * layer->l_weights[j] - alpha * layer->l_dw[j];
      layer->l_weights[j+1] = weight_decay_factor * layer->l_weights[j+1] - alpha * layer->l_dw[j+1];
      layer->l_weights[j+2] = weight_decay_factor * layer->l_weights[j+2] - alpha * layer->l_dw[j+2];
      layer->l_weights[j+3] = weight_decay_factor * layer->l_weights[j+3] - alpha * layer->l_dw[j+3];
      layer->l_weights[j+4] = weight_decay_factor * layer->l_weights[j+4] - alpha * layer->l_dw[j+4];
      layer->l_weights[j+5] = weight_decay_factor * layer->l_weights[j+5] - alpha * layer->l_dw[j+5];
      layer->l_weights[j+6] = weight_decay_factor * layer->l_weights[j+6] - alpha * layer->l_dw[j+6];
      layer->l_weights[j+7] = weight_decay_factor * layer->l_weights[j+7] - alpha * layer->l_dw[j+7];
    }

    for (; j < num_weights; j++) {
      layer->l_weights[j] = weight_decay_factor * layer->l_weights[j] - alpha * layer->l_dw[j];
    }

    int k = 0;
    #pragma GCC ivdep
    for (; k <= biases_unroll_limit; k += 8) {
      if (k + 8 < n) {
        __builtin_prefetch(&layer->l_biases[k+8], 1, 3);
        __builtin_prefetch(&layer->l_db[k+8], 0, 3);
      }
      layer->l_biases[k] -= alpha * layer->l_db[k];
      layer->l_biases[k+1] -= alpha * layer->l_db[k+1];
      layer->l_biases[k+2] -= alpha * layer->l_db[k+2];
      layer->l_biases[k+3] -= alpha * layer->l_db[k+3];
      layer->l_biases[k+4] -= alpha * layer->l_db[k+4];
      layer->l_biases[k+5] -= alpha * layer->l_db[k+5];
      layer->l_biases[k+6] -= alpha * layer->l_db[k+6];
      layer->l_biases[k+7] -= alpha * layer->l_db[k+7];
    }

    for (; k < n; k++) {
      layer->l_biases[k] -= alpha * layer->l_db[k];
    }
  }
}

// Fast inverse square root (Quake III algorithm) (modified)
static inline float fast_inv_sqrt(float number) {
    float y = number;
    float x2 = y * 0.5F;
    
    // Use a union for type punning (avoids strict aliasing violations)
    union {
        float f;
        int32_t i;  // Use int32_t for portability
    } conv;
    
    conv.f = y;
    conv.i = 0x5f3759df - (conv.i >> 1);
    y = conv.f;
    
    // One Newton-Raphson iteration
    y = y * (1.5F - (x2 * y * y));
    
    return y;
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void update_parameters_adam(t_nnpulse3 *x) {
    const float beta1 = x->x_beta_1;
    const float beta2 = x->x_beta_2;
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float alpha = x->x_alpha;
    const float epsilon = 1e-8f;
    const float weight_decay_factor = 1.0f - alpha * x->x_lambda;

    for (int l = 0; l < x->x_num_layers; l++) {
        t_layer *layer = &x->x_layers[l];
        if (l + 1 < x->x_num_layers) {
            __builtin_prefetch(&x->x_layers[l+1], 0, 3);
        }

        int n = layer->l_n;
        int n_prev = layer->l_n_prev;
        int num_weights = n * n_prev;
        int weights_unroll_limit = num_weights - 8;
        int biases_unroll_limit = n - 8;

        // Process weights
        int j = 0;
        #pragma GCC ivdep
        for (; j <= weights_unroll_limit; j += 8) {
            if (j + 8 < num_weights) {
                __builtin_prefetch(&layer->l_weights[j+8], 1, 3);
                __builtin_prefetch(&layer->l_dw[j+8], 0, 3);
                __builtin_prefetch(&layer->l_v_dw[j+8], 1, 3);
                __builtin_prefetch(&layer->l_s_dw[j+8], 1, 3);
            }

            // Unrolled loop for 8 elements at a time
            for (int k = 0; k < 8; k++) {
                int idx = j + k;
                float dw = layer->l_dw[idx];

                // Update momentum
                layer->l_v_dw[idx] = beta1 * layer->l_v_dw[idx] + one_minus_beta1 * dw;

                // Update squared gradient accumulation
                layer->l_s_dw[idx] = beta2 * layer->l_s_dw[idx] + one_minus_beta2 * dw * dw;

                // Compute update
                float update = alpha * layer->l_v_dw[idx] * fast_inv_sqrt(layer->l_s_dw[idx] + epsilon);

                // Apply update with weight decay
                layer->l_weights[idx] = weight_decay_factor * layer->l_weights[idx] - update;
            }
        }

        // Handle remaining weights
        for (; j < num_weights; j++) {
            float dw = layer->l_dw[j];

            // Update momentum
            layer->l_v_dw[j] = beta1 * layer->l_v_dw[j] + one_minus_beta1 * dw;

            // Update squared gradient accumulation
            layer->l_s_dw[j] = beta2 * layer->l_s_dw[j] + one_minus_beta2 * dw * dw;

            // Compute update
            float update = alpha * layer->l_v_dw[j] * fast_inv_sqrt(layer->l_s_dw[j] + epsilon);

            // Apply update with weight decay
            layer->l_weights[j] = weight_decay_factor * layer->l_weights[j] - update;
        }

        // Process biases (similar approach)
        int k = 0;
        #pragma GCC ivdep
        for (; k <= biases_unroll_limit; k += 8) {
            if (k + 8 < n) {
                __builtin_prefetch(&layer->l_biases[k+8], 1, 3);
                __builtin_prefetch(&layer->l_db[k+8], 0, 3);
                __builtin_prefetch(&layer->l_v_db[k+8], 1, 3);
                __builtin_prefetch(&layer->l_s_db[k+8], 1, 3);
            }

            // Unrolled loop for biases
            for (int i = 0; i < 8; i++) {
                int idx = k + i;
                float db = layer->l_db[idx];

                // Update momentum
                layer->l_v_db[idx] = beta1 * layer->l_v_db[idx] + one_minus_beta1 * db;

                // Update squared gradient accumulation
                layer->l_s_db[idx] = beta2 * layer->l_s_db[idx] + one_minus_beta2 * db * db;

                // Compute update
                float update = alpha * layer->l_v_db[idx] * fast_inv_sqrt(layer->l_s_db[idx] + epsilon);

                // Apply update (biases typically don't use weight decay)
                layer->l_biases[idx] -= update;
            }
        }

        // Handle remaining biases
        for (; k < n; k++) {
            float db = layer->l_db[k];

            // Update momentum
            layer->l_v_db[k] = beta1 * layer->l_v_db[k] + one_minus_beta1 * db;

            // Update squared gradient accumulation
            layer->l_s_db[k] = beta2 * layer->l_s_db[k] + one_minus_beta2 * db * db;

            // Compute update
            float update = alpha * layer->l_v_db[k] * fast_inv_sqrt(layer->l_s_db[k] + epsilon);

            // Apply update
            layer->l_biases[k] -= update;
        }
    }
}

static t_int *nnpulse3_perform(t_int *w) {
  t_nnpulse3 *x = (t_nnpulse3 *)(w[1]);
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

  // t_float current_label = x->x_current_label;
  t_float previous_label = x->x_previous_label;
  t_float previous_example = x->x_previous_example;

  t_sample y_hat = x->x_y_hat;

  int example_bang = 0;
  int label_bang = 0;

  t_float example_freq = x->x_example_freq;
  t_float label_freq = x->x_label_freq;

  t_float prev_example_pulse = x->x_prev_example_pulse;
  t_float prev_label_pulse = x->x_prev_label_pulse;

  while (n--) {
    example_freq = *example_freq_in++;
    label_freq = *label_freq_in++;
    t_float current_example_pulse = (example_phase < example_pw) ? (t_float)1.0 : (t_float)0.0;
    t_float current_label_pulse = (label_phase < label_pw) ? (t_float)1.0 : (t_float)0.0;

    t_float current_example = example_phase;
    t_float current_label = label_phase;

    if (current_example < previous_example) example_bang = 1;
    if (current_label < previous_label) label_bang = 1;

    if (current_example_pulse != prev_example_pulse || current_label_pulse != prev_label_pulse) {
      x->x_current_label = current_label_pulse;
      populate_features_optimized_v1(x, example_freq, example_phase);
      model_forward(x);
      y_hat = x->x_layers[output_layer].l_a_cache[0];
    }

    *pred_out++ = y_hat;

    example_phase += example_freq * conv;
    example_phase -= floor(example_phase);
    label_phase += label_freq * conv;
    label_phase -= floor(label_phase);
    previous_example = current_example;
    previous_label = current_label;
    prev_example_pulse = current_example_pulse;
    prev_label_pulse = current_label_pulse;
  }

  x->x_current_label = prev_label_pulse;
  populate_features_optimized_v1(x, example_freq, example_phase);
  model_forward(x);
  model_backward(x);
  update_parameters_adam(x);
  x->x_previous_example = previous_example;
  x->x_previous_label = previous_label;
  x->x_prev_example_pulse = prev_example_pulse;
  x->x_prev_label_pulse = prev_label_pulse;
  x->x_y_hat = y_hat;

  x->x_example_phase = example_phase;
  x->x_label_phase = label_phase;

  if (example_bang) outlet_bang(x->x_example_bang_out);
  if (label_bang) outlet_bang(x->x_label_bang_out);

  return (w+6);
}

static void nnpulse3_dsp(t_nnpulse3 *x, t_signal **sp) {
  x->x_conv = (double)1.0 / (double)sp[0]->s_sr;
  x->x_feature_samps = (int)(sp[0]->s_sr / x->x_num_features);
  x->x_features_conv = x->x_conv * x->x_feature_samps;
  dsp_add(nnpulse3_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_length);
}

static void model_reset(t_nnpulse3 *x) {
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
    memset(layer->l_v_dw, 0, sizeof(t_float) * w_size);
    memset(layer->l_s_dw, 0, sizeof(t_float) * w_size);
    memset(layer->l_biases, 0, sizeof(t_float) * n);
    memset(layer->l_db, 0, sizeof(t_float) * n);
    memset(layer->l_v_db, 0, sizeof(t_float) * n);
    memset(layer->l_s_db, 0, sizeof(t_float) * n);

    init_layer_weights(x, l);
    init_layer_biases(x, l);
    x->x_example_phase = (double)0.0;
    x->x_label_phase = (double)0.0;
  }
}

static void set_example_pw(t_nnpulse3 *x, t_floatarg f) {
  f = (f > (t_float)1.0) ? (t_float)1.0 : (f < (t_float)0.0) ? (t_float)0.0 : f;
  x->x_example_pw = f;
}

static void set_label_pw(t_nnpulse3 *x, t_floatarg f) {
  f = (f > (t_float)1.0) ? (t_float)1.0 : (f < (t_float)0.0) ? (t_float)0.0 : f;
  x->x_label_pw = f;
}

static void set_alpha(t_nnpulse3 *x, t_floatarg f) {
  x->x_alpha = f;
}

static void set_leak(t_nnpulse3 *x, t_floatarg f) {
  x->x_leak = f;
}

static void set_lambda(t_nnpulse3 *x, t_floatarg f) {
  x->x_lambda = (f < (t_float)0.0) ? (t_float)0.0 : f;
}

void nnpulse3_tilde_setup(void) {
  nnpulse3_class = class_new(gensym("nnpulse3~"),
                            (t_newmethod)nnpulse3_new,
                            (t_method)nnpulse3_free,
                            sizeof(t_nnpulse3),
                            CLASS_DEFAULT,
                            0);
  class_addmethod(nnpulse3_class, (t_method)nnpulse3_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(nnpulse3_class, (t_method)model_reset, gensym("reset"), 0);
  class_addmethod(nnpulse3_class, (t_method)set_alpha, gensym("alpha"), A_FLOAT, 0);
  class_addmethod(nnpulse3_class, (t_method)set_leak, gensym("leak"), A_FLOAT, 0);
  class_addmethod(nnpulse3_class, (t_method)set_lambda, gensym("lambda"), A_FLOAT, 0);
  class_addmethod(nnpulse3_class, (t_method)set_example_pw, gensym("example_pw"), A_FLOAT, 0);
  class_addmethod(nnpulse3_class, (t_method)set_label_pw, gensym("label_pw"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(nnpulse3_class, t_nnpulse3, x_example_freq);
}

float he_init(int n_prev) {
  t_float u1 = (t_float)rand() / RAND_MAX;
  t_float u2 = (t_float)rand() / RAND_MAX;
  t_float radius = (t_float)(sqrt(-2 * log(u1)));
  t_float theta = 2 * (t_float)M_PI * u2;
  t_float standard_normal = radius * (t_float)(cos(theta));

  return standard_normal * (t_float)(sqrt(2.0 / n_prev));
}

static void init_layer_weights(t_nnpulse3 *x, int l) {
  t_layer *layer = &x->x_layers[l];
  int size = layer->l_n * layer->l_n_prev;
  for (int i = 0; i < size; i++) {
    t_float weight = he_init(layer->l_n_prev);
    layer->l_weights[i] = weight;
  }
}

static void init_layer_biases(t_nnpulse3 *x, int l) {
  t_layer *layer = &x->x_layers[l];
  for (int i = 0; i < layer->l_n; i++) {
    // not technically needed
    layer->l_biases[i] = (t_float)0.0;
  }
}

static void initialize_layers(t_nnpulse3 *x) {
  x->x_layers = (t_layer *)getbytes(sizeof(t_layer) * x->x_num_layers);
  if (!x->x_layers) {
    pd_error(x, "nnpulse3~: failed to allocate memory for layers");
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
      pd_error(x, "nnpulse3~: failed to allocate memory for layer weights");
      return;
    }
    layer->l_dw = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_dw) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer dw");
      return;
    }
    layer->l_v_dw = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_v_dw) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer v_dw");
      return;
    }
    layer->l_s_dw = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_s_dw) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer s_dw");
      return;
    }
    layer->l_biases = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_biases) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer biases");
      return;
    }
    layer->l_db = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_db) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer db");
      return;
    }
    layer->l_v_db = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_v_db) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer v_db");
      return;
    }
    layer->l_s_db = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_s_db) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer s_db");
      return;
    }
    layer->l_z_cache = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_z_cache) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer z_cache");
      return;
    }
    layer->l_dz = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_dz) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer dz");
      return;
    }
    layer->l_a_cache = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_a_cache) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer a_cache");
      return;
    }
    layer->l_da = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_da) {
      pd_error(x, "nnpulse3~: failed to allocate memory for layer da");
      return;
    }

    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }
}
