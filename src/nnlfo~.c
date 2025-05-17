#include "m_pd.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

/**
 * notes:
 * - phasor_implementation.md
 * - optimization_techniques_for_dsp_code.md
 * - nnlfo_phase_relationships.md
 * - flattened_array_intuitions.md
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

typedef enum {
  PHASE_LINEAR,
  PHASE_LINEAR_INVERSE,
  PHASE_BINARY,
} t_phase_representation;

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

  t_layer *x_layers;
  int x_num_layers; // hidden + output layers
  int x_num_features;
  int *x_layer_dims; // input, hidden, and output layer dimensions

  t_phase_representation x_phase_rep;

  t_float x_f;
  double x_conv;

  t_float x_example_freq;
  double x_features_conv;
  double x_example_phase;
  double x_example_pw;
  t_float x_previous_example;
  t_float x_previous_example_pulse;

  int x_batch_size;
  int x_batch_index;
  int x_features_filled;

  t_float *x_batch_labels;
  t_float x_batch_scale;

  t_float x_label_freq;
  // double x_label_conv;
  double x_label_phase;
  double x_label_pw;
  t_float x_current_label;
  t_float x_previous_label;
  t_float x_previous_label_pulse;

  int x_feature_samps; // int for now
  t_float *x_input_features;
  t_float x_features_reciprocal;
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
} t_nnlfo;

static t_class *nnlfo_class = NULL;

static void set_layer_dims(t_nnlfo *x);
static void initialize_layers(t_nnlfo *x);
static void nnlfo_free(t_nnlfo *x);
static void init_layer_weights(t_nnlfo *x, int l);
static void init_layer_biases(t_nnlfo *x, int l);

static void *nnlfo_new(void) {
  t_nnlfo *x = (t_nnlfo *)pd_new(nnlfo_class);

  x->x_layers = NULL;
  x->x_batch_labels = NULL;
  x->x_num_layers = 7;
  x->x_num_features = 64; // hardcoded for now
  x->x_features_reciprocal = (t_float)((t_float)1.0 / (t_float)x->x_num_features);
  x->x_phase_rep = PHASE_LINEAR;

  x->x_layer_dims = (int *)getbytes(sizeof(int) * x->x_num_layers + 1);
  if (!x->x_layer_dims) {
    pd_error(x, "nnlfo~: failed to allocate memory for layer_dims");
    return NULL;
  }
  set_layer_dims(x);

  x->x_f = (t_float)1.0;
  x->x_conv = (double)0.0;

  x->x_example_freq = (t_float)1.0;
  x->x_features_conv = (double)0.0;
  x->x_example_phase = (double)0.0;
  x->x_example_pw = (double)0.5;
  x->x_previous_example = (t_float)0.0;
  x->x_previous_example_pulse = (t_float)0.0;

  x->x_batch_size = 8; // would ideally be set from x_example_freq
  x->x_batch_scale = (t_float)1.0 / (t_float)x->x_batch_size;
  x->x_batch_labels = getbytes(sizeof(t_float) * x->x_batch_size);
  if (!x->x_batch_labels) {
    pd_error(x, "nnlfo~: failed to allocate memory for batch_labels");
  }
  x->x_batch_index = 0;
  x->x_features_filled = 0;

  x->x_label_freq = (t_float)1.0;
  x->x_label_phase = (double)0.0;
  x->x_label_pw = (double)0.5;
  x->x_current_label = (t_float)0.0;
  x->x_previous_label = (t_float)0.0;
  x->x_previous_label_pulse = (t_float)0.0;

  x->x_feature_samps = 0;

  x->x_input_features = getbytes(sizeof(t_float) * x->x_num_features * x->x_batch_size);
  if (!x->x_input_features) {
    pd_error(x, "nnlfo~: failed to allocate memory for input_features");
    return NULL;
  }

  static int seed_initialized = 0;
  if (!seed_initialized) {
    srand((unsigned int)time(NULL));
    seed_initialized = 1;
  }
  initialize_layers(x);

  x->x_y_hat = (t_sample)0.0;

  x->x_leak = (t_float)0.001;
  x->x_alpha = (t_float)0.0001;
  x->x_lambda = (t_float)0.0001;
  x->x_beta_1 = (t_float)0.9;
  x->x_beta_2 = (t_float)0.999;

  x->x_example_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  pd_float((t_pd *)x->x_example_freq_inlet, x->x_f);
  x->x_label_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  pd_float((t_pd *)x->x_label_freq_inlet, x->x_f);

  x->x_prediction_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_example_bang_out = outlet_new(&x->x_obj, &s_bang);
  x->x_label_bang_out = outlet_new(&x->x_obj, &s_bang);

  return (void *)x;
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void layer_forward(t_nnlfo *x, t_layer *layer, t_float *input_buffer) {
  int n = layer->l_n;
  int n_prev = layer->l_n_prev;
  int n_prev_unroll_limit = n_prev - 8;
  int batch_size = x->x_batch_size;
  t_float leak = x->x_leak;

  int is_relu = (layer->l_activation == ACTIVATION_RELU);

  #pragma GCC ivdep
  for (int j = 0; j < batch_size; j++) {
    int batch_offset = j * n_prev;

    for (int i = 0; i < n; i++) {
      t_float z = layer->l_biases[i];
      if (i + 1 < n) __builtin_prefetch(&layer->l_biases[i+1], 0, 3);

      int k = 0;
      for (; k <= n_prev_unroll_limit; k += 8) {
        __builtin_prefetch(&layer->l_weights[i*n_prev+k+8], 0, 3);
        z += layer->l_weights[i*n_prev+k] * input_buffer[batch_offset + k] +
             layer->l_weights[i*n_prev+k+1] * input_buffer[batch_offset + k+1] +
             layer->l_weights[i*n_prev+k+2] * input_buffer[batch_offset + k+2] +
             layer->l_weights[i*n_prev+k+3] * input_buffer[batch_offset + k+3] +
             layer->l_weights[i*n_prev+k+4] * input_buffer[batch_offset + k+4] +
             layer->l_weights[i*n_prev+k+5] * input_buffer[batch_offset + k+5] +
             layer->l_weights[i*n_prev+k+6] * input_buffer[batch_offset + k+6] +
             layer->l_weights[i*n_prev+k+7] * input_buffer[batch_offset + k+7];
      }

      for (; k < n_prev; k++) {
        z += layer->l_weights[i*n_prev+k] * input_buffer[batch_offset + k];
      }

      int cache_idx = j * n + i;  // batch_idx * units_in_layer + unit_idx
      layer->l_z_cache[cache_idx] = z;
      layer->l_a_cache[cache_idx] = HANDLE_ACTIVATION(z);
    }
  }
}

static void model_forward(t_nnlfo *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    t_float *input = (l == 0) ? x->x_input_features : x->x_layers[l-1].l_a_cache;
    layer_forward(x, layer, input);
  }
}

// see audio_neural_network_optimizations.md for more optimizations
#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void populate_features_linear(t_nnlfo *x,
                                        t_float example_freq,
                                        double current_phase,
                                        int batch_idx) {
  double features_conv = x->x_features_conv;
  int num_features = x->x_num_features;
  int features_unroll_limit = num_features - 8;
  t_float *features_buffer = x->x_input_features + (batch_idx * num_features);
  t_float recip = x->x_features_reciprocal; // keep the phase representation within the range
  // 0, 1

  int i = 0;
  #pragma GCC ivdep
  for (; i <= features_unroll_limit; i += 8) {
    features_buffer[i] = current_phase * recip;
    current_phase += example_freq * features_conv;
    features_buffer[i+1] = current_phase * recip;
    current_phase += example_freq * features_conv;
    features_buffer[i+2] = current_phase * recip;
    current_phase += example_freq * features_conv;
    features_buffer[i+3] = current_phase * recip;
    current_phase += example_freq * features_conv;
    features_buffer[i+4] = current_phase * recip;
    current_phase += example_freq * features_conv;
    features_buffer[i+5] = current_phase * recip;
    current_phase += example_freq * features_conv;
    features_buffer[i+6] = current_phase * recip;
    current_phase += example_freq * features_conv;
    features_buffer[i+7] = current_phase * recip;
    current_phase += example_freq * features_conv;
  }
  // deal with extra features (not currently being used as x_num_features is a
  // multiple of 8)
  for (; i < num_features; i++) {
    features_buffer[i] = current_phase * recip;
    current_phase += example_freq * features_conv;
  }
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_dz(t_nnlfo *x, t_layer *layer) {
  int is_relu = layer->l_activation == ACTIVATION_RELU;
  int n = layer->l_n;
  int batch_size = x->x_batch_size;
  t_float leak = x->x_leak;
  int n_unroll_limit = n - 8;

  #define ACTIVATION_DERIVATIVE(z) \
    (is_relu ? ((z > 0) ? 1.0f : leak) : 1.0f)
  // for "batch as outer dimension" layout
  #define BATCH_IDX(batch, stride, idx) ((batch) * (stride) + (idx))

  for (int j = 0; j < batch_size; j++) {
    int batch_offset = j * n;  // Offset to the start of this batch example

    int i = 0;
    #pragma GCC ivdep
    for (; i <= n_unroll_limit; i += 8) {
      // re the use of i+16, CPUs load memory in fixed size blocks called cache
      // lines (typically 64 bytes or 16 float values), so i+8 can be assumed to
      // have been loaded on the first iteration of the loop
      if (i + 16 < n) {
        __builtin_prefetch(&layer->l_da[batch_offset + i + 16], 0, 3);
        __builtin_prefetch(&layer->l_z_cache[batch_offset + i + 16], 0, 3);
      }

      layer->l_dz[BATCH_IDX(j, n, i)] = 
          layer->l_da[BATCH_IDX(j, n, i)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i)]);
      layer->l_dz[BATCH_IDX(j, n, i+1)] = 
          layer->l_da[BATCH_IDX(j, n, i+1)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i+1)]);
      layer->l_dz[BATCH_IDX(j, n, i+2)] = 
          layer->l_da[BATCH_IDX(j, n, i+2)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i+2)]);
      layer->l_dz[BATCH_IDX(j, n, i+3)] = 
          layer->l_da[BATCH_IDX(j, n, i+3)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i+3)]);
      layer->l_dz[BATCH_IDX(j, n, i+4)] = 
          layer->l_da[BATCH_IDX(j, n, i+4)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i+4)]);
      layer->l_dz[BATCH_IDX(j, n, i+5)] = 
          layer->l_da[BATCH_IDX(j, n, i+5)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i+5)]);
      layer->l_dz[BATCH_IDX(j, n, i+6)] = 
          layer->l_da[BATCH_IDX(j, n, i+6)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i+6)]);
      layer->l_dz[BATCH_IDX(j, n, i+7)] = 
          layer->l_da[BATCH_IDX(j, n, i+7)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i+7)]);
    }

    for (; i < n; i++) {
      layer->l_dz[BATCH_IDX(j, n, i)] = 
          layer->l_da[BATCH_IDX(j, n, i)] * 
          ACTIVATION_DERIVATIVE(layer->l_z_cache[BATCH_IDX(j, n, i)]);
    }
  }

  #undef ACTIVATION_DERIVATIVE
  #undef BATCH_IDX
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_db(t_nnlfo *x, int l, t_layer *layer) {
  int n = layer->l_n;
  int batch_size = x->x_batch_size;
  int n_unroll_limit = n - 8;
  t_float batch_scale = x->x_batch_scale;

  memset(layer->l_db, 0, sizeof(t_float) * n);

  for (int b = 0; b < batch_size; b++) {
    int batch_offset = b * n;
    int i = 0;
    #pragma GCC ivdep
    for (; i <= n_unroll_limit; i += 8) {
      if (i + 16 < n) {
        __builtin_prefetch(&layer->l_dz[batch_offset + i + 16], 0, 3);
      }
      layer->l_db[i] += layer->l_dz[batch_offset + i];
      layer->l_db[i+1] += layer->l_dz[batch_offset + i + 1];
      layer->l_db[i+2] += layer->l_dz[batch_offset + i + 2];
      layer->l_db[i+3] += layer->l_dz[batch_offset + i + 3];
      layer->l_db[i+4] += layer->l_dz[batch_offset + i + 4];
      layer->l_db[i+5] += layer->l_dz[batch_offset + i + 5];
      layer->l_db[i+6] += layer->l_dz[batch_offset + i + 6];
      layer->l_db[i+7] += layer->l_dz[batch_offset + i + 7];
    }

    for (; i < n; i++) {
      layer->l_db[i] += layer->l_dz[batch_offset + i];
    }
  }

  int i = 0;
  #pragma GCC ivdep
  for (; i <= n_unroll_limit; i += 8) {
    if (i + 16 < n) {
      __builtin_prefetch(&layer->l_db[i+16], 1, 3);
    }
    layer->l_db[i] *= batch_scale;
    layer->l_db[i+1] *= batch_scale;
    layer->l_db[i+2] *= batch_scale;
    layer->l_db[i+3] *= batch_scale;
    layer->l_db[i+4] *= batch_scale;
    layer->l_db[i+5] *= batch_scale;
    layer->l_db[i+6] *= batch_scale;
    layer->l_db[i+7] *= batch_scale;
  }
  for (; i < n; i++) {
    layer->l_db[i] *= batch_scale;
  }
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_da_prev(t_nnlfo *x, int l, t_layer *layer) {
  int n = layer->l_n;
  int n_prev = layer->l_n_prev;
  int n_prev_unroll_limit = n_prev - 8;
  int batch_size = x->x_batch_size;
  t_layer *prev_layer = &x->x_layers[l-1];

  memset(prev_layer->l_da, 0, sizeof(t_float) * n_prev * batch_size);

  for (int b = 0; b < batch_size; b++) {
    int batch_offset_curr = b * n;
    int batch_offset_prev = b * n_prev;

    for (int j = 0; j < n; j++) {
      t_float dz_j = layer->l_dz[batch_offset_curr + j];

      if (j + 2 < n) {
        __builtin_prefetch(&layer->l_dz[batch_offset_curr + j + 2], 0, 3);
      }

      int weight_base_idx = j * n_prev;
      int k = 0;
      #pragma GCC ivdep
      for (; k <= n_prev_unroll_limit; k += 8) {
        if (k + 16 < n_prev) {
          __builtin_prefetch(&layer->l_weights[weight_base_idx + k + 16], 0, 3);
          __builtin_prefetch(&prev_layer->l_da[batch_offset_prev + k + 16], 1, 3);
        }

        prev_layer->l_da[batch_offset_prev + k] += layer->l_weights[weight_base_idx + k] * dz_j;
        prev_layer->l_da[batch_offset_prev + k+1] += layer->l_weights[weight_base_idx + k+1] * dz_j;
        prev_layer->l_da[batch_offset_prev + k+2] += layer->l_weights[weight_base_idx + k+2] * dz_j;
        prev_layer->l_da[batch_offset_prev + k+3] += layer->l_weights[weight_base_idx + k+3] * dz_j;
        prev_layer->l_da[batch_offset_prev + k+4] += layer->l_weights[weight_base_idx + k+4] * dz_j;
        prev_layer->l_da[batch_offset_prev + k+5] += layer->l_weights[weight_base_idx + k+5] * dz_j;
        prev_layer->l_da[batch_offset_prev + k+6] += layer->l_weights[weight_base_idx + k+6] * dz_j;
        prev_layer->l_da[batch_offset_prev + k+7] += layer->l_weights[weight_base_idx + k+7] * dz_j;
      }

      for (; k < n_prev; k++) {
        prev_layer->l_da[batch_offset_prev + k] += layer->l_weights[weight_base_idx + k] * dz_j;
      }
    }
  }
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_dw(t_nnlfo *x, int l, t_layer *layer) {
  int n = layer->l_n;
  int n_prev = layer->l_n_prev;
  int batch_size = x->x_batch_size;
  t_float *prev_activations = (l == 0) ? x->x_input_features : x->x_layers[l-1].l_a_cache;
  int n_prev_unroll_limit = n_prev - 8;
  t_float batch_scale = x->x_batch_scale;
  int total_weights = n * n_prev;
  int weight_unroll_limit = total_weights - 8;

  memset(layer->l_dw, 0, sizeof(t_float) * n * n_prev);

  for (int b = 0; b < batch_size; b++) {
    int batch_offset_curr = b * n;
    int batch_offset_prev = b * n_prev;

    for (int i = 0; i < n; i++) {
      t_float dz_i = layer->l_dz[batch_offset_curr + i];

      // guessing a bit with the i+2 offset. the issue is related to "prefetch
      // distance": prefetching by 1 might not provide enough load time (the
      // inner loop might finish before the prefetch is complete)
      if (i + 2 < n) {
        __builtin_prefetch(&layer->l_dz[batch_offset_curr + i + 2], 0, 3);
      }

      int j = 0;
      #pragma GCC ivdep
      for (; j <= n_prev_unroll_limit; j += 8) {
        if (j + 16 < n_prev) {
          __builtin_prefetch(&prev_activations[batch_offset_prev + j + 16], 0, 3);
        }

        int idx = i * n_prev + j;
        layer->l_dw[idx]     += dz_i * prev_activations[batch_offset_prev + j];
        layer->l_dw[idx + 1] += dz_i * prev_activations[batch_offset_prev + j + 1];
        layer->l_dw[idx + 2] += dz_i * prev_activations[batch_offset_prev + j + 2];
        layer->l_dw[idx + 3] += dz_i * prev_activations[batch_offset_prev + j + 3];
        layer->l_dw[idx + 4] += dz_i * prev_activations[batch_offset_prev + j + 4];
        layer->l_dw[idx + 5] += dz_i * prev_activations[batch_offset_prev + j + 5];
        layer->l_dw[idx + 6] += dz_i * prev_activations[batch_offset_prev + j + 6];
        layer->l_dw[idx + 7] += dz_i * prev_activations[batch_offset_prev + j + 7];
      }

      for (; j < n_prev; j++) {
        layer->l_dw[i * n_prev + j] += dz_i * prev_activations[batch_offset_prev + j];
      }
    }
  }

  int i = 0;
  #pragma GCC ivdep
  for (; i <= weight_unroll_limit; i += 8) {
    if (i + 16 < total_weights) {
      __builtin_prefetch(&layer->l_dw[i+16], 1, 3); // note: 1 = write hint
    }
    layer->l_dw[i] *= batch_scale;
    layer->l_dw[i+1] *= batch_scale;
    layer->l_dw[i+2] *= batch_scale;
    layer->l_dw[i+3] *= batch_scale;
    layer->l_dw[i+4] *= batch_scale;
    layer->l_dw[i+5] *= batch_scale;
    layer->l_dw[i+6] *= batch_scale;
    layer->l_dw[i+7] *= batch_scale;

  }
  for (; i < n * n_prev; i++) {
    layer->l_dw[i] *= batch_scale;
  }
}

// note that the output layer always has 1 neuron
#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void calculate_da_outer(t_nnlfo *x, t_layer *layer) {
  int batch_size = x->x_batch_size;
  int batch_size_unroll_limit = batch_size - 8;

  int i = 0;
  #pragma GCC ivdep
  for (; i <= batch_size_unroll_limit; i += 8) {
    // note the use of i+16 here and the prefetching of 16 items ahead
    // see the references to cache lines in flattened_array_intuitions.md
    // for details. I've been using prefetching inconsistently in the code so
    // far
    if (i + 16 < batch_size) {
      __builtin_prefetch(&layer->l_a_cache[i+16], 0, 3);
      __builtin_prefetch(&x->x_batch_labels[i+16], 0, 3);
    }

    layer->l_da[i] = layer->l_a_cache[i] - x->x_batch_labels[i];
    layer->l_da[i+1] = layer->l_a_cache[i+1] - x->x_batch_labels[i+1];
    layer->l_da[i+2] = layer->l_a_cache[i+2] - x->x_batch_labels[i+2];
    layer->l_da[i+3] = layer->l_a_cache[i+3] - x->x_batch_labels[i+3];
    layer->l_da[i+4] = layer->l_a_cache[i+4] - x->x_batch_labels[i+4];
    layer->l_da[i+5] = layer->l_a_cache[i+5] - x->x_batch_labels[i+5];
    layer->l_da[i+6] = layer->l_a_cache[i+6] - x->x_batch_labels[i+6];
    layer->l_da[i+7] = layer->l_a_cache[i+7] - x->x_batch_labels[i+7];
  }

  for (; i < batch_size; i++) {
    layer->l_da[i] = layer->l_a_cache[i] - x->x_batch_labels[i];
  }
}

static void layer_backward(t_nnlfo *x, int l, t_layer *layer) {
  if (l == x->x_num_layers - 1) calculate_da_outer(x, layer);

  calculate_dz(x, layer);
  calculate_dw(x, l, layer);
  calculate_db(x, l, layer);
  if (l > 0) calculate_da_prev(x, l, layer);
}

static void model_backward(t_nnlfo *x) {
  for (int l = x->x_num_layers - 1; l >= 0; l--) {
    t_layer *layer = &x->x_layers[l];
    layer_backward(x, l, layer);
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

static inline float faster_inv_sqrt(float number) {
  union {
    float f;
    int32_t i;
  } conv;
  
  conv.f = number;
  conv.i = 0x5f3759df - (conv.i >> 1);
  
  return conv.f; // Skip the Newton-Raphson iteration
}

#pragma GCC optimize("unroll-loops", "tree-vectorize")
static void update_parameters_adam(t_nnlfo *x) {
  const t_float beta1 = x->x_beta_1;
  const t_float beta2 = x->x_beta_2;
  const t_float one_minus_beta1 = (t_float)1.0 - beta1;
  const t_float one_minus_beta2 = (t_float)1.0 - beta2;
  const t_float alpha = x->x_alpha;
  const t_float epsilon = (t_float)1e-8;
  const t_float weight_decay_factor = (t_float)1.0 - alpha * x->x_lambda;

  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    // guessing a bit here with prefetching ahead by 2
    if (l + 2 < x->x_num_layers) {
        __builtin_prefetch(&x->x_layers[l+2], 0, 3);
    }

    int n = layer->l_n;
    int n_prev = layer->l_n_prev;
    int num_weights = n * n_prev;
    int weights_unroll_limit = num_weights - 8;
    int biases_unroll_limit = n - 8;

    int j = 0;
    #pragma GCC ivdep
    for (; j <= weights_unroll_limit; j += 8) {
      if (j + 16 < num_weights) {
        __builtin_prefetch(&layer->l_weights[j+16], 1, 3);
        __builtin_prefetch(&layer->l_dw[j+16], 0, 3);
        __builtin_prefetch(&layer->l_v_dw[j+16], 1, 3);
        __builtin_prefetch(&layer->l_s_dw[j+16], 1, 3);
      }

      for (int k = 0; k < 8; k++) {
        int idx = j + k;
        t_float dw = layer->l_dw[idx];

        layer->l_v_dw[idx] = beta1 * layer->l_v_dw[idx] + one_minus_beta1 * dw;
        layer->l_s_dw[idx] = beta2 * layer->l_s_dw[idx] + one_minus_beta2 * dw * dw;

        t_float update = alpha * layer->l_v_dw[idx] * faster_inv_sqrt(layer->l_s_dw[idx] + epsilon);

        layer->l_weights[idx] = weight_decay_factor * layer->l_weights[idx] - update;
      }
    }

    for (; j < num_weights; j++) {
      t_float dw = layer->l_dw[j];

      layer->l_v_dw[j] = beta1 * layer->l_v_dw[j] + one_minus_beta1 * dw;
      layer->l_s_dw[j] = beta2 * layer->l_s_dw[j] + one_minus_beta2 * dw * dw;

      t_float update = alpha * layer->l_v_dw[j] * faster_inv_sqrt(layer->l_s_dw[j] + epsilon);

      layer->l_weights[j] = weight_decay_factor * layer->l_weights[j] - update;
    }

    int k = 0;
    #pragma GCC ivdep
    for (; k <= biases_unroll_limit; k += 8) {
      if (k + 16 < n) {
        __builtin_prefetch(&layer->l_biases[k+16], 1, 3);
        __builtin_prefetch(&layer->l_db[k+16], 0, 3);
        __builtin_prefetch(&layer->l_v_db[k+16], 1, 3);
        __builtin_prefetch(&layer->l_s_db[k+16], 1, 3);
      }

      for (int i = 0; i < 8; i++) {
        int idx = k + i;
        t_float db = layer->l_db[idx];

        layer->l_v_db[idx] = beta1 * layer->l_v_db[idx] + one_minus_beta1 * db;
        layer->l_s_db[idx] = beta2 * layer->l_s_db[idx] + one_minus_beta2 * db * db;

        t_float update = alpha * layer->l_v_db[idx] * faster_inv_sqrt(layer->l_s_db[idx] + epsilon);

        layer->l_biases[idx] -= update;
      }
    }

    for (; k < n; k++) {
      t_float db = layer->l_db[k];

      layer->l_v_db[k] = beta1 * layer->l_v_db[k] + one_minus_beta1 * db;
      layer->l_s_db[k] = beta2 * layer->l_s_db[k] + one_minus_beta2 * db * db;

      t_float update = alpha * layer->l_v_db[k] * faster_inv_sqrt(layer->l_s_db[k] + epsilon);

      layer->l_biases[k] -= update;
    }
  }
}

static t_int *nnlfo_perform(t_int *w) {
  t_nnlfo *x = (t_nnlfo *)(w[1]);
  t_sample *example_freq_in = (t_sample *)(w[2]);
  t_sample *label_freq_in = (t_sample *)(w[3]);
  t_sample *pred_out = (t_sample *)(w[4]);
  int n = (int)(w[5]);

  double conv = x->x_conv;
  t_sample example_freq = x->x_example_freq;
  t_sample label_freq = x->x_label_freq;

  double example_phase = x->x_example_phase;
  double example_pw = x->x_example_pw;
  double label_phase = x->x_label_phase;
  double label_pw = x->x_label_pw;

  t_float previous_example_pulse = x->x_previous_example_pulse;
  t_float previous_label_pulse = x->x_previous_label_pulse;

  t_sample y_hat = x->x_y_hat;

  int example_bang = 0;
  int label_bang = 0;
  int output_layer = x->x_num_layers - 1;

  int batch_idx = x->x_batch_index;
  // for now batch_size is a power of 2
  int batch_idx_mask = x->x_batch_size - 1;
  int features_filled = x->x_features_filled;

  while (n--) {
    example_freq = *example_freq_in++;
    label_freq = *label_freq_in++;

    t_float current_example_pulse = (example_phase < example_pw) ? (t_float)1.0 : (t_float)0.0;
    t_float current_label_pulse = (label_phase < label_pw) ? (t_float)1.0 : (t_float)0.0;

    // used to output a bang for every example or lable zero crossing
    if (current_example_pulse != previous_example_pulse) example_bang = 1;
    if (current_label_pulse != previous_label_pulse) label_bang = 1;

    // make predictions at example and label zero crossings
    if (current_example_pulse != previous_example_pulse || current_label_pulse != previous_label_pulse) {
      x->x_batch_labels[batch_idx] = current_label_pulse;
      populate_features_linear(x, example_freq, example_phase, batch_idx);
      if (features_filled) {
        // model_forward(x);
        y_hat = x->x_layers[output_layer].l_a_cache[batch_idx];
        // model_backward(x);
        // update_parameters_adam(x);
      } else {
        features_filled = batch_idx == batch_idx_mask;
      }
      batch_idx = (batch_idx + 1) & batch_idx_mask;
    }

    *pred_out++ = y_hat;

    example_phase += example_freq * conv;
    example_phase -= floor(example_phase);
    label_phase += label_freq * conv;
    label_phase -= floor(label_phase);
    previous_example_pulse = current_example_pulse;
    previous_label_pulse = current_label_pulse;
  }

  // it works surprisingly well calling it here:
  model_forward(x);
  model_backward(x);
  update_parameters_adam(x);

  x->x_batch_index = batch_idx;
  x->x_features_filled = features_filled;
  x->x_y_hat = y_hat;
  x->x_example_freq = example_freq;
  x->x_label_freq = label_freq;
  x->x_example_phase = example_phase;
  x->x_label_phase = label_phase;
  x->x_previous_example_pulse = previous_example_pulse;
  x->x_previous_label_pulse = previous_label_pulse;

  if (example_bang) outlet_bang(x->x_example_bang_out);
  if (label_bang) outlet_bang(x->x_label_bang_out);

  return (w+6);
}

static void nnlfo_dsp(t_nnlfo *x, t_signal **sp) {
  // freq * conv:
  // at 1Hz, increment by 1 sample for each step, at 2Hz increment by 2 samples,
  // etc
  x->x_conv = (double)1.0 / (double)sp[0]->s_sr;
  // number of samples per feature at 1Hz
  x->x_feature_samps = (int)(sp[0]->s_sr / x->x_num_features);
  x->x_features_conv = x->x_conv * x->x_feature_samps;
  dsp_add(nnlfo_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_length);
}

static void reset_model(t_nnlfo *x) {
  int batch_size = x->x_batch_size;

  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->x_layers[l];
    int n = layer->l_n;
    int n_prev = layer->l_n_prev;
    int w_size = n * n_prev;

    memset(layer->l_z_cache, 0, sizeof(t_float) * n * batch_size);
    memset(layer->l_dz, 0, sizeof(t_float) * n * batch_size);
    memset(layer->l_a_cache, 0, sizeof(t_float) * n * batch_size);
    memset(layer->l_da, 0, sizeof(t_float) * n * batch_size);
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
  }
  memset(x->x_batch_labels, 0, sizeof(t_float) * batch_size);
  memset(x->x_input_features, 0, sizeof(t_float) * batch_size * x->x_num_features);
  x->x_example_phase = (double)0.0;
  x->x_label_phase = (double)0.0;
  x->x_batch_index = 0;
  x->x_features_filled = 0;
}

static void set_phase_representation(t_nnlfo *x, t_symbol *_s, int _argc, t_atom *argv) {
  t_symbol *phase_type = atom_getsymbol(argv++);
  if (phase_type == gensym("linear")) {
    x->x_phase_rep = PHASE_LINEAR;
  } else if (phase_type == gensym("binary")) {
    x->x_phase_rep = PHASE_BINARY;
  } else if (phase_type == gensym("linear_inverse")) {
    x->x_phase_rep = PHASE_LINEAR_INVERSE;
  }
}

static void set_alpha(t_nnlfo *x, t_floatarg f) {
  x->x_alpha = f;
}

static void set_beta_1(t_nnlfo *x, t_floatarg f) {
  x->x_beta_1 = f;
}

static void set_beta_2(t_nnlfo *x, t_floatarg f) {
  x->x_beta_2 = f;
}

static void set_leak(t_nnlfo *x, t_floatarg f) {
  x->x_leak = f;
}

static void set_lambda(t_nnlfo *x, t_floatarg f) {
  x->x_lambda = (f < (t_float)0.0) ? (t_float)0.0 : f;
}

static void set_batch_size(t_nnlfo *x, t_floatarg f) {
  f = f > 0 ? f : 8;
  x->x_batch_size = f;
}

void nnlfo_tilde_setup(void) {
  nnlfo_class = class_new(gensym("nnlfo~"),
                          (t_newmethod)nnlfo_new,
                          (t_method)nnlfo_free,
                          sizeof(t_nnlfo),
                          CLASS_DEFAULT,
                          0);
  class_addmethod(nnlfo_class, (t_method)nnlfo_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(nnlfo_class, (t_method)reset_model, gensym("reset"), 0);
  class_addmethod(nnlfo_class, (t_method)set_alpha, gensym("alpha"), A_FLOAT, 0);
  class_addmethod(nnlfo_class, (t_method)set_leak, gensym("leak"), A_FLOAT, 0);
  class_addmethod(nnlfo_class, (t_method)set_lambda, gensym("lambda"), A_FLOAT, 0);
  class_addmethod(nnlfo_class, (t_method)set_beta_1, gensym("beta_1"), A_FLOAT, 0);
  class_addmethod(nnlfo_class, (t_method)set_beta_2, gensym("beta_2"), A_FLOAT, 0);
  class_addmethod(nnlfo_class, (t_method)set_batch_size, gensym("batch_size"), A_FLOAT, 0);

  class_addmethod(nnlfo_class, (t_method)set_phase_representation, gensym("phase_representation"),
                  A_SYMBOL, 0);
  CLASS_MAINSIGNALIN(nnlfo_class, t_nnlfo, x_f);
}

static void nnlfo_free(t_nnlfo *x) {
  if (x->x_example_freq_inlet) inlet_free(x->x_example_freq_inlet);
  if (x->x_label_freq_inlet) inlet_free(x->x_label_freq_inlet);
  if (x->x_prediction_outlet) outlet_free(x->x_prediction_outlet);
  if (x->x_example_bang_out) outlet_free(x->x_example_bang_out);
  if (x->x_label_bang_out) outlet_free(x->x_label_bang_out);

  if (x->x_layer_dims) {
    freebytes(x->x_layer_dims, sizeof(int) * (x->x_num_layers + 1));
  }

  int batch_size = x->x_batch_size;

  if (x->x_input_features) {
    freebytes(x->x_input_features, sizeof(t_float) * x->x_num_features * batch_size);
  }

  if (x->x_batch_labels) {
    freebytes(x->x_batch_labels, sizeof(t_float) * batch_size);
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
      if (layer->l_z_cache) freebytes(layer->l_z_cache, sizeof(t_float) * layer->l_n * batch_size);
      if (layer->l_dz) freebytes(layer->l_dz, sizeof(t_float) * layer->l_n * batch_size);
      if (layer->l_a_cache) freebytes(layer->l_a_cache, sizeof(t_float) * layer->l_n * batch_size);
      if (layer->l_da) freebytes(layer->l_da, sizeof(t_float) * layer->l_n * batch_size);
    }
    freebytes(x->x_layers, sizeof(t_layer) * x->x_num_layers);
  }
}

static void set_layer_dims(t_nnlfo *x) {
  // hardcoded for now
  x->x_layer_dims[0] = x->x_num_features;
  x->x_layer_dims[1] = x->x_num_features * 2;
  x->x_layer_dims[2] = x->x_num_features;
  x->x_layer_dims[3] = x->x_num_features;
  x->x_layer_dims[4] = x->x_num_features;
  x->x_layer_dims[5] = x->x_num_features * 0.5;
  x->x_layer_dims[6] = x->x_num_features * 0.25;
  x->x_layer_dims[7] = 1;
}

static float he_init(int n_prev) {
  t_float u1 = (t_float)rand() / RAND_MAX;
  t_float u2 = (t_float)rand() / RAND_MAX;
  t_float radius = (t_float)(sqrt(-2 * log(u1)));
  t_float theta = 2 * (t_float)M_PI * u2;
  t_float standard_normal = radius * (t_float)(cos(theta));

  return standard_normal * (t_float)(sqrt(2.0 / n_prev));
}

static void init_layer_weights(t_nnlfo *x, int l) {
  t_layer *layer = &x->x_layers[l];
  int size = layer->l_n * layer->l_n_prev;
  for (int i = 0; i < size; i++) {
    t_float weight = he_init(layer->l_n_prev);
    layer->l_weights[i] = weight;
  }
}

static void init_layer_biases(t_nnlfo *x, int l) {
  t_layer *layer = &x->x_layers[l];
  for (int i = 0; i < layer->l_n; i++) {
    // not technically needed
    layer->l_biases[i] = (t_float)0.0;
  }
}

static void initialize_layers(t_nnlfo *x) {
  x->x_layers = (t_layer *)getbytes(sizeof(t_layer) * x->x_num_layers);
  if (!x->x_layers) {
    pd_error(x, "nnlfo~: failed to allocate memory for layers");
    return;
  }

  int batch_size = x->x_batch_size;

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
      pd_error(x, "nnlfo~: failed to allocate memory for layer weights");
      return;
    }
    layer->l_dw = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_dw) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer dw");
      return;
    }
    layer->l_v_dw = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_v_dw) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer v_dw");
      return;
    }
    layer->l_s_dw = (t_float *)getbytes(sizeof(t_float) * layer->l_n * layer->l_n_prev);
    if (!layer->l_s_dw) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer s_dw");
      return;
    }
    layer->l_biases = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_biases) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer biases");
      return;
    }
    layer->l_db = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_db) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer db");
      return;
    }
    layer->l_v_db = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_v_db) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer v_db");
      return;
    }
    layer->l_s_db = (t_float *)getbytes(sizeof(t_float) * layer->l_n);
    if (!layer->l_s_db) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer s_db");
      return;
    }
    layer->l_z_cache = (t_float *)getbytes(sizeof(t_float) * layer->l_n * batch_size);
    if (!layer->l_z_cache) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer z_cache");
      return;
    }
    layer->l_dz = (t_float *)getbytes(sizeof(t_float) * layer->l_n * batch_size);
    if (!layer->l_dz) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer dz");
      return;
    }
    layer->l_a_cache = (t_float *)getbytes(sizeof(t_float) * layer->l_n * batch_size);
    if (!layer->l_a_cache) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer a_cache");
      return;
    }
    layer->l_da = (t_float *)getbytes(sizeof(t_float) * layer->l_n * batch_size);
    if (!layer->l_da) {
      pd_error(x, "nnlfo~: failed to allocate memory for layer da");
      return;
    }

    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }
}
