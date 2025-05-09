#include "m_pd.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  ACTIVATION_LINEAR,
  ACTIVATION_RELU,
  ACTIVATION_SIGMOID,
  ACTIVATION_TANH
} t_activation_type;

typedef struct _layer {
  int n;
  int n_prev;
  t_activation_type activation;

  t_float *weights;
  t_float *dw;
  t_float *biases;
  t_float *db;
  t_float *z_cache;
  t_float *dz;
  t_float *a_cache;
  t_float *da;
} t_layer;

typedef struct _nn4 {
  t_object x_obj;

  t_layer *layers;
  int *x_layer_dims; // input hidden and output dimensions
  int x_num_layers; // excluding input layer

  int x_num_features; // num input features

  int x_batch_size;
  t_float *x_input_features;

  t_float x_leak;
  t_float x_alpha;

  double x_conv;
  double x_example_phase;
  double x_label_phase;
  t_float x_f; // init frequency

  t_float x_current_label;

  t_inlet *x_example_freq_inlet;
  t_inlet *x_label_freq_inlet;
  t_outlet *x_prediction_outlet;
} t_nn4;

#define TABLE_SIZE 4096 // 2^12
static t_float *cos_table = NULL; // shared wavetable
static int cos_table_reference_count = 0;

static void cos_table_init(void) {
  if (cos_table == NULL) {
    cos_table = (t_float *)getbytes(sizeof(t_float) * (TABLE_SIZE));
    if (cos_table) {
      for (int i = 0; i < TABLE_SIZE; i++) {
        cos_table[i] = cosf((i * 2.0f * (t_float)M_PI) / (t_float)TABLE_SIZE);
      }
      post("nn4~: initialized cosine table of size %d", TABLE_SIZE);
    } else {
      post("nn4~ error: failed to allocate memory for cosine table");
      return;
    }
  }
  cos_table_reference_count++;
}

static void cos_table_free(void) {
  cos_table_reference_count--;
  if (cos_table_reference_count <= 0 && cos_table != NULL) {
    freebytes(cos_table, sizeof(t_float) * TABLE_SIZE);
    cos_table = NULL;
    post("nn4~: freed cosine table");
    cos_table_reference_count = 0; // just to be safe
  }
}

float he_init(int n_prev) {
  t_float u1 = (t_float)rand() / RAND_MAX;
  t_float u2 = (t_float)rand() / RAND_MAX;
  t_float radius = (t_float)(sqrt(-2 * log(u1)));
  t_float theta = 2 * (t_float)M_PI * u2;
  t_float standard_normal = radius * (t_float)(cos(theta));

  return standard_normal * (t_float)(sqrt(2.0 / n_prev));
}

static t_class *nn4_class = NULL;

static void init_layer_weights(t_nn4 *x, int l) {
  t_layer *layer = &x->layers[l];
  int size = layer->n * layer->n_prev;
  for (int i = 0; i < size; i++) {
    t_float weight = he_init(layer->n_prev);
    layer->weights[i] = weight;
  }
}

static void init_layer_biases(t_nn4 *x, int l) {
  t_layer *layer = &x->layers[l];
  for (int i = 0; i < layer->n; i++) {
    // not technically needed
    layer->biases[i] = (t_float)0.0;
  }
}

static t_float apply_activation(t_nn4 *x, t_layer *layer, t_float z) {
  switch(layer->activation) {
    case ACTIVATION_SIGMOID:
      return 1.0 / (1.0 + exp(-z));
    case ACTIVATION_TANH:
      return tanh(z);
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

static void nn4_free(t_nn4 *x) {
  cos_table_free();

  if (x->x_example_freq_inlet) inlet_free(x->x_example_freq_inlet);
  if (x->x_label_freq_inlet) inlet_free(x->x_label_freq_inlet);
  if (x->x_prediction_outlet) outlet_free(x->x_prediction_outlet);

  if (x->x_layer_dims) {
    freebytes(x->x_layer_dims, sizeof(int) * (x->x_num_layers + 1));
    x->x_layer_dims = NULL; // probably not needed
  }

  if (x->layers) {
    for (int l = 0; l < x->x_num_layers; l++) {
      t_layer *layer = &x->layers[l];
      if (layer->weights) freebytes(layer->weights, sizeof(t_float) * layer->n * layer->n_prev);
      if (layer->dw) freebytes(layer->dw, sizeof(t_float) * layer->n * layer->n_prev);
      if (layer->biases) freebytes(layer->biases, sizeof(t_float) * layer->n);
      if (layer->db) freebytes(layer->db, sizeof(t_float) * layer->n);
      if (layer->z_cache) freebytes(layer->z_cache, sizeof(t_float) * layer->n * x->x_batch_size);
      if (layer->dz) freebytes(layer->dz, sizeof(t_float) * layer->n * x->x_batch_size);
      if (layer->a_cache) freebytes(layer->a_cache, sizeof(t_float) * layer->n * x->x_batch_size);
      if (layer->da) freebytes(layer->da, sizeof(t_float) * layer->n * x->x_batch_size);
    }
    freebytes(x->layers, sizeof(t_layer) * x->x_num_layers);
  }

  if (x->x_input_features) {
    freebytes(x->x_input_features, sizeof(t_float) * x->x_num_features * x->x_batch_size);
  }
}

static void init_layers(t_nn4 *x) {
  x->layers = (t_layer *)getbytes(sizeof(t_layer) * x->x_num_layers);
  if (!x->layers) {
    pd_error(x, "nn4~: failed to allocate memory for layers");
    return;
  }

  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->layers[l];
    layer->n = x->x_layer_dims[l+1];
    layer->n_prev = x->x_layer_dims[l];
    int cache_size = layer->n * x->x_batch_size;

    if (l < x->x_num_layers - 1) {
      layer->activation = ACTIVATION_TANH;
    } else {
      layer->activation = ACTIVATION_LINEAR; // output layer
    }

    layer->weights = (t_float *)getbytes(sizeof(t_float) * layer->n * layer->n_prev);
    if (!layer->weights) {
      pd_error(x, "nn4~: failed to allocate memory for layer weights");
      return;
    }
    layer->dw = (t_float *)getbytes(sizeof(t_float) * layer->n * layer->n_prev);
    if (!layer->dw) {
      pd_error(x, "nn4~: failed to allocate memory for layer dw");
      return;
    }
    layer->biases = (t_float *)getbytes(sizeof(t_float) * layer->n);
    if (!layer->biases) {
      pd_error(x, "nn4~: failed to allocate memory for layer biases");
      return;
    }
    layer->db = (t_float *)getbytes(sizeof(t_float) * layer->n);
    if (!layer->db) {
      pd_error(x, "nn4~: failed to allocate memory for layer db");
      return;
    }
    layer->z_cache = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->z_cache) {
      pd_error(x, "nn4~: failed to allocate memory for layer z_cache");
      return;
    }
    layer->dz = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->dz) {
      pd_error(x, "nn4~: failed to allocate memory for layer dz");
      return;
    }
    layer->a_cache = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->a_cache) {
      pd_error(x, "nn4~: failed to allocate memory for layer a_cache");
      return;
    }
    layer->da = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->da) {
      pd_error(x, "nn4~: failed to allocate memory for layer da");
      return;
    }

    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }
}

static void *nn4_new(t_symbol *s, int argc, t_atom *argv) {
  t_nn4 *x = (t_nn4 *)pd_new(nn4_class);

  x->x_input_features = NULL;

  x->x_num_layers = 5;
  x->x_num_features = 64; // for now, will eventually be set to Pd bin size
  x->x_batch_size = 1;
  x->x_leak = (t_float)0.001;
  x->x_alpha = (t_float)0.0001;
  x->x_f = (t_float)220.0;
  x->x_conv = (double)0.0;
  x->x_example_phase = (double)0.0;
  x->x_label_phase = (double)0.0;

  x->x_current_label = (t_float)0.0;

  if (argc < 2) {
    pd_error(x, "nn4~: args for 2 hidden layer dimensions required");
    return NULL;
  }

  x->x_layer_dims = (int *)getbytes(sizeof(int) * x->x_num_layers + 1);
  if (!x->x_layer_dims) {
    pd_error(x, "nn4~: failed to allocate memory for layer dims");
    return NULL;
  }

  // hardcoded for now
  x->x_layer_dims[0] = x->x_num_features;
  x->x_layer_dims[1] = 16;
  x->x_layer_dims[2] = 16;
  x->x_layer_dims[3] = 8;
  x->x_layer_dims[4] = 8;
  x->x_layer_dims[5] = 1;
  
  static int seed_initialized = 0;
  if (!seed_initialized) {
    srand((unsigned int)time(NULL));
    seed_initialized = 1;
  }

  cos_table_init();
  init_layers(x);

  x->x_input_features = getbytes(sizeof(t_float) * x->x_num_features * x->x_batch_size);
  if (!x->x_input_features) {
    pd_error(x, "nn4~: failed to allocate memory for input features buffer");
    return NULL;
  }

  x->x_example_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  pd_float((t_pd *)x->x_example_freq_inlet, x->x_f);
  x->x_label_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  x->x_prediction_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void layer_forward(t_nn4 *x, int l, t_layer *layer, t_float *input_buffer) {
  int batch_size = x->x_batch_size; // batch_size = 1
  int n = layer->n;
  int n_prev = layer->n_prev;
  for (int j = 0; j < batch_size; j++) { // batch_size = 1, so j = 0
    for (int i = 0; i < n; i++) {
      t_float z = layer->biases[i];
      for (int k = 0; k < n_prev; k++) {
        z+= layer->weights[i*n_prev+k] * input_buffer[k*batch_size+j];
      }
      int idx = i*batch_size+j;
      layer->z_cache[idx] = z;
      t_float activation = apply_activation(x, layer, z);
      layer->a_cache[idx] = activation;
    }
  }
}

static void model_forward(t_nn4 *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->layers[l];
    t_float *input_buffer = (l == 0) ? x->x_input_features : x->layers[l-1].a_cache;
    layer_forward(x, l, layer, input_buffer);
  }
}

static void calculate_output_layer_da(t_nn4 *x, t_layer *layer) {
  for (int i = 0; i < layer->n; i++) {
    for (int j = 0; j < x->x_batch_size; j++) {
      int idx = i * x->x_batch_size + j; // idx = 0
      layer->da[idx] = layer->a_cache[idx] - x->x_current_label;
    }
  }
}

static void calculate_dz(t_nn4 *x, t_layer *layer) {
  for (int i = 0; i < layer->n; i++) {
    for (int j = 0; j < x->x_batch_size; j++) {
      int idx = i * x->x_batch_size + j;
      layer->dz[idx] = layer->da[idx] *
       activation_derivative(layer->activation,
                             layer->z_cache[idx],
                             layer->a_cache[idx],
                             x->x_leak);
    }
  }
}

static void calculate_dw(t_nn4 *x, int l, t_layer *layer) {
  int n = layer->n;
  int n_prev = layer->n_prev;
  int batch_size = x->x_batch_size;
  t_float *prev_activations = (l == 0) ? x->x_input_features : x->layers[l-1].a_cache;

  memset(layer->dw, 0, sizeof(t_float) * n * n_prev);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_prev; j++) {
      for (int k = 0; k < batch_size; k++) {
        layer->dw[i*n_prev+j] += layer->dz[i*batch_size+k] *
                                prev_activations[j*batch_size+k];
      }
      // average for batch (note this isn't needed with batch_size 1)
      layer->dw[i*n_prev+j] /= batch_size;
    }
  }
}

static void calculate_db(t_nn4 *x, t_layer *layer) {
  memset(layer->db, 0, sizeof(t_float) * layer->n);

  for (int i = 0; i < layer->n; i++) {
    for (int j = 0; j < x->x_batch_size; j++) {
      layer->db[i] += layer->dz[i*x->x_batch_size+j];
    }
    layer->db[i] /= x->x_batch_size;
  }
}

static void calculate_da_prev(t_nn4 *x, int l, t_layer *layer) {
  int n = layer->n;
  int n_prev = layer->n_prev;
  int batch_size = x->x_batch_size;
  t_layer *prev_layer = &x->layers[l-1];

  memset(prev_layer->da, 0, sizeof(t_float) * n_prev * batch_size);

  for (int k = 0; k < n_prev; k++) {
    for (int j = 0; j < batch_size; j++) {
      for (int i = 0; i < n; i++) {
        prev_layer->da[k*batch_size+j] += layer->weights[i*n_prev+k] *
                                        layer->dz[i*batch_size+j];
      }
    }
  }
}

static void layer_backward(t_nn4 *x, int l, t_layer *layer) {
  if (l == x->x_num_layers - 1) {
    calculate_output_layer_da(x, layer);
  }

  calculate_dz(x, layer);
  calculate_dw(x, l, layer);

  if (l > 0) {
    calculate_da_prev(x, l, layer);
  }
}

static void model_backward(t_nn4 *x) {
  for (int l = x->x_num_layers - 1; l >= 0; l--) {
    t_layer *layer = &x->layers[l];
    layer_backward(x, l, layer);
  }
}

static void update_parameters(t_nn4 *x) {
  for (int l = 0; l < x->x_num_layers; l++) {
    t_layer *layer = &x->layers[l];
    int n = layer->n;
    int n_prev = layer->n_prev;

    for (int i = 0; i < n * n_prev; i++) {
      layer->weights[i] -= x->x_alpha * layer->dw[i];
    }

    for (int i = 0; i < n; i++) {
      layer->biases[i] -= x->x_alpha * layer->db[i];
    }
  }
}

static t_int *nn4_perform(t_int *w) {
  t_nn4 *x = (t_nn4 *)(w[1]);
  t_sample *example_freq_in = (t_sample *)(w[2]);
  t_sample *label_freq_in = (t_sample *)(w[3]);
  t_sample *out = (t_sample *)(w[4]);
  int n = (int)(w[5]);

  double conv = x->x_conv;

  t_float *table = cos_table;
  t_float *features = x->x_input_features;
  int table_mask = TABLE_SIZE - 1;
  int features_mask = x->x_num_features - 1;
  double example_phase = x->x_example_phase;
  double label_phase = x->x_label_phase;
  double features_phase = (double)0.0;

  int features_step = TABLE_SIZE / x->x_num_features;

  while (n--) {
    t_sample example_freq = *example_freq_in++;
    t_sample label_freq = *label_freq_in++;
    t_sample y_hat = (t_sample)0.0;

    // populate the features buffer
    // assuming batch_size = 1 for now
    features_phase = example_phase;
    for (int i = 0; i < x->x_num_features; i++) {
      unsigned int f_idx = (unsigned int)features_phase;
      double f_frac = features_phase - (double)f_idx;
      f_idx &= table_mask;
      t_sample f1 = table[f_idx];
      t_sample f2 = table[(f_idx + 1) & table_mask];
      t_sample f = f1 + f_frac * (f2 - f1);
      features[i] = f;
      features_phase += example_freq * conv * features_step;
    }

    unsigned int l_idx = (unsigned int)label_phase;
    double l_frac = label_phase - (double)l_idx;
    l_idx &= table_mask;
    t_sample l1 = table[l_idx];
    t_sample l2 = table[(l_idx + 1) & table_mask];
    x->x_current_label = l1 + l_frac * (l2 - l1);

    model_forward(x);
    model_backward(x);
    update_parameters(x);

    y_hat = x->layers[x->x_num_layers - 1].a_cache[0];

    *out++ = y_hat;

    example_phase += example_freq * conv;
    label_phase += label_freq * conv;
  }

  while (example_phase >= TABLE_SIZE) example_phase -= TABLE_SIZE;
  while (example_phase < 0) example_phase += TABLE_SIZE;
  x->x_example_phase = example_phase;

  while (label_phase >= TABLE_SIZE) label_phase -= TABLE_SIZE;
  while (label_phase < 0) label_phase += TABLE_SIZE;
  x->x_label_phase = label_phase;

  return (w+6);
}

static void nn4_dsp(t_nn4 *x, t_signal **sp) {
  x->x_conv = (double)((double)TABLE_SIZE / (double)sp[0]->s_sr);
  dsp_add(nn4_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_length);
}

void nn4_tilde_setup(void) {
  nn4_class = class_new(gensym("nn4~"),
                        (t_newmethod)nn4_new,
                        (t_method)nn4_free,
                        sizeof(t_nn4),
                        CLASS_DEFAULT,
                        A_GIMME, 0);
  class_addmethod(nn4_class, (t_method)nn4_dsp, gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(nn4_class, t_nn4, x_f);
}
