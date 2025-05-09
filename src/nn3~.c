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

typedef struct _nn3 {
  t_object x_obj;
  
  int *layer_dims; // hidden layer dims
  int num_layers; // hidden + output layer
  int num_features; // block size
  int L_plus_1;
  int batch_size;
  t_float alpha;
  t_float leak;

  t_layer *layers;
  t_float *x_features;
  int features_buffer_filled;
  t_float *predictions;
  t_float current_label;

  double conv; // maybe t_float?
  double x_phase;
  double y_phase;
  int bin_phase;
  int pred_phase;

  t_float x_f;
  t_float y_f;

  t_inlet *x_freq_inlet;
  t_inlet *y_freq_inlet;
  t_outlet *seq_outlet;
} t_nn3;

#define TABLE_SIZE 4096 // 2^12
// #define TABLE_SIZE 512 // 2^9
static t_float *cos_table = NULL; // shared wavetable
static int cos_table_reference_count = 0;
static t_float *lin_table = NULL;
static int lin_table_reference_count = 0;

static void cos_table_init(void) {
  if (cos_table == NULL) {
    cos_table = (t_float *)getbytes(sizeof(t_float) * (TABLE_SIZE));
    if (cos_table) {
      for (int i = 0; i < TABLE_SIZE; i++) {
        cos_table[i] = cosf((i * 2.0f * (t_float)M_PI) / (t_float)TABLE_SIZE);
      }
      post("nn3~: initialized cosine table of size %d", TABLE_SIZE);
    } else {
      post("nn3~ error: failed to allocate memory for cosine table");
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
    post("nn3~: freed cosine table");
    cos_table_reference_count = 0; // just to be safe
  }
}

static void lin_table_init(void) {
  if (lin_table == NULL) {
    lin_table = (t_float *)getbytes(sizeof(t_float) * TABLE_SIZE);
    if (lin_table) {
      for (int i = 0; i < TABLE_SIZE; i++) {
        lin_table[i] = (t_float)(((t_float)i / (t_float)TABLE_SIZE) * (t_float)2.0 - (t_float)1.0);
      }
      post("nn3~: initialized linear table of size %d", TABLE_SIZE);
    } else {
      post("nn3~: failed to allocate memory for linear table");
      return;
    }
  }
  lin_table_reference_count++;
}

static void lin_table_free(void) {
  lin_table_reference_count--;
  if (lin_table_reference_count <= 0 && lin_table != NULL) {
    freebytes(lin_table, sizeof(t_float) * TABLE_SIZE);
    lin_table = NULL;
    post("nn3~: freed linear table");
    lin_table_reference_count = 0;
  }
}

float he_init(int n_prev)
{
  t_float u1 = (t_float)rand() / RAND_MAX;
  t_float u2 = (t_float)rand() / RAND_MAX;
  t_float radius = (t_float)(sqrt(-2 * log(u1)));
  t_float theta = 2 * (t_float)M_PI * u2;
  t_float standard_normal = radius * (t_float)(cos(theta));

  return standard_normal * (t_float)(sqrt(2.0 / n_prev));
}

/* starts here */

static t_class *nn3_class = NULL;

static void nn3_free(t_nn3 *x) {
  cos_table_free();
  lin_table_free();

  if (x->seq_outlet) outlet_free(x->seq_outlet);
  if (x->x_freq_inlet) inlet_free(x->x_freq_inlet);
  if (x->y_freq_inlet) inlet_free(x->y_freq_inlet);

  if (x->x_features) {
    freebytes(x->x_features, sizeof(t_float) * x->num_features);
    x->x_features = NULL;
  }

  if (x->predictions) {
    freebytes(x->predictions, sizeof(t_float) * TABLE_SIZE);
  }

  if (x->layer_dims) {
    freebytes(x->layer_dims, sizeof(int) * (x->L_plus_1));
    x->layer_dims = NULL;
  }

  if (x->layers) {
    for (int l = 0; l < x->num_layers; l++) {
      t_layer *layer = &x->layers[l];
      if (layer->weights) freebytes(layer->weights, sizeof(t_float) * layer->n * layer->n_prev);
      if (layer->dw) freebytes(layer->dw, sizeof(t_float) * layer->n * layer->n_prev);
      if (layer->biases) freebytes(layer->biases, sizeof(t_float) * layer->n);
      if (layer->db) freebytes(layer->db, sizeof(t_float) * layer->n);
      if (layer->z_cache) freebytes(layer->z_cache, sizeof(t_float) * layer->n * x->batch_size);
      if (layer->dz) freebytes(layer->dz, sizeof(t_float) * layer->n * x->batch_size);
      if (layer->a_cache) freebytes(layer->a_cache, sizeof(t_float) * layer->n * x->batch_size);
      if (layer->da) freebytes(layer->da, sizeof(t_float) * layer->n * x->batch_size);
    }
    freebytes(x->layers, sizeof(t_layer) * x->num_layers);
  }
}

static void init_layer_weights(t_nn3 *x, int l) {
  post("initializing weights for layer %d", l);
  t_layer *layer = &x->layers[l];
  int size = layer->n * layer->n_prev;
  post("size for weight init: %d", size);
  for (int i = 0; i < size; i++) {
    t_float weight = he_init(layer->n_prev);
    layer->weights[i] = weight;
  }
}

static void init_layer_biases(t_nn3 *x, int l) {
  t_layer *layer = &x->layers[l];
  for (int i = 0; i < layer->n; i++) {
    // not technically needed
    layer->biases[i] = (t_float)0.0;
  }
}

static void init_layers(t_nn3 *x) {
  x->layers = (t_layer *)getbytes(sizeof(t_layer) * x->num_layers);
  if (!x->layers) {
    pd_error(x, "nn3~: failed to allocate memory for layers");
    return;
  }

  for (int l = 0; l < x->num_layers; l++) {
    t_layer *layer = &x->layers[l];
    layer->n = x->layer_dims[l+1]; // confusing but correct
    layer->n_prev = x->layer_dims[l];
    int cache_size = layer->n * x->batch_size;

    if (l < x->num_layers - 1) {
      layer->activation = ACTIVATION_TANH;
    } else {
      layer->activation = ACTIVATION_LINEAR; // output layer
    }

    layer->weights = (t_float *)getbytes(sizeof(t_float) * layer->n * layer->n_prev);
    if (!layer->weights) {
      pd_error(x, "nn3~: failed to allocate memory for layer weights");
      return;
    }
    layer->dw = (t_float *)getbytes(sizeof(t_float) * layer->n * layer->n_prev);
    if (!layer->dw) {
      pd_error(x, "nn3~: failed to allocate memory for layer dw");
      return;
    }
    layer->biases = (t_float *)getbytes(sizeof(t_float) * layer->n);
    if (!layer->biases) {
      pd_error(x, "nn3~: failed to allocate memory for layer biases");
      return;
    }
    layer->db = (t_float *)getbytes(sizeof(t_float) * layer->n);
    if (!layer->db) {
      pd_error(x, "nn3~: failed to allocate memory for layer db");
      return;
    }
    layer->z_cache = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->z_cache) {
      pd_error(x, "nn3~: failed to allocate memory for layer z_cache");
      return;
    }
    layer->dz = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->dz) {
      pd_error(x, "nn3~: failed to allocate memory for layer dz");
      return;
    }
    layer->a_cache = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->a_cache) {
      pd_error(x, "nn3~: failed to allocate memory for layer a_cache");
      return;
    }
    layer->da = (t_float *)getbytes(sizeof(t_float) * cache_size);
    if (!layer->da) {
      pd_error(x, "nn3~: failed to allocate memory for layer da");
      return;
    }
    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }
}

static void *nn3_new(t_symbol *s, int argc, t_atom *argv) {
  t_nn3 *x = (t_nn3 *)pd_new(nn3_class);

  x->layer_dims = NULL;
  x->alpha = (t_float)0.00001;
  x->leak = (t_float)0.01;
  x->layers = NULL;
  x->x_features = NULL;
  x->predictions = NULL;
  // input layer isn't a layer; 2 hidden layers + output layer
  x->num_layers= 3;
  x->L_plus_1 = x->num_layers + 1; // to account for input layer/dimension
  x->conv = (double)0.0;
  x->x_phase = (double)0.0;
  x->y_phase = (double)0.0;
  x->bin_phase = 0;
  x->pred_phase = 0;
  x->x_f = (t_float)220.0;
  x->y_f = (t_float)440.0;
  x->current_label = (t_float)0.0;
  x->features_buffer_filled = 0;

  if (argc < 2) {
    pd_error(x, "nn3~: args for 2 hidden layer dimensions required");
    return NULL;
  }

  x->layer_dims = (int *)getbytes(sizeof(int) * (x->L_plus_1));
  if (!x->layer_dims) {
    pd_error(x, "nn3~: failed to allocate memory for layer_dims");
    return NULL;

  }

  x->batch_size = 1;
  x->num_features = 64; // equals the Pd bin size, but adding here for now

  for (int i = 0; i < x->L_plus_1; i++) {
    if (i == 0) {
      x->layer_dims[i] = 64; // will get updated to batch size
    } else if (i == (x->num_layers)) {
      x->layer_dims[i] = 1; // this is awkward
    } else {
      x->layer_dims[i] = atom_getint(argv++);
    }
  }

  static int seed_initialized = 0;
  if (!seed_initialized) {
    srand((unsigned int)time(NULL));
    seed_initialized = 1;
  }

  init_layers(x);
  for (int l = 0; l < x->num_layers; l++) {
    t_layer *layer = &x->layers[l];
  }
  cos_table_init();
  lin_table_init();
  x->predictions = getbytes(sizeof(t_float) * TABLE_SIZE);
  if (!x->predictions) {
    pd_error(x, "failed to allocate memory for predictions buffer");
    return NULL;
  }

  x->x_features = getbytes(sizeof(t_float) * x->num_features);
  if (!x->x_features) {
    pd_error(x, "nn3~: unable to allocate memory to features buffer");
    return NULL;
  }

  x->x_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  pd_float((t_pd *)x->x_freq_inlet, x->x_f);
  x->y_freq_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  pd_float((t_pd *)x->y_freq_inlet, x->y_f);
  x->seq_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static t_float apply_activation(t_nn3 *x, t_layer *layer, t_float z) {
  switch(layer->activation) {
    case ACTIVATION_SIGMOID:
      return 1.0 / (1.0 + exp(-z));
    case ACTIVATION_TANH:
      return tanh(z);
    case ACTIVATION_RELU:
      return z > 0 ? z : z * x->leak;
    case ACTIVATION_LINEAR:
    default:
      return z;
  }
}

static void layer_forward(t_nn3 *x, int l, t_layer *layer, t_float *input) {
  int batch_size = x->batch_size; // hardcoded to 1 for now
  int n = layer->n;
  int n_prev = layer->n_prev;
  int table_mask = TABLE_SIZE - 1;

  for (int j = 0; j < batch_size; j++) { // the outer loop could be removed
    for (int i = 0; i < n; i++) {
      t_float z = layer->biases[i];

      for (int k = 0; k < n_prev; k++) {
        z += (t_float)(layer->weights[i*n_prev+k] * input[k*batch_size+j]);
      }
      int idx = i*batch_size+j;
      layer->z_cache[idx] = z;
      t_float activation = apply_activation(x, layer, z);
      layer->a_cache[idx] = activation;
      if (l == x->num_layers - 1) {
        post("prediction: %f", activation);
        x->predictions[x->pred_phase] = activation;
        x->pred_phase = (x->pred_phase + 1) & (TABLE_SIZE - 1);
      }
    }
  }
}

static void model_forward(t_nn3 *x) {
  for (int l = 0; l < x->num_layers; l++) {
    t_layer *layer = &x->layers[l];
    t_float *input = (l == 0) ? x->x_features : layer->a_cache;
    layer_forward(x, l, layer, input);
  }
}

static void model_forward_info(t_nn3 *x) {
  t_float *current_features = x->x_features;
  t_float current_label = x->current_label;
  post("current label: %f", current_label);
  // for (int i = 0; i < x->num_features; i++) {
  //   post("feature: %d, %f", i, current_features[i]);
  // }
}


static void calculate_output_layer_da(t_nn3 *x, t_layer *layer) {
  int n = layer->n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < x->batch_size; j++) {
      int idx = i * x->batch_size + j;
      // For MSE loss: dL/dA = 2*(A-Y)/m, simplified to (A - Y)
      // layer->da[idx] = layer->a_cache[idx] - x->y_labels[idx];
      layer->da[idx] = layer->a_cache[idx] - x->current_label;
      post("da for outer layer: %f", layer->da[idx]);
      }
  }
}

static void calculate_da_prev(t_nn3 *x, int l, t_layer *layer)
{
  int n_neurons = layer->n;
  int n_prev = layer->n_prev;
  t_layer *prev_layer = &x->layers[l-1];

  // initialize previous layer da to 0
  memset(prev_layer->da, 0, sizeof(t_float) * n_prev * x->batch_size);

  // da_prev = W^T * dz
  for (int k = 0; k < n_prev; k++) {
    for (int j = 0; j < x->batch_size; j++) {
      for (int i = 0; i < n_neurons; i++) {
        prev_layer->da[k*x->batch_size+j] += layer->weights[i*n_prev+k] * 
                                            layer->dz[i*x->batch_size+j];
      }
    }
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
      return z > 0 ? 1.0 : leak; // note: update for leaky relu
    case ACTIVATION_LINEAR:
    default:
      return 1.0;
  }
}

static void calculate_dw(t_nn3 *x, int l, t_layer *layer) {
  int n = layer->n;
  int n_prev = layer->n_prev;
  int batch_size = x->batch_size;
  t_float *prev_activations = (l == 0) ? cos_table : x->layers[l-1].a_cache;

  // initialize dw to zero
  memset(layer->dw, 0, sizeof(t_float) * n * n_prev);

  // calculate gradients: dW = dZ × A_prev.T
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_prev; j++) {
      for (int k = 0; k < batch_size; k++) {
        // dZ[i,k] × A_prev[j,k]
        layer->dw[i*n_prev+j] += layer->dz[i*batch_size+k] * 
                                  prev_activations[j*batch_size+k];
      }
      layer->dw[i*n_prev+j] /= batch_size;

      // TODO: add L2 regularization term once enabled enabled
      // if (x->optimizer == OPTIMIZATION_L2 || x->optimizer == OPTIMIZATION_ADAM) {
      //   layer->dw[i*n_inputs+j] += x->lambda * layer->weights[i*n_inputs+j];
      // }
    }
  }
}

static void calculate_dz(t_nn3 *x, t_layer *layer) {
  int n = layer->n;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < x->batch_size; j++) {
      int idx = i * x->batch_size + j;
      layer->dz[idx] = layer->da[idx] *
       activation_derivative(layer->activation,
                             layer->z_cache[idx],
                             layer->a_cache[idx],
                             x->leak);
    }
  }
}

static void calculate_db(t_nn3 *x, t_layer *layer) {
  int n = layer->n;

  // initialize db to zero
  memset(layer->db, 0, sizeof(t_float) * n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < x->batch_size; j++) {
      layer->db[i] += layer->dz[i*x->batch_size+j];
    }
    layer->db[i] /= x->batch_size;
  }
}

static void update_parameters(t_nn3 *x) {
  for (int l = 0; l < x->num_layers; l++) {
    t_layer *layer = &x->layers[l];
    int n = layer->n;
    int n_prev = layer->n_prev;

    for (int i = 0; i < n * n_prev; i++) {
      layer->weights[i] -= x->alpha * layer->dw[i];
    }

    for (int i = 0; i < n; i++) {
      layer->biases[i] -= x->alpha * layer->db[i];
    }
  }
}

static void layer_backward(t_nn3 *x, int l, t_layer *layer) {
  if (l == x->num_layers - 1) {
    calculate_output_layer_da(x, layer);
  }

  calculate_dz(x, layer);
  calculate_dw(x, l, layer);
  calculate_db(x, layer);

  if (l > 0) {
    calculate_da_prev(x, l, layer);
  }
}

static void model_backward(t_nn3 *x) {
  for (int l = x->num_layers - 1; l >= 0; l--) {
    t_layer *layer = &x->layers[l];
    layer_backward(x, l, layer);
  }
}

static t_int *nn3_perform(t_int *w) {
  t_nn3 *x = (t_nn3 *)(w[1]);
  t_sample *x_freq_in = (t_sample *)(w[2]);
  t_sample *y_freq_in = (t_sample *)(w[3]);
  t_sample *out = (t_sample *)(w[4]);
  int n = (int)(w[5]);

  int bin_size = n; // a power of 2 (defaults to 64)
  int bin_mask = bin_size - 1;
  int bin_phase = x->bin_phase;
  double conv = x->conv;
  double x_phase = x->x_phase;
  double y_phase = x->y_phase;

  t_float *features_buffer = x->x_features;

  int table_mask = TABLE_SIZE - 1;

  // run forward prop
  // the input layer is the x->predictions buffer that was filled on the
  // previous pass
  // the y label is the x->current_label value that was set on the previous pass
  // batch size is 1
  if (x->features_buffer_filled) {

  // model_forward(x);
  // model_backward(x);
  // update_parameters(x);

  }

  // update x->current_label
  unsigned int y_idx = (unsigned int)y_phase;
  double y_frac = y_phase - (double)y_idx;
  y_idx &= table_mask;

  t_sample y1 = cos_table[y_idx];
  t_sample y2 = cos_table[(y_idx + 1) & table_mask];
  t_sample y = y1 + y_frac * (y2 - y1);
  x->current_label = y;

  // advance phase pointers and fill the predictions buffer
  while (n--) {
    unsigned int f_idx = (unsigned int)x_phase;
    double f_frac = x_phase - (double)f_idx;
    f_idx &= table_mask;

    t_sample f1 = cos_table[f_idx];
    t_sample f2 = cos_table[(f_idx + 1) & table_mask];
    t_sample f = f1 + f_frac * (f2 - f1);
    features_buffer[bin_phase] = f;

    unsigned int p_idx = (unsigned int)y_phase;
    double p_frac = y_phase - (double)p_idx;
    p_idx &= table_mask;

    t_sample p1 = x->predictions[p_idx];
    t_sample p2 = x->predictions[(p_idx + 1) & table_mask];
    t_sample p = p1 + p_frac * (p2 - p1);
    *out++ = p;

    y_phase += *y_freq_in++ * conv;
    // x_phase += *x_freq_in++ * conv * bin_size;
    x_phase += *x_freq_in++ * conv;
    bin_phase = (bin_phase + 1) & bin_mask;
  }
  while (x_phase >= TABLE_SIZE) x_phase -= TABLE_SIZE;
  while (x_phase < 0) x_phase += TABLE_SIZE; // could this happen?
  x->x_phase = x_phase;

  while (y_phase >= TABLE_SIZE) y_phase -= TABLE_SIZE;
  while (y_phase < 0) y_phase += TABLE_SIZE;
  x->y_phase = y_phase;

  x->bin_phase = bin_phase;
  x->features_buffer_filled = 1;

  return (w+6);
}

static void nn3_dsp(t_nn3 *x, t_signal **sp) {
  x->conv = (double)((double)TABLE_SIZE / (double)sp[0]->s_sr);
  // update input layer dimension
  // x->layer_dims[0] = (int)(sp[0]->s_length);
  // update num_features
  // x->num_features = (int)(sp[0]->s_length);
  // static int features_buffer_set = 0;
  // if (!features_buffer_set) {
  //   x->x_features = getbytes(sizeof(t_float) * x->num_features);
  //   if (!x->x_features) {
  //     pd_error(x, "nn3~: unable to allocate memory to features buffer");
  //     return;
  //   }
  //   features_buffer_set = 1;
  // }
  dsp_add(nn3_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_length);
}

void nn3_tilde_setup(void) {
  nn3_class = class_new(gensym("nn3~"),
                        (t_newmethod)nn3_new,
                        (t_method)nn3_free,
                        sizeof(t_nn3),
                        CLASS_DEFAULT,
                        A_GIMME, 0);
  class_addmethod(nn3_class, (t_method)nn3_dsp, gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(nn3_class, t_nn3, x_f);
}

