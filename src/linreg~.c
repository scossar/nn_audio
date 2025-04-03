#include "m_pd.h"

#define MAX_FEATURES 256

typedef struct _linreg {
  t_object x_obj;

  int x_num_inputs;
  t_float *x_weights;
  t_float *x_ring_buffer;
  t_float *x_features_buffer;
  int x_ring_pos;
  int x_ring_buffer_size;
  int x_samp_buffer_size;
  int x_buffer_mask;
  t_float x_bias;
  t_float x_alpha;

  t_inlet *x_y_inlet;
} t_linreg;

static t_class *linreg_class = NULL;

static t_int *linreg_perform(t_int *w)
{
  t_linreg *x = (t_linreg *)(w[1]);
  t_sample *in_x = (t_float *)(w[2]);
  t_sample *in_y = (t_float *)(w[3]);
  t_sample *out = (t_float *)(w[4]);
  int n = (int)(w[5]);
  int pos = x->x_ring_pos;
  int samp_buffer_size = x->x_samp_buffer_size;
  int ring_buffer_size = x->x_ring_buffer_size;
  int mask = x->x_buffer_mask;
  t_float alpha = x->x_alpha;
  int num_inputs = x->x_num_inputs;

  while (n--) {
    t_float x_in = *in_x++;
    t_float y_in = *in_y++;

    x->x_ring_buffer[pos] = x_in;

    t_float z = x->x_bias;
    for (int i = 0; i < num_inputs; i++) {
      int offset = (pos - i * samp_buffer_size + ring_buffer_size) & mask;
      x->x_features_buffer[i] = x->x_ring_buffer[offset];
      z += x->x_features_buffer[i] * x->x_weights[i];
    }

    t_float dz = z - y_in;
    for (int i = 0; i < num_inputs; i++) {
      x->x_weights[i] -= alpha * dz * x->x_features_buffer[i];
    }
    x->x_bias -= alpha * dz;

    pos = (pos + 1) & mask;
    *out++ = z;
  }

  x->x_ring_pos = pos;
  return (w+6);
}

static void linreg_dsp(t_linreg *x, t_signal **sp)
{
  dsp_add(linreg_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_length);
}

static void linreg_print_params(t_linreg *x)
{
  for (int i = 0; i < x->x_num_inputs; i++) {
    post("linreg~: w[%d]: %f", i, x->x_weights[i]);
  }

  post("linreg~: b: %f", x->x_bias);
}

static int linreg_initialize_weights(t_linreg *x)
{
  x->x_weights = (t_float *)getbytes(sizeof(t_float) * x->x_num_inputs);
  if (x->x_weights == NULL) {
    pd_error(x, "linreg~: failed to allocate memory for weights");
    return 0;
  }

  return 1;
}

// see pure_data_linear_regression_audio_object.md (the ring buffer size needs
// to be a power of 2 for bit masking)
static int linreg_initialize_ring_buffer(t_linreg *x)
{
  int buffer_size = 1;
  while (buffer_size < x->x_num_inputs * x->x_samp_buffer_size) {
    buffer_size *= 2;
  }

  x->x_ring_buffer_size = buffer_size;
  x->x_ring_buffer = (t_float *)getbytes(sizeof(t_float) * x->x_ring_buffer_size);
  if (x->x_ring_buffer == NULL) {
    pd_error(x, "linreg~: failed to allocate memory to ring buffer");
    return 0;
  }

  x->x_ring_pos = 0;
  x->x_buffer_mask = x->x_ring_buffer_size - 1;

  return 1;
}

static int linreg_initialize_features_buffer(t_linreg *x)
{
  x->x_features_buffer = (t_float *)getbytes(sizeof(t_float) * MAX_FEATURES);
  if (x->x_features_buffer == NULL) {
    pd_error(x, "linreg~: failed to allocate memory to features buffer");
    return 0;
  }

  return 1;
}

static void linreg_set_alpha(t_linreg *x, t_floatarg f)
{
  if (f <= 0 || f >= 0.1) {
    post("linreg~: alpha needs to be in the range (0, 0.1). Setting to 0.0001");
    f = 0.0001f;
  }
  x->x_alpha = f;
}


static void linreg_reset_params(t_linreg *x)
{
  for (int i = 0; i < x->x_num_inputs; i++) {
    x->x_weights[i] = 0.0f;
  }

  x->x_bias = 0.0f;
}

static void linreg_free(t_linreg *x)
{
  if (x->x_weights != NULL) {
    freebytes(x->x_weights, x->x_num_inputs * sizeof(t_float));
    x->x_weights = NULL;
  }

  if (x->x_ring_buffer != NULL) {
    freebytes(x->x_ring_buffer, x->x_ring_buffer_size * sizeof(t_float));
    x->x_ring_buffer = NULL;
  }

  if (x->x_features_buffer != NULL) {
    freebytes(x->x_features_buffer, MAX_FEATURES * sizeof(t_float));
    x->x_features_buffer = NULL;
  }

  if (x->x_y_inlet != NULL) {
    inlet_free(x->x_y_inlet);
  }
}

static void *linreg_new(t_floatarg f)
{
  t_linreg *x = (t_linreg *)pd_new(linreg_class);

  x->x_num_inputs = (f <= 256) ? f : 256; // todo: print warning if exceeded
  x->x_alpha = 0.0001f; // default;
  x->x_bias = 0.0f; // initial value

  x->x_samp_buffer_size = sys_getblksize();

  if (!linreg_initialize_weights(x)) {
    linreg_free(x);
    return NULL;
  }

  if (!linreg_initialize_ring_buffer(x)) {
    linreg_free(x);
    return NULL;
  }

  if (!linreg_initialize_features_buffer(x)) {
    linreg_free(x);
    return NULL;
  }

  post("ring buffer size: %d", x->x_ring_buffer_size);
  post("samp buffer size: %d", x->x_samp_buffer_size);

  x->x_y_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);

  outlet_new(&x->x_obj, gensym("signal"));

  return (void *)x;
}

void linreg_tilde_setup(void)
{
  linreg_class = class_new(gensym("linreg~"),
                           (t_newmethod)linreg_new,
                           (t_method)linreg_free,
                           sizeof(t_linreg),
                           CLASS_DEFAULT,
                           A_DEFFLOAT , 0);

  class_addmethod(linreg_class, (t_method)linreg_dsp, gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(linreg_class, t_linreg, x_num_inputs);

  class_addmethod(linreg_class, (t_method)linreg_print_params, gensym("print_params"), 0);
  class_addmethod(linreg_class, (t_method)linreg_reset_params, gensym("clear"), 0);
  class_addmethod(linreg_class, (t_method)linreg_set_alpha, gensym("set_alpha"), A_DEFFLOAT, 0);
}
