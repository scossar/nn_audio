#include "m_pd.h"

/**
 * notes:
 * - pd_audio_rate_neural_network.md
 **/

t_class *esn_class = NULL;

typedef struct _esn {
  t_object x_obj;

  int x_reservoir_size;
  int x_n_samples;

  t_sample *x_reservoir;
  t_sample *x_w_in;
  t_sample *x_w;
  t_sample *x_w_out;

  t_inlet *x_example_inlet;
  t_inlet *x_label_inlet;
  t_outlet *x_orig;
  t_outlet *x_pred;
} t_esn;

static void buffers_init(t_esn *x) {
  x->x_reservoir = getbytes(sizeof(t_sample) * x->x_reservoir_size);
  if (x->x_reservoir == NULL) {
    pd_error(x, "esn~: failed to allocate reservoir buffer memory");
    return;
  }

  x->x_w_in = getbytes(sizeof(t_sample) * x->x_reservoir_size);
  if (x->x_w_in == NULL) {
    pd_error(x, "esn~: failed to allocate w_in buffer memory");
    return;
  }

  // shape (conceptual) (reservoir_size, reservoir_size)
  x->x_w = getbytes(sizeof(t_sample) * x->x_reservoir_size * x->x_reservoir_size);
  if (x->x_w == NULL) {
    pd_error(x, "esn~: failed to allocate w buffer memory");
    return;
  }

  x->x_w_out = getbytes(sizeof(t_sample) * x->x_reservoir_size);
  if (x->x_w_out == NULL) {
    pd_error(x, "esn~: failed to allocate w_out buffer memory");
    return;
  }
}

static void *esn_new(void) {
  t_esn *x = (t_esn *)pd_new(esn_class);

  x->x_reservoir_size = 1024; // 2^10
  x->x_n_samples = 8192; // 2^13; for context 48000 / 8 = 8000

  x->x_reservoir = NULL;
  x->x_w_in = NULL;
  x->x_w = NULL;
  x->x_w_out = NULL;

  buffers_init(x);

  return (void *)x;
}


