#include "m_pd.h"
#include <math.h>

/**
 * notes:
 * - lorenz_systems.md
 * just outputs the z element
 **/

typedef struct _lorenz_z {
  t_object x_obj;

  t_float x_x;
  t_float x_y;
  t_float x_z;

  t_float x_dt;

  t_float x_sigma;
  t_float x_rho;
  t_float x_beta;

  t_float x_min;
  t_float x_max;
  t_float x_scale;

  t_inlet *x_dt_inlet;
  t_outlet *x_z_outlet;

} t_lorenz_z;

static t_class *lorenz_z_class = NULL;

static void reset(t_lorenz_z *x);

static void *lorenz_z_new(void) {
  t_lorenz_z *x = (t_lorenz_z *)pd_new(lorenz_z_class);

  x->x_x = (t_float)1.0;
  x->x_y = (t_float)1.0;
  x->x_z = (t_float)1.0;

  x->x_dt = (t_float)0.0001;

  x->x_sigma = (t_float)10.0;
  x->x_rho = (t_float)28.0;
  x->x_beta = (t_float)((t_float)8.0 / (t_float)3.0);

  x->x_min = (t_float)0.0;
  x->x_max = (t_float)0.0;
  x->x_scale = (t_float)1.0;

  x->x_dt_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  pd_float((t_pd *)x->x_dt_inlet, x->x_dt);

  x->x_z_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void lorenz_z_free(t_lorenz_z *x) {
  if (x->x_z_outlet) outlet_free(x->x_z_outlet);
  if (x->x_dt_inlet) inlet_free(x->x_dt_inlet);
}

static inline t_float fast_reciprocal(t_float x) {
    // Get initial estimate using bit manipulation (IEEE 754 magic)
    union {
        t_float f;
        int32_t i;
    } val;
    
    val.f = x;
    val.i = 0x5F3759DF - (val.i >> 1); // Magic number from Quake III
    
    // One Newton-Raphson iteration for better accuracy
    val.f = val.f * (2.0f - x * val.f);
    
    return val.f;
}

static t_int *lorenz_z_perform(t_int *w) {
  t_lorenz_z *x = (t_lorenz_z *)(w[1]);
  t_sample *in_dt = (t_sample *)(w[2]);
  t_sample *out_z = (t_sample *)(w[3]);
  int n = (int)(w[4]);

  t_float x_x = x->x_x;
  t_float x_y = x->x_y;
  t_float x_z = x->x_z;

  t_float sigma = x->x_sigma;
  t_float rho = x->x_rho;
  t_float beta = x->x_beta;

  t_float min = x->x_min;
  t_float max = x->x_max;
  // try only updating the scale at the start of the DSP loop
  // t_float scale = x->x_scale / (max - min);

  // t_float range = max - min;
  // t_float recip_range = fast_reciprocal(range);
  // t_float scale = x->x_scale * recip_range;


  while (n--) {
    t_sample dt = *in_dt++;
    // it explodes at 0.1. I'll find a safe max value
    if (dt > (t_float)0.01) dt = (t_float)0.01;
    if (dt < (t_float)1e-8) dt = (t_float)1e-8;

    t_float dx = sigma * (x_y - x_x);
    t_float dy = x_x * (rho - x_z) - x_y;
    t_float dz = x_x * x_y - beta * x_z;

    x_x += dx * dt;
    x_y += dy * dt;
    x_z += dz * dt;

    min = x_x;
    if (x_y < min) min = x_y;
    if (x_z < min) min = x_z;

    max = x_x;
    if (x_y > max) max = x_y;
    if (x_z > max) max = x_z;

    *out_z++ = x_z;
  }
  x->x_x = x_x;
  x->x_y = x_y;
  x->x_z = x_z;
  x->x_min = min;
  x->x_max = max;

  return (w+5);
}

static void lorenz_z_dsp(t_lorenz_z *x, t_signal **sp) {
  dsp_add(lorenz_z_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_length);
}

static void reset(t_lorenz_z *x) {
  x->x_x = (t_float)1.0;
  x->x_y = (t_float)1.0;
  x->x_z = (t_float)1.0;
  x->x_min = (t_float)0.0;
  x->x_max = (t_float)0.0;
}

static void set_sigma(t_lorenz_z *x, t_floatarg f) {
  x->x_sigma = f;
}

static void set_rho(t_lorenz_z *x, t_floatarg f) {
  x->x_rho = f;
}

static void set_beta(t_lorenz_z *x, t_floatarg f) {
  x->x_beta = f;
}

void lorenz_z_tilde_setup(void) {
  lorenz_z_class = class_new(gensym("lorenz_z~"),
                           (t_newmethod)lorenz_z_new,
                           (t_method)lorenz_z_free,
                           sizeof(t_lorenz_z),
                           CLASS_DEFAULT,
                           0);

  class_addmethod(lorenz_z_class, (t_method)lorenz_z_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(lorenz_z_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(lorenz_z_class, (t_method)set_sigma, gensym("sigma"), A_FLOAT, 0);
  class_addmethod(lorenz_z_class, (t_method)set_rho, gensym("rho"), A_FLOAT, 0);
  class_addmethod(lorenz_z_class, (t_method)set_beta, gensym("beta"), A_FLOAT, 0);

  CLASS_MAINSIGNALIN(lorenz_z_class, t_lorenz_z, x_dt);
}
