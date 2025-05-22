#include "m_pd.h"

/**
 * notes:
 * - lorenz_systems.md
 **/

typedef struct _lorenzrk4 {
  t_object x_obj;

  t_float x_x;
  t_float x_y;
  t_float x_z;

  t_float x_init_x;
  t_float x_init_y;
  t_float x_init_z;

  t_float x_dt;

  t_float x_sigma;
  t_float x_rho;
  t_float x_beta;

  t_float x_2_recip;
  t_float x_6_recip;

  t_outlet *x_z_outlet;
  t_outlet *x_y_outlet;
  t_outlet *x_x_outlet; // when more than one signal outlet is needed, all
  // signal outlets need to be explicitly created
} t_lorenzrk4;

static t_class *lorenzrk4_class = NULL;

static void reset(t_lorenzrk4 *x);

static void *lorenzrk4_new(t_symbol *s, int argc, t_atom *argv) {
  t_lorenzrk4 *x = (t_lorenzrk4 *)pd_new(lorenzrk4_class);

  t_float x_x, x_y, x_z;

  if (argc < 3) {
    x_x = (t_float)1.0;
    x_y = (t_float)1.0;
    x_z = (t_float)1.0;
  } else {
    x_x = atom_getfloat(argv++);
    x_y = atom_getfloat(argv++);
    x_z = atom_getfloat(argv++);
  }

  x->x_x = x_x;
  x->x_init_x = x_x;
  x->x_y = x_y;
  x->x_init_y = x_y;
  x->x_z = x_z;
  x->x_init_z = x_z;

  post("lorenzrk4~: x_init: %f", x->x_x);
  post("lorenzrk4~: y_init: %f", x->x_y);
  post("lorenzrk4~: z_init: %f", x->x_z);

  x->x_dt = (t_float)0.0001;

  x->x_sigma = (t_float)10.0;
  x->x_rho = (t_float)28.0;
  x->x_beta = (t_float)((t_float)8.0 / (t_float)3.0);

  x->x_2_recip = (t_float)0.5; // for completeness
  x->x_6_recip = (t_float)((t_float)1.0 / (t_float)6.0);

  x->x_z_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_y_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_x_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void lorenzrk4_free(t_lorenzrk4 *x) {
  if (x->x_z_outlet) outlet_free(x->x_z_outlet);
  if (x->x_y_outlet) outlet_free(x->x_y_outlet);
  if (x->x_x_outlet) outlet_free(x->x_x_outlet);
}

static t_int *lorenzrk4_perform(t_int *w) {
  t_lorenzrk4 *x = (t_lorenzrk4 *)(w[1]);
  t_sample *in_dt = (t_sample *)(w[2]);
  t_sample *out_x = (t_sample *)(w[3]);
  t_sample *out_y = (t_sample *)(w[4]);
  t_sample *out_z = (t_sample *)(w[5]);
  int n = (int)(w[6]);

  t_float x_x = x->x_x;
  t_float x_y = x->x_y;
  t_float x_z = x->x_z;

  t_float sigma = x->x_sigma;
  t_float rho = x->x_rho;
  t_float beta = x->x_beta;

  const t_float recip_2 = x->x_2_recip;
  const t_float recip_6 = x->x_6_recip;

  while (n--) {
    t_sample dt = *in_dt++;
    // it explodes at values around 0.1
    if (dt > (t_float)0.1) dt = (t_float)0.1;
    if (dt <= (t_float)0.0) dt = (t_float)1e-10;

    t_float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    t_float x_temp, y_temp, z_temp;

    k1x = sigma * (x_y - x_x);
    k1y = x_x * (rho - x_z) - x_y;
    k1z = x_x * x_y - beta * x_z;

    x_temp = x_x + dt * k1x * recip_2;
    y_temp = x_y + dt * k1y * recip_2;
    z_temp = x_z + dt * k1z * recip_2;

    k2x = sigma * (y_temp - x_temp);
    k2y = x_temp * (rho - z_temp) - y_temp;
    k2z = x_temp * y_temp - beta * z_temp;

    x_temp = x_x + dt * k2x * recip_2;
    y_temp = x_y + dt * k2y * recip_2;
    z_temp = x_z + dt * k2z * recip_2;

    k3x = sigma * (y_temp - x_temp);
    k3y = x_temp * (rho - z_temp) - y_temp;
    k3z = x_temp * y_temp - beta * z_temp;

    x_temp = x_x + dt * k3x;
    y_temp = x_y + dt * k3y;
    z_temp = x_z + dt * k3z;

    k4x = sigma * (y_temp - x_temp);
    k4y = x_temp * (rho - z_temp) - y_temp;
    k4z = x_temp * y_temp - beta * z_temp;

    x_x += dt * (k1x + 2*k2x + 2*k3x + k4x) * recip_6;
    x_y += dt * (k1y + 2*k2y + 2*k3y + k4y) * recip_6;
    x_z += dt * (k1z + 2*k2z + 2*k3z + k4z) * recip_6;

    *out_x++ = x_x;
    *out_y++ = x_y;
    *out_z++ = x_z;
  }
  x->x_x = x_x;
  x->x_y = x_y;
  x->x_z = x_z;

  return (w+7);
}

static void lorenzrk4_dsp(t_lorenzrk4 *x, t_signal **sp) {
  dsp_add(lorenzrk4_perform, 6,
          x,
          sp[0]->s_vec, // dt
          sp[1]->s_vec, // x_outlet
          sp[2]->s_vec, // y_outlet
          sp[3]->s_vec, // z_outlet
          sp[0]->s_length);
}

static void reset(t_lorenzrk4 *x) {
  x->x_x = x->x_init_x;
  x->x_y = x->x_init_y;
  x->x_z = x->x_init_z;
}

static void set_sigma(t_lorenzrk4 *x, t_floatarg f) {
  x->x_sigma = f;
}

static void set_rho(t_lorenzrk4 *x, t_floatarg f) {
  x->x_rho = f;
}

static void set_beta(t_lorenzrk4 *x, t_floatarg f) {
  x->x_beta = f;
}

void lorenzrk4_tilde_setup(void) {
  lorenzrk4_class = class_new(gensym("lorenzrk4~"),
                           (t_newmethod)lorenzrk4_new,
                           (t_method)lorenzrk4_free,
                           sizeof(t_lorenzrk4),
                           CLASS_DEFAULT,
                           A_GIMME,
                           0);

  class_addmethod(lorenzrk4_class, (t_method)lorenzrk4_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(lorenzrk4_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(lorenzrk4_class, (t_method)set_sigma, gensym("sigma"), A_FLOAT, 0);
  class_addmethod(lorenzrk4_class, (t_method)set_rho, gensym("rho"), A_FLOAT, 0);
  class_addmethod(lorenzrk4_class, (t_method)set_beta, gensym("beta"), A_FLOAT, 0);

  CLASS_MAINSIGNALIN(lorenzrk4_class, t_lorenzrk4, x_dt);
}
