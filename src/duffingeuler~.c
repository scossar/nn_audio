#include "m_pd.h"
#include <math.h>

/**
 * notes:
 * - duffing_equation.md
 * - a Eular method solution (numerically unstable)
 **/

typedef struct _duffingeuler {
  t_object x_obj;

  t_float x_x; // position
  t_float x_v; // velocity
  t_float x_t; // time

  t_float x_delta; // damping coefficient
  t_float x_alpha; // linear stiffness
  t_float x_beta; // non-linear stiffness
  t_float x_gamma; // forcing amplitude
  t_float x_omega; // forcing frequency

  t_float x_conv; // frequency conversion for dt
  t_float x_f; // initial frequency

  t_outlet *x_duffingeuler_outlet;
} t_duffingeuler;

static t_class *duffingeuler_class = NULL;

static void *duffingeuler_new(void) {
  t_duffingeuler *x = (t_duffingeuler *)pd_new(duffingeuler_class);

  x->x_x = (t_float)0.1;
  x->x_v = (t_float)0.0;
  x->x_t = (t_float)0.0;

  x->x_delta = (t_float)0.2;
  x->x_alpha = (t_float)-1.0;
  x->x_beta = (t_float)1.0;
  x->x_gamma = (t_float)0.3;
  x->x_omega = (t_float)1.0;

  x->x_conv = (t_float)0.0;
  x->x_f = (t_float)1.0;

  x->x_duffingeuler_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void duffingeuler_free(t_duffingeuler *x) {
  if (x->x_duffingeuler_outlet) outlet_free(x->x_duffingeuler_outlet);
}

static t_int *duffingeuler_perform(t_int *w) {
  t_duffingeuler *x = (t_duffingeuler *)(w[1]);
  t_sample *in_f = (t_sample *)(w[2]);
  t_sample *out_x = (t_sample *)(w[3]);
  int n = (int)(w[4]);

  t_float x_x = x->x_x;
  t_float x_v = x->x_v;
  t_float x_t = x->x_t;

  t_float delta = x->x_delta;
  t_float alpha = x->x_alpha;
  t_float beta = x->x_beta;
  t_float gamma = x->x_gamma;
  t_float omega = x->x_omega;

  t_float conv = x->x_conv;

  while (n--) {
    t_sample f = *in_f++;
    t_float dt = f * conv;
    t_float dx_dt = x_v; // technically correct, but maybe just use the x_v
    // variable?
    t_float dv_dt = -delta*x_v - alpha*x_x - beta * (x_x*x_x*x_x) +
      gamma*cosf(omega*x_t);

    x_x += dx_dt * dt;
    x_v += dv_dt * dt;
    x_t += dt;

    *out_x++ = x_x;
  }
  x->x_x = x_x;
  x->x_v = x_v;
  x->x_t = x_t;

  return (w+5);
}

static void duffingeuler_dsp(t_duffingeuler *x, t_signal **sp) {
  x->x_conv = (t_float)((t_float)1.0 / sp[0]->s_sr);
  dsp_add(duffingeuler_perform, 4,
          x,
          sp[0]->s_vec,
          sp[1]->s_vec,
          sp[0]->s_length);
}

static void set_delta(t_duffingeuler *x, t_floatarg f) {
  x->x_delta = f;
}

static void set_alpha(t_duffingeuler *x, t_floatarg f) {
  x->x_alpha = f;
}

static void set_beta(t_duffingeuler *x, t_floatarg f) {
  x->x_beta = f;
}

static void set_gamma(t_duffingeuler *x, t_floatarg f) {
  x->x_gamma = f;
}

static void set_omega(t_duffingeuler *x, t_floatarg f) {
  x->x_omega = f;
}

static void reset(t_duffingeuler *x) {
  x->x_x = (t_float)0.1;
  x->x_v = (t_float)0.0;
  x->x_t = (t_float)0.0;
}

void duffingeuler_tilde_setup(void) {
  duffingeuler_class = class_new(gensym("duffingeuler~"),
                            (t_newmethod)duffingeuler_new,
                            (t_method)duffingeuler_free,
                            sizeof(t_duffingeuler),
                            CLASS_DEFAULT,
                            0);
  class_addmethod(duffingeuler_class, (t_method)duffingeuler_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(duffingeuler_class, (t_method)set_delta, gensym("delta"), A_FLOAT, 0);
  class_addmethod(duffingeuler_class, (t_method)set_alpha, gensym("alpha"), A_FLOAT, 0);
  class_addmethod(duffingeuler_class, (t_method)set_beta, gensym("beta"), A_FLOAT, 0);
  class_addmethod(duffingeuler_class, (t_method)set_gamma, gensym("gamma"), A_FLOAT, 0);
  class_addmethod(duffingeuler_class, (t_method)set_omega, gensym("omega"), A_FLOAT, 0);
  class_addmethod(duffingeuler_class, (t_method)reset, gensym("reset"), 0);

  CLASS_MAINSIGNALIN(duffingeuler_class, t_duffingeuler, x_f);
}


