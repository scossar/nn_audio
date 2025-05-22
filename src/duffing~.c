#include "m_pd.h"
#include <math.h>

/**
 * notes:
 * an RK4 implementation of the Duffing equation
 * notes:
 * - duffing_equation.md
 **/

#define WAVETABLE_SIZE 4096 // 2^12
#define WAVETABLE_MASK (WAVETABLE_SIZE-1)
static float *cos_table = NULL;
static int table_reference_count = 0;

static t_class *duffing_class = NULL;

typedef struct _duffing {
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

  t_float x_recip_6;
  t_float x_recip_2pi;

  t_outlet *x_duffing_outlet;
} t_duffing;

static void wavetable_init(void) {
  if (cos_table == NULL) {
    cos_table = (float *)getbytes(sizeof(float) * WAVETABLE_SIZE);
    if (cos_table) {
      for (int i = 0; i < WAVETABLE_SIZE; i++) {
        cos_table[i] = cosf((i * 2.0f * (float)M_PI) / (float)WAVETABLE_SIZE);
      }
      post("duffing~: initialized cosine table of size %d", WAVETABLE_SIZE);
    } else {
      post("duffing~ error: failed to allocate memory for cosine table");
    }
  }
  table_reference_count++;
}

static void wavetable_free(void) {
  table_reference_count--;
  if (table_reference_count <= 0 && cos_table != NULL) {
    freebytes(cos_table, sizeof(float) * WAVETABLE_SIZE);
    cos_table = NULL;
    post("duffing~: freed cosine table");
    table_reference_count = 0; // just to be safe
  }
}

static void *duffing_new(t_symbol *s, int argc, t_atom *argv) {
  t_duffing *x = (t_duffing *)pd_new(duffing_class);

  t_float x_x, x_v;
  if (argc < 2) {
    x_x = (t_float)0.1;
    x_v = (t_float)0.0;
  } else {
    x_x = atom_getfloat(argv++);
    x_v = atom_getfloat(argv++);
  }

  x->x_x = x_x;
  x->x_v = x_v;
  x->x_t = (t_float)0.0;

  x->x_delta = (t_float)0.2;
  x->x_alpha = (t_float)-1.0;
  x->x_beta = (t_float)1.0;
  x->x_gamma = (t_float)0.3;
  x->x_omega = (t_float)1.0;

  x->x_conv = (t_float)0.0;
  x->x_f = (t_float)1.0;

  x->x_recip_6 = (t_float)1.0 / (t_float)6.0;
  x->x_recip_2pi = (t_float)1.0 / (t_float)((t_float)2.0 * (t_float)M_PI);

  wavetable_init();

  x->x_duffing_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void duffing_free(t_duffing *x) {
  if (x->x_duffing_outlet) outlet_free(x->x_duffing_outlet);

  wavetable_free();
}

static inline t_float fast_cos(t_duffing *x, t_float phase) {
  float *costab = cos_table;
  // normalize phase to [0, 1]
  phase = phase * x->x_recip_2pi;  // 1 / (2 * M_PI)
  phase -= floorf(phase);
  phase *= WAVETABLE_SIZE;

  int idx = (int)phase & WAVETABLE_MASK;
  int idx_next = (idx + 1) & WAVETABLE_MASK;
  t_float frac = phase - floorf(phase);

  return costab[idx] * ((t_float)1.0 - frac) + costab[idx_next] * frac;
}

static t_int *duffing_perform(t_int *w) {
  t_duffing *x = (t_duffing *)(w[1]);
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

  t_float conv = x->x_conv; // 1.0 / sample_rate

  t_float recip_6 = x->x_recip_6; // 1.0/6.0

  while (n--) {
    t_sample f = *in_f++;
    *out_x++ = x_x;

    t_sample dt = f * conv;
    t_float half_dt = (t_float)0.5 * dt;
    t_float temp_x;

    // first derivative
    t_float k1_x = x_v;
    t_float k1_v = -delta * x_v - alpha * x_x - beta * (x_x * x_x * x_x)
      + gamma * fast_cos(x, omega * x_t);

    // second derivative
    temp_x = x_x + half_dt * k1_x;
    t_float k2_x = x_v + half_dt * k1_v;
    t_float mid_time = x_t + half_dt;
    t_float cos_mid_time = fast_cos(x, omega * mid_time);
    t_float k2_v = -delta * (x_v + half_dt * k1_v)
      - alpha * temp_x
      - beta * (temp_x * temp_x * temp_x)
      + gamma * cos_mid_time;

    // third derivative
    temp_x = x_x + half_dt * k2_x;
    t_float k3_x = x_v + half_dt * k2_v;
    t_float k3_v = -delta * (x_v + half_dt * k2_v)
      - alpha * temp_x
      - beta * (temp_x * temp_x * temp_x)
      + gamma * cos_mid_time;

    // fourth derivative
    temp_x = x_x + dt * k3_x;
    t_float k4_x = x_v + dt * k3_v;
    t_float k4_v = -delta * (x_v + dt * k3_v)
      - alpha * temp_x
      - beta * (temp_x * temp_x * temp_x)
      + gamma * fast_cos(x, omega * (x_t + dt));

    // update state with weighted averages
    x_x += dt * recip_6 * (k1_x + (t_float)2.0 * k2_x + (t_float)2.0 * k3_x + k4_x);
    x_v += dt * recip_6 * (k1_v + (t_float)2.0 * k2_v + (t_float)2.0 * k3_v + k4_v);
    x_t += dt;
  }

  x->x_x = x_x;
  x->x_v = x_v;
  x->x_t = x_t;

  return (w+5);
}

static void duffing_dsp(t_duffing *x, t_signal **sp) {
  x->x_conv = (t_float)((t_float)1.0 / sp[0]->s_sr);
  dsp_add(duffing_perform, 4,
          x,
          sp[0]->s_vec,
          sp[1]->s_vec,
          sp[0]->s_length);
}

static void set_delta(t_duffing *x, t_floatarg f) {
  x->x_delta = f;
}

static void set_alpha(t_duffing *x, t_floatarg f) {
  x->x_alpha = f;
}

static void set_beta(t_duffing *x, t_floatarg f) {
  x->x_beta = f;
}

static void set_gamma(t_duffing *x, t_floatarg f) {
  x->x_gamma = f;
}

static void set_omega(t_duffing *x, t_floatarg f) {
  x->x_omega = f;
}

static void reset(t_duffing *x) {
  x->x_x = (t_float)0.1;
  x->x_v = (t_float)0.0;
  x->x_t = (t_float)0.0;
}

void duffing_tilde_setup(void) {
  duffing_class = class_new(gensym("duffing~"),
                            (t_newmethod)duffing_new,
                            (t_method)duffing_free,
                            sizeof(t_duffing),
                            CLASS_DEFAULT,
                            A_GIMME,
                            0);
  class_addmethod(duffing_class, (t_method)duffing_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(duffing_class, (t_method)set_delta, gensym("delta"), A_FLOAT, 0);
  class_addmethod(duffing_class, (t_method)set_alpha, gensym("alpha"), A_FLOAT, 0);
  class_addmethod(duffing_class, (t_method)set_beta, gensym("beta"), A_FLOAT, 0);
  class_addmethod(duffing_class, (t_method)set_gamma, gensym("gamma"), A_FLOAT, 0);
  class_addmethod(duffing_class, (t_method)set_omega, gensym("omega"), A_FLOAT, 0);
  class_addmethod(duffing_class, (t_method)reset, gensym("reset"), 0);

  CLASS_MAINSIGNALIN(duffing_class, t_duffing, x_f);
}


