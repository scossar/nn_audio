#include "m_pd.h"
#include <math.h>

/**
 * implements a double pendulum system
 * notes: double_pendulum.md
 **/

typedef struct _doublependulum {
  t_object x_obj;

  t_float x_theta1;
  t_float x_theta2;
  t_float x_w1; // note that w is "omega"
  t_float x_w2;

  t_float x_theta1_init;
  t_float x_theta2_init;
  t_float x_w1_init;
  t_float x_w2_init;

  t_float x_m1;
  t_float x_m2;
  t_float x_L1;
  t_float x_L2;
  t_float x_g;

  t_float x_recip2pi;
  t_float x_half_pi;
  t_float x_conv;
  t_float x_f;


  t_outlet *x_theta1_x_outlet;
  t_outlet *x_theta1_y_outlet;
  t_outlet *x_theta2_x_outlet;
  t_outlet *x_theta2_y_outlet;

} t_doublependulum;

static t_class *doublependulum_class = NULL;

#define WAVETABLE_SIZE 4096 // 2^12
#define WAVETABLE_MASK (WAVETABLE_SIZE-1)
#define QUARTER_TABLE (WAVETABLE_SIZE / 4)

static float *cos_table = NULL;
static int table_reference_count = 0;

static void wavetable_init(void) {
  if (cos_table == NULL) {
    cos_table = (float *)getbytes(sizeof(float) * WAVETABLE_SIZE);
    if (cos_table) {
      for (int i = 0; i < WAVETABLE_SIZE; i++) {
        cos_table[i] = cosf((i * 2.0f * (float)M_PI) / (float)WAVETABLE_SIZE);
      }
      post("doublependulum~: initialized cosine table of size %d", WAVETABLE_SIZE);
    } else {
      post("doublependulum~ error: failed to allocate memory for cosine table");
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

static void *doublependulum_new(void) {
  t_doublependulum *x = (t_doublependulum *)pd_new(doublependulum_class);

  // these need to be set to reasonable values (not sure yet what those are)
  x->x_theta1 = (t_float)((t_float)M_PI / (t_float)2.0);
  x->x_theta2 = x->x_theta1;
  x->x_w1 = (t_float)0.0;
  x->x_w2 = (t_float)0.0;

  x->x_theta1_init = x->x_theta1;
  x->x_theta2_init = x->x_theta2;
  x->x_w1_init = x->x_w1;
  x->x_w2_init = x->x_w2;

  x->x_L1 = (t_float)1.0;
  x->x_L2 = x->x_L1;
  x->x_m1 = (t_float)1.0;
  x->x_m2 = x->x_m1;

  x->x_g = (t_float)9.81;

  x->x_recip2pi = (t_float)1.0 / (t_float)((t_float)2.0 * (t_float)M_PI);
  x->x_half_pi = (t_float)((t_float)M_PI / (t_float)2.0);
  x->x_conv = (t_float)0.0;
  x->x_f = (t_float)1.0;

  wavetable_init();

  x->x_recip2pi = (t_float)1.0 / (t_float)((t_float)2.0 * (t_float)M_PI);

  x->x_theta2_y_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_theta2_x_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_theta1_y_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_theta1_x_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void doublependulum_free(t_doublependulum *x) {
  wavetable_free();

  if (x->x_theta2_y_outlet) outlet_free(x->x_theta2_y_outlet);
  if (x->x_theta2_x_outlet) outlet_free(x->x_theta2_x_outlet);
  if (x->x_theta1_y_outlet) outlet_free(x->x_theta1_y_outlet);
  if (x->x_theta1_x_outlet) outlet_free(x->x_theta1_x_outlet);
}

static inline t_float fast_cos(t_doublependulum *x, t_float phase) {
  float *costab = cos_table;
  // normalize phase to [0, 1]
  phase = phase * x->x_recip2pi;  // 1 / (2 * M_PI)
  phase -= floorf(phase);
  phase *= WAVETABLE_SIZE;

  int idx = (int)phase & WAVETABLE_MASK;
  int idx_next = (idx + 1) & WAVETABLE_MASK;
  t_float frac = phase - floorf(phase);

  return costab[idx] * ((t_float)1.0 - frac) + costab[idx_next] * frac;
}

// static inline t_float fast_sin(t_doublependulum *x, t_float phase) {
//   // same normalization as fast_cos
//   phase = phase * x->x_recip2pi;
//   phase -= floorf(phase);
//   phase *= WAVETABLE_SIZE;
//
//   int idx = ((int)phase + QUARTER_TABLE) & WAVETABLE_MASK;
//   int idx_next = (idx + 1) & WAVETABLE_MASK;
//   t_float frac = phase - floorf(phase);
//
//   return cos_table[idx] * ((t_float)1.0 - frac) + cos_table[idx_next] * frac;
// }

static inline t_float fast_sin(t_doublependulum *x, t_float phase) {
  return fast_cos(x, phase - x->x_half_pi);
}

static t_int *doublependulum_perform(t_int *w) {
  t_doublependulum *x = (t_doublependulum *)(w[1]);
  t_sample *in_f = (t_sample *)(w[2]);
  t_sample *out_theta1_x = (t_sample *)(w[3]);
  t_sample *out_theta1_y = (t_sample *)(w[4]);
  t_sample *out_theta2_x = (t_sample *)(w[5]);
  t_sample *out_theta2_y = (t_sample *)(w[6]);
  int n = (int)(w[7]);

  t_float conv = x->x_conv; // 1 / sample_rate

  t_float theta1 = x->x_theta1;
  t_float theta2 = x->x_theta2;
  t_float w1 = x->x_w1;
  t_float w2 = x->x_w2;
  t_float m1 = x->x_m1;
  t_float m2 = x->x_m2;
  t_float l1 = x->x_L1;
  t_float l2 = x->x_L2;
  t_float g = x->x_g;

  // see notes for possible improvements
  while (n--) {
    t_sample f = *in_f++;
    t_float dt = f * conv;
    if (dt > (t_float)0.01) dt = (t_float)0.01;
    if (dt <= 0) dt = (t_float)1e-9;

    t_float d_theta1 = w1; // for clarity
    t_float d_theta2 = w2; // for clarity
    t_float delta = theta1 - theta2;
    // fast_sin and fast_cos use a lookup table
    t_float sin_delta = fast_sin(x, delta);
    t_float cos_delta = fast_cos(x, delta);
    // are there names for the parts of these functions?
    t_float d_w1_t1 = m2 * l1 * d_theta1*d_theta1 * sin_delta * cos_delta;
    t_float d_w1_t2 = m2 * g * fast_sin(x, theta2) * cos_delta;
    t_float d_w1_t3 = m2 * l2 * d_theta2*d_theta2 * sin_delta;
    t_float d_w1_t4 = (m1 + m2) * g * fast_sin(x, theta1);
    t_float d_w1_t5 = l1 * (m1 + m2) - m2 * l1 * cos_delta*cos_delta;

    t_float d_w1 = (d_w1_t1 + d_w1_t2 + d_w1_t3 - d_w1_t4) / d_w1_t5;

    t_float d_w2_t1 = -m2 * l2 * d_theta2*d_theta2 * sin_delta * cos_delta;
    t_float d_w2_t2 = (m1 + m2) * (g * fast_sin(x, theta1) * cos_delta
      - l1 * d_theta1*d_theta1 * sin_delta
      - g * fast_sin(x, theta2));
    t_float d_w2_t3 = l2 * (m1 + m2) - m2 * l2 * cos_delta*cos_delta;

    t_float d_w2 = (d_w2_t1 + d_w2_t2) / d_w2_t3;

    theta1 += d_theta1 * dt;
    theta2 += d_theta2 * dt;
    w1 += d_w1 * dt;
    w2 += d_w2 * dt;

    t_sample x1 = l1 * fast_sin(x, theta1);
    t_sample y1 = -l1 * fast_cos(x, theta1);
    t_sample x2 = x1 + l2 * fast_sin(x, theta2);
    t_sample y2 = y1 - l2 * fast_cos(x, theta2);

    *out_theta1_x++ = x1;
    *out_theta1_y++ = y1;
    *out_theta2_x++ = x2;
    *out_theta2_y++ = y2;
  }

  x->x_w1 = w1;
  x->x_w2 = w2;
  x->x_theta1 = theta1;
  x->x_theta2 = theta2;

  return (w+8);
}

static void doublependulum_dsp(t_doublependulum *x, t_signal **sp) {
  x->x_conv = (t_float)((t_float)1.0 / sp[0]->s_sr);
  dsp_add(doublependulum_perform, 7, x,
          sp[0]->s_vec,
          sp[1]->s_vec,
          sp[2]->s_vec,
          sp[3]->s_vec,
          sp[4]->s_vec,
          sp[0]->s_length);
}

static void reset(t_doublependulum *x) {
  x->x_theta1 = x->x_theta1_init;
  x->x_theta2 = x->x_theta2_init;
  x->x_w1 = x->x_w1_init;
  x->x_w2 = x->x_w2_init;
}

void doublependulum_tilde_setup(void) {
  doublependulum_class = class_new(gensym("doublependulum~"),
                                   (t_newmethod)doublependulum_new,
                                   (t_method)doublependulum_free,
                                   sizeof(t_doublependulum),
                                   CLASS_DEFAULT,
                                   0);
  class_addmethod(doublependulum_class, (t_method)doublependulum_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(doublependulum_class, (t_method)reset, gensym("reset"), 0);
  CLASS_MAINSIGNALIN(doublependulum_class, t_doublependulum, x_f);
}

