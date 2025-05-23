#include "m_pd.h"
#include <math.h>

/**
 * implements a double pendulum system
 * notes:
 * - double_pendulum.md
 * - double_pendulum_equations.md
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
    post("doublependulum~: freed cosine table");
    table_reference_count = 0; // just to be safe
  }
}

static void *doublependulum_new(t_symbol *s, int argc, t_atom *argv) {
  t_doublependulum *x = (t_doublependulum *)pd_new(doublependulum_class);

  t_float theta1, theta2;
  if (argc < 2) {
    // both pendulums at 90 degrees
    theta1 = (t_float)((t_float)M_PI / (t_float)2.0);
    theta2 = theta1;
  } else {
    theta1 = atom_getfloat(argv++);
    theta2 = atom_getfloat(argv++);
  }
  x->x_theta1 = theta1;
  x->x_theta2 = theta2;

  // pendulums are initially at rest
  x->x_w1 = (t_float)0.0;
  x->x_w2 = (t_float)0.0;

  // for the reset function
  x->x_theta1_init = x->x_theta1;
  x->x_theta2_init = x->x_theta2;
  x->x_w1_init = x->x_w1;
  x->x_w2_init = x->x_w2;

  // the constants can be set with messages
  x->x_L1 = (t_float)1.0;
  x->x_L2 = x->x_L1;
  x->x_m1 = (t_float)1.0;
  x->x_m2 = x->x_m1;

  // not configurable for now
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

static inline t_float fast_sin(t_doublependulum *x, t_float phase) {
  float *costab = cos_table;
  // offset phase by 90 degrees before normalization
  phase = (phase - x->x_half_pi) * x->x_recip2pi;
  phase -= floorf(phase);
  phase *= WAVETABLE_SIZE;

  int idx = (int)phase & WAVETABLE_MASK;
  int idx_next = (idx + 1) & WAVETABLE_MASK;
  t_float frac = phase - floorf(phase);

  return costab[idx] * (1.0f - frac) + costab[idx_next] * frac;
}

static inline void fast_sincos(t_doublependulum *x, t_float phase, t_float *sin_out, t_float *cos_out) {
  float *costab = cos_table;
  // normalize phase to [0, WAVETABLE_SIZE)
  phase = phase * x->x_recip2pi * WAVETABLE_SIZE;
  phase -= floorf(phase / WAVETABLE_SIZE) * WAVETABLE_SIZE;

  int cos_idx = (int)phase & WAVETABLE_MASK;
  int cos_idx_next = (cos_idx + 1) & WAVETABLE_MASK;
  t_float frac = phase - floorf(phase);
  // using the float type intentionally here. the cosine table uses float (not
  // double) values. 
  *cos_out = costab[cos_idx] * (1.0f - frac) + costab[cos_idx_next] * frac;

  int sin_offset = WAVETABLE_SIZE >> 2;  // WAVETABLE_SIZE / 4
  int sin_idx = (cos_idx + sin_offset) & WAVETABLE_MASK;
  int sin_idx_next = (sin_idx + 1) & WAVETABLE_MASK;
  *sin_out = costab[sin_idx] * (1.0f - frac) + costab[sin_idx_next] * frac;
}

static void compute_derivatives(t_doublependulum *x,
                                t_float theta1, t_float theta2, t_float w1, t_float w2,
                                t_float *d_w1, t_float *d_w2) {

  t_float m1 = x->x_m1;
  t_float m2 = x->x_m2;
  t_float l1 = x->x_L1;
  t_float l2 = x->x_L2;
  t_float g = x->x_g;

  t_float delta = theta1 - theta2;
  t_float sin_delta, cos_delta;
  fast_sincos(x, delta, &sin_delta, &cos_delta);
  t_float cos_delta_squared = cos_delta * cos_delta;
  t_float sin_theta1 = fast_sin(x, theta1);
  t_float sin_theta2 = fast_sin(x, theta2);
  t_float summed_mass = m1 + m2;
  t_float w1_squared = w1 * w1;
  t_float w2_squared = w2 * w2;
  t_float sin_x_cos_delta = sin_delta * cos_delta;

  /**
   * See double_pendulum_equations.md for the equations.
   *
   * The equations capture three main physical effects:
   *  - gravitational torque: terms with `g*sin(theta)`
   *  - centrifugal forces: terms with w^2 (or d_theta^2 below)
   *  - coupling effects: terms with sin(delta) and cos(delta) that modulate how
   *  the pendulums influence each other
   **/

  // 1: a centrifugal effect - the self-coupling term from pendulum 1's rotation
  // affecting its own acceleration through the interaction with pendulum 2.
  // when pendulum 1 rotates, it creates a centrifugal force that through the
  // coupling with pendulum 2 feeds back into its own motion
  t_float centforce_pen1 = m2 * l1 * w1_squared * sin_x_cos_delta;
  // 2: the gravitational torque from pendulum 2 acting on pendulum 1 - the
  // cos(delta) factor show how the effect depends on the relative angle between
  // the pendulums
  t_float gravity_pen2_pen1 = m2 * g * sin_theta2 * cos_delta;
  // 3: the centrifugal force from pendulum 2's rotation acting on pendulum 1 -
  // when pendulum 2 swings, it pulls the joint, creating a torque on pendulum
  // 1
  t_float centforce_from_pen2 = m2 * l2 * w2_squared * sin_delta;
  // 4: the gravitational torque on pendulum 1 from both masses:
  t_float gravity_pen1 = summed_mass * g * sin_theta1;
  // the effective moment of inertia of pendulum 1 (the ratio between the torque
  // applied and angular acceleration):
  t_float inertia_pen1 = l1 * summed_mass - m2 * l1 * cos_delta_squared;
  t_float inv_inertia_pen1 = (t_float)1.0 / inertia_pen1;

  *d_w1 = (centforce_pen1 + gravity_pen2_pen1 + centforce_from_pen2 - gravity_pen1) * inv_inertia_pen1;

  // 1: similar to pendulum 1, the self-coupling centrifugal effect for pendulum
  // 2
  t_float centforce_pen2 = -m2 * l2 * w2_squared * sin_x_cos_delta;
  // 2: the gravitational effect from pendulum 1's position on pendulum 2
  t_float gravity_pen1_pen2 = g * sin_theta1 * cos_delta;
  // 3: the centrifugal force from pendulum 1's rotation acting on pendulum 2
  t_float centforce_from_pen1 = l1 * w1_squared * sin_delta;
  // 4: direct gravitational torque on pendulum 2
  t_float gravity_pen2 = g * sin_theta2;
  // the effective moment of inertia of pendulum 2
  t_float inertia_pen2 = l2 * summed_mass - m2 * l2 * cos_delta_squared;
  t_float inv_inertia_pen2 = (t_float)1.0 / inertia_pen2;
  *d_w2 = (centforce_pen2 + summed_mass * (gravity_pen1_pen2 - centforce_from_pen1 - gravity_pen2)) * inv_inertia_pen2;
}

static void compute_positions(t_doublependulum *x,
                              t_float theta1, t_float theta2,
                              t_float *x1, t_float *y1, t_float *x2, t_float *y2) {
  t_float l1 = x->x_L1;
  t_float l2 = x->x_L2;

  *x1 = l1 * fast_sin(x, theta1);
  *y1 = -l1 * fast_cos(x, theta1);
  *x2 = *x1 + l2 * fast_sin(x, theta2);
  *y2 = *y1 - l2 * fast_cos(x, theta2);
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

  while (n--) {
    t_sample f = *in_f++;
    t_float dt = f * conv;
    if (dt > (t_float)0.01) dt = (t_float)0.01;
    if (dt <= 0) dt = (t_float)1e-9;

    t_float d_w1, d_w2;
    compute_derivatives(x, theta1, theta2, w1, w2, &d_w1, &d_w2);

    theta1 += w1 * dt;
    theta2 += w2 * dt;
    w1 += d_w1 * dt;
    w2 += d_w2 * dt;

    t_float x1, y1, x2, y2;
    compute_positions(x, theta1, theta2, &x1, &y1, &x2, &y2);

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

static void set_lengths(t_doublependulum *x, t_floatarg l1, t_floatarg l2) {
  l1 = (l1 > 0) ? l1 : (t_float)1.0;
  l2 = (l2 > 0) ? l2 : (t_float)1.0;
  x->x_L1 = l1;
  x->x_L2 = l2;
}

static void set_masses(t_doublependulum *x, t_floatarg m1, t_floatarg m2) {
  m1 = (m1 > 0) ? m1 : (t_float)1.0;
  m2 = (m2 > 0) ? m2 : (t_float)1.0;
  x->x_m1 = m1;
  x->x_m2 = m2;
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
                                   A_GIMME, 0);
  class_addmethod(doublependulum_class, (t_method)doublependulum_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(doublependulum_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(doublependulum_class, (t_method)set_lengths, gensym("lengths"),
                  A_FLOAT, A_FLOAT, 0);
  class_addmethod(doublependulum_class, (t_method)set_masses, gensym("masses"),
                  A_FLOAT, A_FLOAT, 0);
  CLASS_MAINSIGNALIN(doublependulum_class, t_doublependulum, x_f);
}

