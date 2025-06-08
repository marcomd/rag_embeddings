#include <ruby.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
  uint16_t dim;
  float values[];
} embedding_t;

static const rb_data_type_t embedding_type = {
  "RagEmbeddings/Embedding",
  { 0, 0, 0 },
  0, 0,
  RUBY_TYPED_FREE_IMMEDIATELY | RUBY_TYPED_EMBEDDABLE
};

static VALUE embedding_alloc(VALUE klass) {
  uint16_t dim = 768;
  embedding_t *ptr;
  VALUE obj = TypedData_Make_Struct(klass, embedding_t, &embedding_type, ptr);
  ptr->dim = dim;
  for (int i = 0; i < dim; ++i) ptr->values[i] = 0.0f;
  return obj;
}

static VALUE embedding_set(VALUE self, VALUE rb_array) {
  embedding_t *ptr;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);
  Check_Type(rb_array, T_ARRAY);
  if (RARRAY_LEN(rb_array) != ptr->dim)
    rb_raise(rb_eArgError, "Wrong dimension");
  for (int i = 0; i < ptr->dim; ++i) {
    ptr->values[i] = (float)NUM2DBL(rb_ary_entry(rb_array, i));
  }
  return Qtrue;
}

static VALUE embedding_cosine_similarity(VALUE self, VALUE other) {
  embedding_t *a, *b;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, a);
  TypedData_Get_Struct(other, embedding_t, &embedding_type, b);
  if (a->dim != b->dim)
    rb_raise(rb_eArgError, "Dimension mismatch");
  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (int i = 0; i < a->dim; ++i) {
    dot += a->values[i] * b->values[i];
    norm_a += a->values[i] * a->values[i];
    norm_b += b->values[i] * b->values[i];
  }
  return DBL2NUM(dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-8));
}

void Init_embedding(void) {
  VALUE mRag = rb_define_module("RagEmbeddings");
  VALUE cEmbedding = rb_define_class_under(mRag, "Embedding", rb_cObject);
  rb_define_alloc_func(cEmbedding, embedding_alloc);
  rb_define_method(cEmbedding, "set", embedding_set, 1);
  rb_define_method(cEmbedding, "cosine_similarity", embedding_cosine_similarity, 1);
}