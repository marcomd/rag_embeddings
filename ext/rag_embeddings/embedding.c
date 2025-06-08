#include <ruby.h>     // Ruby API
#include <stdint.h>   // For integer types like uint16_t
#include <stdlib.h>   // For memory allocation functions
#include <math.h>     // For math functions like sqrt
#include "embedding_config.h" // Import the configuration

typedef struct {
  float values[EMBEDDING_DIMENSION];
} embedding_t;

static const rb_data_type_t embedding_type = {
  "RagEmbeddings/Embedding",
  {0, 0, 0,},
  0, 0,
  RUBY_TYPED_FREE_IMMEDIATELY | RUBY_TYPED_EMBEDDABLE
};

static VALUE embedding_from_array(VALUE klass, VALUE rb_array) {
  Check_Type(rb_array, T_ARRAY);
  if (RARRAY_LEN(rb_array) != EMBEDDING_DIMENSION)
    rb_raise(rb_eArgError, "Wrong dimension, must be %d", EMBEDDING_DIMENSION);

  embedding_t *ptr;
  VALUE obj = TypedData_Make_Struct(klass, embedding_t, &embedding_type, ptr);
  for (int i = 0; i < EMBEDDING_DIMENSION; ++i)
    ptr->values[i] = (float)NUM2DBL(rb_ary_entry(rb_array, i));
  return obj;
}

static VALUE embedding_to_a(VALUE self) {
  embedding_t *ptr;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, ptr);
  VALUE arr = rb_ary_new2(EMBEDDING_DIMENSION);
  for (int i = 0; i < EMBEDDING_DIMENSION; ++i)
    rb_ary_push(arr, DBL2NUM(ptr->values[i]));
  return arr;
}

static VALUE embedding_cosine_similarity(VALUE self, VALUE other) {
  embedding_t *a, *b;
  TypedData_Get_Struct(self, embedding_t, &embedding_type, a);
  TypedData_Get_Struct(other, embedding_t, &embedding_type, b);

  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (int i = 0; i < EMBEDDING_DIMENSION; ++i) {
    dot += a->values[i] * b->values[i];
    norm_a += a->values[i] * a->values[i];
    norm_b += b->values[i] * b->values[i];
  }
  return DBL2NUM(dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-8));
}

void Init_embedding(void) {
  VALUE mRag = rb_define_module("RagEmbeddings");
  VALUE cEmbedding = rb_define_class_under(mRag, "Embedding", rb_cObject);
  rb_define_singleton_method(cEmbedding, "from_array", embedding_from_array, 1);
  rb_define_method(cEmbedding, "to_a", embedding_to_a, 0);
  rb_define_method(cEmbedding, "cosine_similarity", embedding_cosine_similarity, 1);
}