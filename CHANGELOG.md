# CHANGELOG

## v0.2.0 09/06/2025 - Added the model selection feature and many improvements on the c extension

- Model selection e.g. `RagEmbeddings.embed(text, model: 'qwen3:0.6b')`
- Added rb_undef_alloc_func(cEmbedding) in Init_embedding() function to eliminate warning. This tells Ruby not to use default allocator since we are managing memory manually.
- Performance Optimizations
    - Better Ruby array handling: Use RARRAY_CONST_PTR and rb_ary_store instead of rb_ary_entry and rb_ary_push for better performance
    - Cache-friendly cosine similarity: A single loop instead of separate loops to be kinder to the processor cache
    - Improved numerical precision: Use double for intermediate calculations to reduce accumulation errors
    - Type switching for indices: Use uint16_t instead of int for consistency
- Robustness Improvements
    - Validation checks: Check for empty arrays, oversized arrays, and non-numeric types
    - Error Handling: Clean up memory before raising exceptions
    - Edge case handling: Check for zero arrays and clamp similarity results
    - Null pointer check: Check for null in embedding_free
- New Features
    - magnitude(): Calculate the L2 norm of the array
    - normalize!(): Normalize the array in-place to optimize similarity calculations

## v0.1.0 08/06/2025 - Project started

- Generate an embedding from text
- Create a C embedding object
- Compute similarity between two texts
- Store and search embeddings in a database