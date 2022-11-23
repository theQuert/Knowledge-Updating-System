### srcIndex < srcSelectDimsize
- The input/output may be too long (PRIMERA: `input: 4096`, `output: 1024`).
- Before fixing the spaces problem, the input and output size only work with less then 512 tokens.
- The define special token `<KEEP>`, `<ADD>`, `<SUB>`.
- Too many spaces in contents (replace '.\n\n' with '.\c\c').

### Hardware problem (ECC error)
- Create new instance.

### CUDA Error
- Downgrade the CUDA version to meet Pytorch's requirements.

### Add special token to tokenizer/vocab
- Add the additional special token utilizing Huggingface's built-in function.
- Add our special token to tokenizer, then update the `vocab.json`. Then use the updated tokenizer to do truncation and encoding.

### Applied method and Next step
- Replace '.\n\n' with '.\c\c' to fix the srcIndex problem.
- Special tokens are not added successfully, the built-in function may not work. *Sol: Try to utilize the updated tokenizer in the main model*
- The #outupts is more then given #instances, still needed to debug. 
