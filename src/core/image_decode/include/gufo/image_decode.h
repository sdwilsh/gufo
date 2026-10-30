#ifndef GUFO_IMAGE_DECODE_H_
#define GUFO_IMAGE_DECODE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors GufoImageStatus in src/lib.rs. */
typedef enum GufoImageStatus {
  GUFO_IMAGE_OK = 0,
  GUFO_IMAGE_INVALID = 1,
  GUFO_IMAGE_TOO_LARGE = 2,
  GUFO_IMAGE_INTERNAL = 3,
} GufoImageStatus;

/* Packed RGB8 owned by Rust; release with gufo_image_free. */
typedef struct GufoDecodedImage {
  uint32_t width;
  uint32_t height;
  uint8_t* pixels;
  size_t len;
  size_t capacity;
} GufoDecodedImage;

GufoImageStatus gufo_image_decode_png(const uint8_t* bytes, size_t len,
                                      uint64_t max_pixels,
                                      GufoDecodedImage* out);
GufoImageStatus gufo_image_decode_webp(const uint8_t* bytes, size_t len,
                                       uint64_t max_pixels,
                                       GufoDecodedImage* out);
void gufo_image_free(GufoDecodedImage* image);

#ifdef __cplusplus
}
#endif

#endif
