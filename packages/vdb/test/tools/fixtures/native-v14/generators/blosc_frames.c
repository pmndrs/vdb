/*
 * blosc_frames.c (K7.C1): runs the Blosc NanoVDB v14 links (`nanovdb_blosc`, LZ4 1.9.4) over a batch of
 * files, so `test/tools/blosc-write.test.ts` can compare our LZ4 Blosc writer with it byte for byte.
 *
 * Each line of the manifest (argv[1]) is one job:
 *   c <clevel> <shuffle> <typesize> <blocksize> <destsize> <in> <out>   compress; <out> gets the frame
 *                                                                       (empty when Blosc returns 0)
 *   d <in> <out>                                                        decompress a frame (ours)
 * Build (see ../README.md):
 *   clang -O1 -I <ovdb>/ext/blosc/openvdb/ext/blosc blosc_frames.c <nanovdb_blosc.o> -o blosc_frames
 * MIT, Copyright (c) 2026 Dennis Smolek and Poimandres contributors.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nanovdb_blosc.h"

static unsigned char* slurp(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char* buf = malloc(len > 0 ? (size_t)len : 1);
  if (len > 0 && fread(buf, 1, (size_t)len, f) != (size_t)len) { perror(path); exit(2); }
  fclose(f);
  *n = (size_t)len;
  return buf;
}

static void spit(const char* path, const void* data, size_t n) {
  FILE* f = fopen(path, "wb");
  if (!f) { perror(path); exit(2); }
  if (n && fwrite(data, 1, n, f) != n) { perror(path); exit(2); }
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc != 2) { fprintf(stderr, "usage: blosc_frames <manifest>\n"); return 2; }
  FILE* m = fopen(argv[1], "r");
  if (!m) { perror(argv[1]); return 2; }
  char line[4096];
  while (fgets(line, sizeof line, m)) {
    char op[4], in[1024], out[1024];
    int clevel, shuffle;
    size_t typesize, blocksize, destsize, n;
    if (sscanf(line, "%3s", op) != 1) continue;
    if (op[0] == 'c') {
      if (sscanf(line, "%3s %d %d %zu %zu %zu %1023s %1023s", op, &clevel, &shuffle, &typesize, &blocksize, &destsize, in, out) != 8) {
        fprintf(stderr, "bad line: %s", line);
        return 2;
      }
      unsigned char* src = slurp(in, &n);
      unsigned char* dst = malloc(destsize ? destsize : 1);
      int count = nanovdb_blosc_compress_ctx(clevel, shuffle, typesize, n, src, dst, destsize, NANOVDB_BLOSC_LZ4_COMPNAME, blocksize, 1);
      if (count < 0) { fprintf(stderr, "compress error %d for %s\n", count, in); return 1; }
      spit(out, dst, (size_t)count);
      free(src);
      free(dst);
    } else if (op[0] == 'd') {
      if (sscanf(line, "%3s %1023s %1023s", op, in, out) != 3) { fprintf(stderr, "bad line: %s", line); return 2; }
      unsigned char* src = slurp(in, &n);
      size_t nbytes = 0, cbytes = 0, bs = 0;
      nanovdb_blosc_cbuffer_sizes(src, &nbytes, &cbytes, &bs);
      unsigned char* dst = malloc(nbytes ? nbytes : 1);
      int count = nanovdb_blosc_decompress_ctx(src, dst, nbytes, 1);
      if (count < 0 || (size_t)count != nbytes) { fprintf(stderr, "decompress error %d for %s\n", count, in); return 1; }
      spit(out, dst, nbytes);
      free(src);
      free(dst);
    }
  }
  fclose(m);
  return 0;
}
