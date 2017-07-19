/*
 * This file is part of an experimental software implementation of the
 * Erickson-Monma-Veinott algorithm for solving the Steiner problem in graphs.
 * The algorithm runs in edge-linear time and the exponential complexity is
 * restricted to the number of terminal vertices.
 *
 * This software was developed as part of my master thesis work 
 * "Scalable Parameterised Algorithms for two Steiner Problems" at Aalto
 * University, Finland.
 *
 * The following graph generator [in particular, the fast-forward psuedo-random
 * number generator] is adapted from
 *    A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri,
 *    "Engineering motif search for large graphs",
 *    ALENEX15 Meeting on Algorithm Engineering and Experiments,
 *    5 January 2015, San Diego, CA.
 *
 * The source code is configured for a gcc build for Intel
 * microarchitectures. Other builds are possible but require manual
 * configuration of the 'Makefile'.
 * 
 * The source code is subject to the following license.
 *
 * Copyright (c) 2017 Suhas Thejaswi
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include<sys/utsname.h>
#include<string.h>
#include<stdarg.h>
#include<assert.h>
#include<math.h>

#include<omp.h>

typedef long int index_t;

#include"ffprng.h"

#define BUILD_PARALLEL

/******************************************************* Common subroutines. */

#define ERROR(...) error(__FILE__,__LINE__,__func__,__VA_ARGS__);

static void error(const char *fn, int line, const char *func, 
                  const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, 
            "ERROR [file = %s, line = %d]\n"
            "%s: ",
            fn,
            line,
            func);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();    
}

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}


#define MALLOC(id, x) malloc_wrapper(id, x)
#define FREE(id, x) free_wrapper(id, x)

index_t malloc_balance = 0;

void *malloc_wrapper(const char *id, size_t size)
{
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;
#ifdef MEM_INVENTORY
    fprintf(stdout, "alloc: %10s %7.3lf GiB\n", id, inGiB(size));
    fflush(stdout);
#endif
    return p;
}

void free_wrapper(const char *id, void *p)
{
    free(p);
    malloc_balance--;
#ifdef MEM_INVENTORY
    fprintf(stdout, "free: %10s\n", id);
    fflush(stdout);
#endif
}

void shellsort(index_t n, index_t *a)
{
    index_t h = 1;
    index_t i;
    for(i = n/3; h < i; h = 3*h+1)
        ;
    do {
        for(i = h; i < n; i++) {
            index_t v = a[i];
            index_t j = i;
            do {
                index_t t = a[j-h];
                if(t <= v)
                    break;
                a[j] = t;
                j -= h;
            } while(j >= h);
            a[j] = v;
        }
        h /= 3;
    } while(h > 0);
}

/************************************************ Rudimentary graph builder. */

typedef struct 
{
    index_t num_vertices;
    index_t num_edges;
    index_t num_terminals;
    index_t edge_capacity;
    index_t num_coordinates;
    index_t *edges;
    index_t *terminals;
    index_t cost;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC("a", sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE("was", was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);

    graph_t *g = (graph_t *) MALLOC("g", sizeof(graph_t));
    g->num_vertices  = n;
    g->num_edges     = 0;
    g->num_terminals = 0;
    g->edge_capacity = 1000;
    g->edges = enlarge(2*g->edge_capacity, 0, (void *) 0);
    g->terminals = NULL;
    g->cost = -1;
    return g;
}

void graph_free(graph_t *g)
{
    if(g->terminals != NULL)
        FREE("g->terminals", g->terminals);
    FREE("g->edges", g->edges);
    FREE("g", g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v)
{
    assert(u >= 0 && 
           v >= 0 && 
           u < g->num_vertices &&
           v < g->num_vertices);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(4*g->edge_capacity, 2*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 2*g->num_edges;
    g->num_edges++;
    e[0] = u;
    e[1] = v;
    shellsort(2, e);
}

index_t *graph_edgebuf(graph_t *g, index_t cap)
{
    g->edges = enlarge(2*g->edge_capacity+2*cap, 2*g->edge_capacity, g->edges);
    index_t *e = g->edges + 2*g->num_edges;
    g->edge_capacity += cap;
    g->num_edges += cap;
    return e;
}

void graph_set_target(graph_t *g, index_t k, index_t *kk) 
{
    g->num_terminals = k;
    g->terminals = (index_t *) MALLOC("g->target", sizeof(index_t)*k);
    for(index_t j = 0; j < k; j++)
        g->terminals[j] = kk[j];
}

void graph_out_dimacs(FILE *out, graph_t *g, index_t w, index_t seed)
{
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t k = g->num_terminals;
    index_t *e = g->edges;
    index_t *kk = g->terminals;
    fprintf(out, "section graph\n");
    fprintf(out, "nodes %ld\nedges %ld\n", (long) n, (long) m);
    srand(seed);
    for(index_t i = 0; i < m; i++)
    {
        fprintf(out, "e %ld %ld %ld\n", 
                     (long) e[2*i+0]+1, 
                     (long) e[2*i+1]+1, 
                     ((long)rand()%w)+1);
    }
    fprintf(out, "end\n\n");

    //for(index_t i = 0; i < n; i++)
    //    fprintf(out, "n %ld %ld\n", (long) i+1, (long) c[i]+1);
    fprintf(out, "section terminals\n");
    fprintf(out, "terminals %ld\n", k);
    for(index_t i = 0; i < k; i++)
        fprintf(out, "t %ld\n", kk[i]+1);
    fprintf(out, "end\n\n");

    //fprintf(out, "cost %ld\n\n", g->cost);
    fprintf(out, "eof\n");
}

#define BIN_MAGIC 0x1234567890ABCDEFUL

#define CMD_TEST_UNIQUE 1

void randshuffle_seq(index_t n, index_t *p, ffprng_t gen)
{
    for(index_t i = 0; i < n-1; i++) {
        ffprng_scalar_t rnd;
        FFPRNG_RAND(rnd, gen);     
        index_t x = i+(rnd%(n-i));
        index_t t = p[x];
        p[x] = p[i];
        p[i] = t;
    }
}

void randperm(index_t n, index_t *p, index_t seed)
{
#ifdef BUILD_PARALLEL
    index_t nt = 64;
#else
    index_t nt = 1;
#endif
    index_t block_size = n/nt;
    index_t f[128][128];
    assert(nt < 128);

    ffprng_t base;
    FFPRNG_INIT(base, seed);    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        for(index_t j = 0; j < nt; j++)
            f[t][j] = 0;        
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        ffprng_t gen;
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            f[t][bin]++;
        }
    }    

    for(index_t bin = 0; bin < nt; bin++) {
        for(index_t t = 1; t < nt; t++) {
            f[0][bin] += f[t][bin];
        }
    }
    index_t run = 0;
    for(index_t j = 1; j <= nt; j++) {
        index_t fp = f[0][j-1];
        f[0][j-1] = run;
        run += fp;
    }
    f[0][nt] = run;

    FFPRNG_INIT(base, seed);    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = 0;
        index_t stop = n-1;
        index_t pos = f[0][t];
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            if(bin == t)
                p[pos++] = i;
        }
        assert(pos == f[0][t+1]);
    }

    FFPRNG_INIT(base, (seed^0x9078563412EFDCABL));    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t fwd, gen;
        index_t start = f[0][t];
        index_t stop = f[0][t+1]-1;
        index_t u;
        FFPRNG_FWD(fwd, (1234567890123456L*t), base);
        FFPRNG_RAND(u, fwd);
        FFPRNG_INIT(gen, u);
        randshuffle_seq(stop-start+1, p + start, gen);
    }
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC("t", sizeof(index_t)*n);
    return t;
}

index_t *alloc_randperm(index_t n, index_t seed)
{
    index_t *p = alloc_idxtab(n);
    randperm(n, p, seed);
    return p;
}

index_t *graph_degree_dist(graph_t *g)
{
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t *deg = alloc_idxtab(n);
    for(index_t i = 0; i < n; i++)
        deg[i] = 0;
    for(index_t j = 0; j < m; j++) {
        deg[g->edges[2*j]]++;
        deg[g->edges[2*j+1]]++;
    }
    return deg;
}

void print_stat(FILE *out, graph_t *g) 
{
    index_t n = g->num_vertices;
    index_t *deg = graph_degree_dist(g);
    for(index_t i = 0; i < n; i++) {
        fprintf(out, "deg[%ld] = %ld\n", i, deg[i]);
    }
    FREE("deg", deg);
}

/***************** A quick-and-dirty generator for a power law distribution. */

double beta(index_t n, index_t k)
{
    double min, mid, max;
    index_t niter = 200;
    min = 1.0;
    max = exp((log(n)/(double) (k-1)));
    for(index_t i = 0; i < niter; i++) {
        mid = (min+max)/2.0;
        double nn = (1-pow(mid,k))/(1-mid);
        if(nn < n)
            min = mid;
        if(nn > n)
            max = mid;
    }
    return mid;
}

double degnormalizer(index_t n, index_t d, index_t *freq, index_t k, double alpha)
{
    double fpowasum = 0.0;
    for(index_t i = 0; i < k; i++)
        fpowasum += pow(freq[i],alpha+1);
    return log(d*n)-log(fpowasum);
}

void mkpowlawdist(index_t *deg, index_t *freq,
                  index_t n, index_t d, double alpha, index_t k)
{

    double b = beta(n, k);
    index_t fsum = 0;
    for(index_t j = 0; j < k; j++) {
        freq[j] = round((1-pow(b,j+1))/(1-b)-fsum);
        fsum += freq[j];
    }
    double dn = degnormalizer(n, d, freq, k, alpha);

    double t = 0.0; 
    index_t dfsum = 0; 
    for(index_t j = 0; j < k; j++) {
        t += exp(dn)*pow(freq[j],alpha+1);
        double tt = t-dfsum;
        deg[j] = round(tt/freq[j]);
        dfsum += deg[j]*freq[j];
    }


    if(dfsum % 2 == 1) {
        index_t i = k-1;
        for(; i >= 0; i--) { 
            if(deg[i] % 2 == 1 &&
               freq[i] % 2 == 1) {
                freq[i]++;
                dfsum += deg[i];
                break;
            }
        }
        assert(i >= 0);
    }

    fprintf(stderr, 
            "powlaw: n = %ld, d = %ld, alpha = %lf, w = %ld, beta = %lf, norm = %lf\n",
            n, d, alpha, k, b, dn);
}

/************************************************** Test graph generator(s). */

/* The configuration model. 
 * Warning: no check for repeated edges and loops. */

graph_t *graph_config_rand(index_t n, index_t ndd, index_t *degree, index_t *freq, index_t seed) 
{

    index_t num_incidences = 0;
    for(index_t j = 0; j < ndd; j++)
        num_incidences += degree[j]*freq[j];
    assert(num_incidences % 2 == 0);

    index_t *vertex_id = alloc_idxtab(num_incidences);
    index_t pos = 0;
    index_t vno = 0;
    for(index_t j = 0; j < ndd; j++) {
        for(index_t k = 0; k < freq[j]; k++) {
            for(index_t l = 0; l < degree[j]; l++)
                vertex_id[pos++] = vno;
            vno++;
        }
    }
    index_t *vertex_shuffle = alloc_randperm(n, seed);
    index_t *incidence_shuffle = alloc_randperm(num_incidences, seed);

    graph_t *g = graph_alloc(n);
    index_t *e = graph_edgebuf(g, num_incidences/2);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < num_incidences/2; i++) {
        index_t u = vertex_shuffle[vertex_id[incidence_shuffle[2*i]]];
        index_t v = vertex_shuffle[vertex_id[incidence_shuffle[2*i+1]]];
        if(u > v) {
            index_t t = u;
            u = v;
            v = t;
        }
        e[2*i] = u;
        e[2*i+1] = v;
    }

    FREE("vertex_id", vertex_id);
    FREE("vertex_shuffle", vertex_shuffle);
    FREE("incidence_shuffle", incidence_shuffle);
    return g;
}

graph_t *graph_reg_rand(index_t n, index_t d, index_t seed)
{
    index_t deg[128];
    index_t freq[128];
    deg[0] = d;
    freq[0] = n;
    return graph_config_rand(n, 1, deg, freq, seed);
}

graph_t *graph_powlaw_rand(index_t n, index_t d, double alpha, index_t k, index_t seed)
{
    index_t *deg = (index_t *) MALLOC("deg", k*sizeof(index_t));
    index_t *freq = (index_t *) MALLOC("freq", k*sizeof(index_t));
    mkpowlawdist(deg, freq, n, d, alpha, k);
    graph_t *g = graph_config_rand(n, k, deg, freq, seed);
    FREE("deg", deg);
    FREE("freq", freq);
    return g;
}

graph_t *graph_set_rand_target(graph_t *g, index_t k, index_t seed)
{
    index_t n = g->num_vertices;
    index_t *p = alloc_randperm(n, seed); 

    for(index_t i = 1; i < k; i++)
        graph_add_edge(g, p[i-1], p[i]);

    graph_set_target(g, k, p);

    FREE("p", p);
    return g;
}

graph_t *graph_clique_rand(index_t n, index_t d, index_t seed)
{
    graph_t* g = graph_alloc(n);
    index_t* v = alloc_randperm(n, seed);
    index_t u = sqrt((n * d) / 2);

    for(index_t i = 0; i < u; ++i)
        for(index_t j = i + 1; j < u; ++j)
        graph_add_edge(g, v[i], v[j]);
    
    FREE("v", v);
    return g;
}

/****************************************************** Program entry point. */

int main(int argc, char **argv)
{
    if(argc < 2) {
        fprintf(stdout, 
                "usage: %s <type> <arguments>\n"
                "available types (all parameters positive integers unless indicated otherwise):\n"
                "\n"
                "  small   <n> <idx>                    (with 1 <= idx <= 2^n-1)\n"
                "  regular <n> <d> <k> <ew> <seed>           (with 1 <= k <= n and n*d even)\n"
                "  powlaw  <n> <d> <al> <w> <k> <ew> <seed>  (with al < 0.0, 2 <= w <= n, 1 <= k <= n)\n"
                "  clique  <n> <d> <k> <ew> <seed>           (with 1 <= k <= n)\n"
                "\n"
                ,
                argv[0]);
        return 0;
    }

    fprintf(stderr, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stderr, " %s", argv[f]);
    fprintf(stderr, "\n");    

    index_t seed = 123456789;
    index_t ew = 0;
    graph_t *g = (graph_t *) 0;

    char *type = argv[1];
    if(!strcmp("regular", type)) { 
        assert(argc-2 >= 5);
        index_t n    = atol(argv[2+0]);
        index_t d    = atol(argv[2+1]);
        index_t k    = atol(argv[2+2]);
        ew   = atol(argv[2+3]);
        seed = atol(argv[2+4]);
        assert(n >= 1 && d >= 0 && n*d % 2 == 0 && k >= 1 
                && k <= n && seed >= 1 && ew >= 1);
        g = graph_reg_rand(n, d, seed);
        graph_set_rand_target(g, k, seed);
    }
    if(!strcmp("powlaw", type)) { 
        assert(argc-2 >= 7); 
        index_t n    = atol(argv[2+0]);
        index_t d    = atol(argv[2+1]);
        double al    = atof(argv[2+2]);
        index_t w    = atol(argv[2+3]);
        index_t k    = atol(argv[2+4]);
        ew   = atol(argv[2+5]);
        seed = atol(argv[2+6]);
        assert(n >= 1 && d >= 0 && al < 0.0 && w >= 2 && w <= n 
                && k >= 1 && k <= n && seed >= 1 && ew >= 1); 
        g = graph_powlaw_rand(n, d, al, w, seed);
        graph_set_rand_target(g, k, seed);
    }
    if(!strcmp("clique", type)) {
        assert(argc-2 >= 5); 
        index_t n    = atol(argv[2+0]);
        index_t d    = atol(argv[2+1]);
        index_t k    = atol(argv[2+2]);
		ew   = atol(argv[2+3]);
        seed = atol(argv[2+4]);
        g = graph_clique_rand(n, d, seed);
        graph_set_rand_target(g, k, seed);
    }
    if(g == (graph_t *) 0) {
        fprintf(stderr, "unknown type: %s\n", type);
        return 1;
    }

    graph_out_dimacs(stdout, g, ew, seed);

    fprintf(stderr, "gen-unique [%s]: n = %ld, m = %ld, k = %ld, cost = %ld, seed = %ld\n", 
            type,
            g->num_vertices,
            g->num_edges,
            g->num_terminals,
            g->cost,
            seed);
    graph_free(g);

    return 0;
}
