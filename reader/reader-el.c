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
#include<string.h>
#include<stdarg.h>
#include<assert.h>
#include<omp.h>
#include<sys/utsname.h>
#include<math.h>
#include<ctype.h>

/************************************************************* Configuration. */
#ifdef DEFAULT
#define BIN_HEAP
#define TRACK_OPTIMAL
#define BUILD_PARALLEL
#endif

#ifdef TRACK_RESOURCES
#define TRACK_MEMORY
#define TRACK_BANDWIDTH
#endif

#define MAX_K        32 
#define MAX_THREADS 128

typedef long int index_t; // default to 64-bit indexing

/********************************************************** Global constants. */

#define MAX_DISTANCE ((index_t)0x7FFFFFFFFFFFFFFF)
#define MATH_INF ((index_t)0x7FFFFFFFFFFFFFFF)
#define UNDEFINED -1

/************************************************************* Common macros. */

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// Linked list navigation macros. 

#define pnlinknext(to,el) { (el)->next = (to)->next; (el)->prev = (to); (to)->next->prev = (el); (to)->next = (el); }
#define pnlinkprev(to,el) { (el)->prev = (to)->prev; (el)->next = (to); (to)->prev->next = (el); (to)->prev = (el); }
#define pnunlink(el) { (el)->next->prev = (el)->prev; (el)->prev->next = (el)->next; }
#define pnrelink(el) { (el)->next->prev = (el); (el)->prev->next = (el); }


/*********************************************************** Error reporting. */

#define ERROR(...) error(__FILE__,__LINE__,__func__,__VA_ARGS__);

static void error(const char *fn, int line, const char *func, 
                  const char *format, ...) 
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, 
            "ERROR [file = %s, line = %d] "
            "%s: ",
            fn,
            line,
            func);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();    
}

/********************************************************* Get the host name. */

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

/********************************************************* Available threads. */

index_t num_threads(void)
{
#ifdef BUILD_PARALLEL
    return omp_get_max_threads();
#else
    return 1;
#endif
}

index_t thread_id(void)
{
#ifdef BUILD_PARALLEL
    return omp_get_thread_num();
#else
    return 1;
#endif
}

/******************************************************* String manipulation. */

char* strlower( char *s)
{
   char* t;

   for(t = s; *s != '\0'; s++)
      *s = (char)tolower(*s);

   return(t);
}

/************************************************************** Combinations. */

index_t choose(index_t n, index_t r)  
{
    if(r > n / 2) r = n - r; 
    long long ans = 1;
    index_t i;

    for(i = 1; i <= r; i++) {
        ans *= n - r + i;
        ans /= i;
    }   

    return (index_t)(ans);
}

/********************************************** Memory allocation & tracking. */
/* 
 * The following memory allocation and tracking subroutines are adapted from
 *    A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri,
 *    "Engineering motif search for large graphs",
 *    ALENEX15 Meeting on Algorithm Engineering and Experiments,
 *    5 January 2015, San Diego, CA.
 * 
 * source: https://github.com/pkaski/motif-search
 *
 */

#ifdef TRACK_MEMORY
#define MALLOC(x) malloc_wrapper(x)
#define CALLOC(x, y) calloc_wrapper((x), (y))
#define FREE(x) free_wrapper(x)

#else

#define MALLOC(x) malloc((x))
#define CALLOC(x, y) calloc((x), (y))
#define FREE(x) free((x))

#endif

#ifdef TRACK_MEMORY
index_t malloc_balance = 0;

struct malloc_track_struct
{
    void *p;
    size_t size;
    struct malloc_track_struct *prev;
    struct malloc_track_struct *next;
};

typedef struct malloc_track_struct malloc_track_t;

malloc_track_t malloc_track_root;
size_t malloc_total = 0;

#define MEMTRACK_STACK_CAPACITY 512
size_t memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t memtrack_stack_top = -1;

void *malloc_wrapper(size_t size)
{
    if(malloc_balance == 0) {
        malloc_track_root.prev = &malloc_track_root;
        malloc_track_root.next = &malloc_track_root;
    }
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;

    malloc_track_t *t = (malloc_track_t *) malloc(sizeof(malloc_track_t));
    t->p = p;
    t->size = size;
    pnlinkprev(&malloc_track_root, t);
    malloc_total += size;
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < malloc_total)
            memtrack_stack[i] = malloc_total;
    return p;
}

void *calloc_wrapper(size_t n, size_t size)
{
    if(malloc_balance == 0) {
        malloc_track_root.prev = &malloc_track_root;
        malloc_track_root.next = &malloc_track_root;
    }
    void *p = calloc(n, size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;

    malloc_track_t *t = (malloc_track_t *) malloc(sizeof(malloc_track_t));
    t->p = p;
    t->size = (n*size);
    pnlinkprev(&malloc_track_root, t);
    malloc_total += (n*size);
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < malloc_total)
            memtrack_stack[i] = malloc_total;
    return p;
}

void free_wrapper(void *p)
{
    malloc_track_t *t = malloc_track_root.next;
    for(;
        t != &malloc_track_root;
        t = t->next) {
        if(t->p == p)
            break;
    }
    if(t == &malloc_track_root)
        ERROR("FREE issued on a non-tracked pointer %p", p);
    malloc_total -= t->size;
    pnunlink(t);
    free(t);

    free(p);
    malloc_balance--;
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC(sizeof(index_t)*n);
    return t;
}

void push_memtrack(void)
{
    assert(memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    memtrack_stack[++memtrack_stack_top] = malloc_total;
}

size_t pop_memtrack(void)
{
    assert(memtrack_stack_top >= 0);
    return memtrack_stack[memtrack_stack_top--];
}

size_t current_mem(void)
{
    return malloc_total;
}

double inGiB(size_t s)
{
    return (double) s / (1 << 30);
}

void print_current_mem(void)
{
    fprintf(stdout, "{curr: %.2lfGiB}", inGiB(current_mem()));
    fflush(stdout);
}

void print_pop_memtrack(void)
{
    fprintf(stdout, "{peak: %.2lfGiB}", inGiB(pop_memtrack()));
    fflush(stdout);
}

void inc_malloc_total(size_t size)
{
    malloc_total += size;
}

void dec_malloc_total(size_t size)
{
    malloc_total -= size;
}
#endif

/******************************************************** Timing subroutines. */

#define TIME_STACK_CAPACITY 256
double start_stack[TIME_STACK_CAPACITY];
index_t start_stack_top = -1;

void push_time(void)
{
    assert(start_stack_top + 1 < TIME_STACK_CAPACITY);
    start_stack[++start_stack_top] = omp_get_wtime();
}

double pop_time(void)
{
    double wstop = omp_get_wtime();
    assert(start_stack_top >= 0);
    double wstart = start_stack[start_stack_top--];
    return (double) (1000.0*(wstop-wstart));
}

/*************************************************** Graph build subroutines. */

#define GRAPH_SEC_COMMENT         0x01
#define GRAPH_SEC_GRAPH           0x02
#define GRAPH_SEC_TERMINALS       0x04
#define GRAPH_SEC_COORDINATES     0x08
#define GRAPH_EDGES_ALLOC         0x10
#define GRAPH_TERMINALS_ALLOC     0x20
#define GRAPH_COORDINATES_ALLOC   0x40
#define GRAPH_STEINER_COST        0x80

typedef struct graph
{
    index_t root;
    index_t n;
    index_t m;
    index_t k;
    index_t num_edges;
    index_t num_terminals;
    index_t num_coordinates;
    index_t *edges;
    index_t *terminals;
    index_t *coordinates;
    index_t flags;
    index_t cost;
    index_t edge_capacity;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC(sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) { 
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE(was);
    }    
    return a;
}

graph_t *graph_alloc()
{
    graph_t *g = (graph_t *) MALLOC(sizeof(graph_t));
    g->root  = 0;
    g->flags = 0x00;
    g->cost  = -1;
    g->n = 0; 
    g->m = 0; 
    g->k = 0;
    g->edge_capacity    = 100;
    g->num_edges        = 0;
    g->num_terminals    = 0;
    g->num_coordinates  = 0;
    g->edges            = enlarge(3*g->edge_capacity, 0, (void *) 0);
    g->terminals        = NULL;
    g->coordinates      = NULL;
    
    return g;
}

void graph_free(graph_t *g)
{
    if(g->edges != NULL)
        FREE(g->edges);
    if(g->terminals != NULL)
        FREE(g->terminals);
    if(g->coordinates != NULL)
        FREE(g->coordinates);
    FREE(g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v, index_t w)
{
    assert(u >= 0 && v >= 0 && u < g->n && v < g->n);

    if(g->num_edges == g->edge_capacity)
    {
        g->edges = enlarge(6*g->edge_capacity, 3*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 3*g->num_edges;
    g->num_edges++;
    e[0] = u;
    e[1] = v;
    e[2] = w;
}

void graph_add_terminal(graph_t *g, index_t u)
{
    if(g->terminals == NULL)
        ERROR("section terminals not initialised");

    assert(u >= 0 && u < g->n);
    index_t *t = g->terminals + g->num_terminals;
    g->num_terminals++;

    assert(g->num_terminals <= g->k);
    t[0] = u;
}

void graph_add_coordinate(graph_t *g, index_t u, index_t x, index_t y)
{
    if(g->coordinates == NULL)
        ERROR("section coordinates not initialised");

    assert(u >= 0 && u < g->n);

    index_t *coord = g->coordinates + 3*g->num_coordinates;
    g->num_coordinates++;

    assert(g->num_coordinates <= g->n);

    coord[0] = u;
    coord[1] = x;
    coord[2] = y;
}

#define MAX_LINE_SIZE 1024
#define MAX_SECTION_SIZE 256

graph_t * graph_load(FILE *in)
{
    push_time();
#ifdef TRACK_MEMORY
    push_memtrack();
#endif

    char buf[MAX_LINE_SIZE];
    char line[MAX_LINE_SIZE];
    index_t n = 0;
    index_t m = 0;
    index_t k = 0;
    index_t u, v, w;
    index_t cost = 0;
    char section[MAX_SECTION_SIZE];

    index_t in_section = 0;

    graph_t *g = graph_alloc();

    while(fgets(line, MAX_LINE_SIZE, in) != NULL)
    {
        strcpy(buf, line);
        char *c = strtok(buf, " ");
        
        if(!strcmp(c, "section"))
        {
            if(in_section == 1) { ERROR("nested sections");}
            in_section = 1;

            if(sscanf(line, "section %s", section) != 1)
                ERROR("invalid section line");

            if(!strcmp(section, "comment"))
                g->flags |= GRAPH_SEC_COMMENT;
            else if(!strcmp(section, "graph"))
                g->flags |= GRAPH_SEC_GRAPH;
            else if(!strcmp(section, "terminals"))
                g->flags |= GRAPH_SEC_TERMINALS;
            else if (!strcmp(section, "coordinates"))
            {
                continue; //ignore
                //g->flags |= GRAPH_SEC_COORDINATES;
                //g->coordinates = (index_t *) MALLOC(3*n*sizeof(index_t));
            }
            else
                ERROR("invalid section");
        }
        else if(!strcmp(c, "end\n"))
        {
            if(in_section == 0) { ERROR("no section to end");}
            in_section = 0;
        }
        else if(!strcmp(c, "nodes"))
        {
            if(sscanf(line, "nodes %ld", &n) != 1)
                ERROR("invalid nodes line");
            g->n = n;
        }
        else if(!strcmp(c, "edges"))
        {
            if(sscanf(line, "edges %ld", &m) != 1)
                ERROR("invalid edges line");
            g->m = m;
            //g->edges = (index_t *) MALLOC(3*m*sizeof(index_t));
            g->flags |= GRAPH_EDGES_ALLOC;
        }
        else if(!strcmp(c, "terminals"))
        {
            if(sscanf(line, "terminals %ld", &k) != 1)
                ERROR("invalid terminals line");
            g->k = k;
            g->terminals = (index_t *) MALLOC(k*sizeof(index_t));
            g->flags |= GRAPH_TERMINALS_ALLOC;
        }
        else if(!strcmp(c, "coordinates"))
        {
            if(g->coordinates != NULL)
            {
                g->coordinates = (index_t *) MALLOC(3*n*sizeof(index_t));
                g->flags |= GRAPH_COORDINATES_ALLOC;
            }
            else
                ERROR("duplicate section coordinates");
        }
        else if(!strcmp(c, "e"))
        {
            if(sscanf(line, "e %ld %ld %ld", &u, &v, &w) != 3)
                ERROR("invalid edge line %s", line);
            graph_add_edge(g, u-1, v-1, w);
        }
        else if(!strcmp(c, "t"))
        {
            if(sscanf(line, "t %ld", &u) != 1)
                ERROR("invalid terminal line %s", line);
            graph_add_terminal(g, u-1);
        }
        else if(!strcmp(c, "dd"))
        {
            continue; // ignore coordinates
            //index_t x, y;
            //if(sscanf(line, "dd %ld %ld %ld", &u, &x, &y) != 3)
            //    ERROR("invalid coordinate line %s", line);
            //graph_add_coordinate(g, u-1, x, y);
        }
        else if(!strcmp(c, "cost"))
        {
            if(sscanf(line, "cost %ld", &cost) != 1)
                ERROR("invalid cost line %s", line);
            g->flags |= GRAPH_STEINER_COST;
            g->cost = cost;
        }
        else if(!strcmp(c, "eof"))
        {
            continue;
        }
        else
        {
            continue;
            //ERROR("invalid line %s", line);
        }

    }

    assert(g->n != 0);
    assert(g->m == g->num_edges && g->m != 0);
    assert(g->k == g->num_terminals && g->k != 0);
    assert((g->flags & GRAPH_SEC_GRAPH) && (g->flags & GRAPH_SEC_TERMINALS));

    double time = pop_time();
    fprintf(stdout, "input: n = %ld, m = %ld, k = %ld, cost = %ld [%.2lf ms] ",
                    g->n, g->m, g->k, g->cost, time);
#ifdef TRACK_MEMORY
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
#endif
    fprintf(stdout, "\n");
    fprintf(stdout, "terminals:");
    for(index_t i=0; i<g->k; i++) 
        fprintf(stdout, " %ld", g->terminals[i]+1);
    fprintf(stdout, "\n");
    fflush(stdout);

    return g;
}


/***************************************************** (Parallel) prefix sum. */
/* 
 * The following (parallel) prefixsum subroutine is adapted from
 *    A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri,
 *    "Engineering motif search for large graphs",
 *    ALENEX15 Meeting on Algorithm Engineering and Experiments,
 *    5 January 2015, San Diego, CA.
 * 
 * source: https://github.com/pkaski/motif-search
 *
 */

index_t prefixsum(index_t n, index_t *a, index_t k)
{

#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n; 
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = (stop-start+1)*k;
        for(index_t u = start; u <= stop; u++) 
            tsum += a[u];
        s[t] = tsum;
    }    

    index_t run = 0; 
    for(index_t t = 1; t <= nt; t++) {
        index_t v = s[t-1];
        s[t-1] = run;
        run += v;
    }
    s[nt] = run;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t trun = s[t];
        for(index_t u = start; u <= stop; u++) {
            index_t tv = a[u];
            a[u] = trun;
            trun += tv + k;
        }
        assert(trun == s[t+1]);
    }

#else

    index_t run = 0;
    for(index_t u = 0; u < n; u++) {
        index_t tv = a[u];
        a[u] = run;
        run += tv + k;
    }

#endif
    return run;
}

/**************************************************************** Indexing. */

// subset major index
#define FV_INDEX(v, n, k, X) ((index_t)(X) * (n) + (v))
#define BV_INDEX(v, n, k, X) (((index_t)(X) * (2*(n))) + (2*(v)))

/******************************************************** Root query builder. */

typedef struct steinerq
{
    index_t     n;
    index_t     m;
    index_t     k;
    index_t     *kk;
    index_t     *pos;
    index_t     *adj;
}steinerq_t;

steinerq_t *root_build(graph_t *g)
{
#ifdef TRACK_MEMORY
    push_memtrack();
#endif
    push_time();

    index_t n = g->n;
    index_t m = g->m;
    index_t k = g->k;
    index_t *kk = (index_t *) MALLOC(k*sizeof(index_t));
#ifdef BUILD_PARALLEL
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);
    index_t *pos = (index_t *) MALLOC((n+nt)*sizeof(index_t));
    index_t *adj = (index_t *) MALLOC(((n+nt)+(4*m)+(2*n*nt))*sizeof(index_t));
#else
    index_t *pos = (index_t *) MALLOC((n+1)*sizeof(index_t));
    index_t *adj = (index_t *) MALLOC(((n+1)+(4*m)+(2*n))*sizeof(index_t));
#endif

    steinerq_t *root = (steinerq_t *) MALLOC(sizeof(steinerq_t));
    root->n = n;
    root->m = m;
    root->k = k;
    root->kk  = kk;
    root->pos = pos;
    root->adj = adj;

    fprintf(stdout, "root build: ");

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
    index_t *e = g->edges;

#ifdef BUILD_PARALLEL
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) 
    {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) 
        {
            index_t u = e[j];
            index_t v = e[j+1];
            if(start <= u && u <= stop)
                pos[u]+=2; 
            if(start <= v && v <= stop)
                pos[v]+=2;
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3)
    {
        pos[e[j]]+=2;
        pos[e[j+1]]+=2;
    }
#endif

#ifdef BUILD_PARALLEL
    for(index_t th = 0; th < nt; th++)
        pos[n+th] = (2*n);

    index_t run = prefixsum(n+nt, pos, 1);
    assert(run == ((n+nt)+(4*m)+(2*n*nt)));
#else
    pos[n] = (2*n);
    index_t run = prefixsum(n+1, pos, 1);
    assert(run == ((n+1)+(4*m)+(2*n)));
#endif

    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        adj[pos[u]] = 0;

#ifdef BUILD_PARALLEL
    block_size = n/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) 
    {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) 
        {
            index_t u = e[j+0];
            index_t v = e[j+1];
            index_t w  = e[j+2];
            if(start <= u && u <= stop) 
            {
                index_t pu = pos[u];
                adj[pu + 1 + (2*adj[pu])] = v;
                adj[pu + 1 + ((2*adj[pu])+1)] = w;
                adj[pu]++;
            }
            if(start <= v && v <= stop) 
            {
                index_t pv = pos[v];
                adj[pv + 1 + (2*adj[pv])] = u;
                adj[pv + 1 + (2*adj[pv]+1)] = w;
                adj[pv]++;
            }
        }
    } 
#else
    for(index_t j = 0; j < 3*m; j+=3)
    {
        index_t u = e[j+0];
        index_t v = e[j+1];
        index_t w  = e[j+2];
        index_t pu = pos[u];
        index_t pv = pos[v];

        adj[pv + 1 + (2*adj[pv])] = u;
        adj[pv + 1 + ((2*adj[pv])+1)] = w;
        adj[pv]++;
        adj[pu + 1 + (2*adj[pu])] = v;
        adj[pu + 1 + ((2*adj[pu])+1)] = w;
        adj[pu]++; 
    }
#endif

#ifdef BUILD_PARALLEL
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++)
    {
        index_t u = n+th;
        index_t pu = pos[u];
        adj[pu] = 0;

        for(index_t v = 0; v < n; v++)
        {
            adj[pu + 1 + (2*adj[pu])] = v;
            adj[pu + 1 + ((2*adj[pu])+1)] = MATH_INF;
            adj[pu]++;
        }
    }
#else
    index_t u = n;
    adj[pos[u]] = 0;
    index_t pu = pos[u];
    for(index_t v = 0; v < n; v++)
    {
        adj[pu + 1 + (2*adj[pu])] = v;
        adj[pu + 1 + ((2*adj[pu])+1)] = MATH_INF;
        adj[pu]++;
    }
#endif

    pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
    index_t *tt = g->terminals;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < k; u++)
        kk[u] = tt[u];

    time = pop_time(); 
    fprintf(stdout, "[term: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
#ifdef TRACK_MEMORY
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
#endif
    fprintf(stdout, "\n");
    fflush(stdout);

    return root;
}

void steinerq_free(steinerq_t *root)
{
    if(root->pos != NULL)
        FREE(root->pos);
    if(root->adj != NULL)
        FREE(root->adj);
    if(root->kk != NULL)
        FREE(root->kk);
    FREE(root);
}

/********************************************************* Debug routines. */

#ifdef DEBUG
void print_nbits(index_t n, index_t N)
{            
    index_t bits[64];
    for(index_t i=0; i<64; i++)
        bits[i] = (N>>i)&0x01;
    for(index_t i=n-1; i >= 0; i--)
        fprintf(stdout, "%ld", bits[i]);
    fflush(stdout);
}

void print_bits(index_t n)
{
    fprintf(stdout, "n: %ld bits: ", n); 
    index_t size = sizeof(index_t)*8;

    for(index_t i = 0; i<size; i++)
    {   
        index_t mask = ((index_t)(0x01) << (size-1-i));
        fprintf(stdout, "%1d", ((n&mask) ? 1 : 0));
    }
    fprintf(stdout, "\n");
    fflush(stdout);
}

void print_array(index_t n, index_t *a)
{
    fprintf(stdout, "n: %ld a:", n);

    for(index_t i = 0; i < n; i++)
    {
        fprintf(stdout, " %ld", a[i]+1);
    }
    fprintf(stdout, "\n");
    fflush(stdout);
}

void print_adj(index_t u, index_t *pos, index_t *adj)
{
    index_t p = pos[u];
    index_t nu = adj[p];
    index_t *adj_u = adj + p + 1;
    fprintf(stdout, "adjacency list (%ld) : ", u);
    for(index_t i = 0; i < nu; i++)
        fprintf(stdout, " %ld %ld|", adj_u[2*i]+1, adj_u[2*i+1]);
    fprintf(stdout, "\n");
}

void print_dist(index_t n, index_t *d)
{
    fprintf(stdout, "Shortest distance: \n");
    for(index_t u = 0; u < n; u++)
        fprintf(stdout, "%10ld: %10ld\n", u+1, d[u]);
    fflush(stdout);
}


void print_dist_matrix(index_t *d_N, index_t n)
{
    fprintf(stdout, "distance matrix: \n");
    for(index_t u = 0; u < n; u++)
    {
        for(index_t v = 0; v < n; v ++)
        {
            fprintf(stdout, " %3ld", d_N[u*n+v]);
        }
        fprintf(stdout, "\n");
    }
    fflush(stdout);
}

void print_graph(graph_t *g)
{
    fprintf(stdout, "graph_t: \n");
    fprintf(stdout, "root: %ld\n", g->root);
    fprintf(stdout, "n: %ld\n", g->n);
    fprintf(stdout, "m: %ld\n", g->m);
    fprintf(stdout, "k: %ld\n", g->k);
    fprintf(stdout, "num edges: %ld\n", g->num_edges);
    fprintf(stdout, "num terminals: %ld\n", g->num_terminals);
    fprintf(stdout, "num coordinates: %ld\n", g->num_coordinates);
    fprintf(stdout, "edge_capacity: %ld\n", g->edge_capacity);
    fprintf(stdout, "edges: \n");
    for(index_t i = 0; i < g->num_edges; i++) 
    {    
        index_t *e = g->edges + (3*i);
        index_t u = e[0];
        index_t v = e[1];
        index_t w = e[2];
        fprintf(stdout, "E %ld %ld %ld\n", u+1, v+1, w);
    }

    fprintf(stdout, "terminals: \n");
    for(index_t i = 0; i < g->num_terminals; i++) 
        fprintf(stdout, "T %ld\n", g->terminals[i]+1);

    if(g->coordinates)
    {
        fprintf(stdout, "co-ordinates: \n");
        for(index_t i = 0; i < g->n; i++) 
        {    
            index_t *c = g->coordinates + (3*i);
            index_t v  = c[0];
            index_t dx = c[1];
            index_t dy = c[2];
            fprintf(stdout, "DD %ld %ld %ld\n", v+1, dx, dy);
        }   fprintf(stdout, "\n");
    }
    fflush(stdout);
}

void print_steinerq(steinerq_t *root)
{
    fprintf(stdout, "steinerq_t: \n");
    fprintf(stdout, "n: %ld\n", root->n);
    fprintf(stdout, "m: %ld\n", root->m);
    fprintf(stdout, "k: %ld\n", root->k);
    index_t *pos = root->pos;
    index_t *adj = root->adj;
    fprintf(stdout, "pos:");
    for(index_t i = 0; i < root->n; i++)
        fprintf(stdout, " %ld", pos[i]);
    fprintf(stdout, "\nadj:\n");
#ifdef BUILD_PARALLEL
    index_t n = root->n + num_threads();
#else
    index_t n = root->n + 1;
#endif
    for(index_t u = 0; u < n; u++)
    {
        index_t pu = pos[u];
        index_t adj_u = adj[pu];
        fprintf(stdout, "node: %ld edges: %ld|", u+1, adj_u);
        for(index_t i = 0; i < adj_u; i++)
        {
            fprintf(stdout, " %4ld %4ld|", 
                            adj[pu + 1 + (2*i)]+1, 
                            adj[pu + 1 + (2*i+1)]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
    fflush(stdout);
}

void print_f_v(index_t n, index_t k, index_t *f_v)
{
    fprintf(stdout, "f_v: \n");
    for(index_t v = 0; v < n; v++) 
    {
        fprintf(stdout, "v-> %ld\n", v+1);
        for(index_t X = 0; X < (1<<k); X++) 
        {
            index_t i_X = FV_INDEX(v, n, k, X);
            print_nbits(k, X);
            if(f_v[i_X] == MATH_INF)
                fprintf(stdout, " MATH_INF\n");
            else
                fprintf(stdout, " %ld\n", f_v[i_X]);
        }    
    }    
    fflush(stdout);
}

void print_b_v(index_t n, index_t k, index_t *b_v)
{
    fprintf(stdout, "b_v: \n");
    for(index_t v = 0; v < n; v++) 
    {    
        fprintf(stdout, "v-> %ld\n", v+1);
        for(index_t X = 0; X < (1<<k); X++) 
        {
            fprintf(stdout, "X: ");
            print_nbits(k, X);
            index_t i_bv = BV_INDEX(v, n, k, X);
            index_t v = b_v[i_bv];
            index_t Xd = b_v[i_bv + 1];
            fprintf(stdout, " v: %3ld Xd: ", v == -1 ? -1 : v+1);
            print_nbits(k, Xd); 
            fprintf(stdout, "\n");
        }
    }    
}

#endif

/******************************************************* Heap implementaions. */

/************************************************ Binary heap implementation. */
#ifdef BIN_HEAP
typedef struct bheap_item
{
    index_t item;
    index_t key; 
} bheap_item_t;

typedef struct bheap
{
    index_t max_n;
    index_t n;       // size of binary heap
    bheap_item_t *a; // stores (distance, vertex) pairs of the binary heap
    index_t *p;      // stores the positions of vertices in the binary heap
#ifdef TRACK_BANDWIDTH
    index_t key_comps;
    index_t mem;
#endif
} bheap_t;

bheap_t * bh_alloc(index_t n)
{
    bheap_t *h = (bheap_t *) malloc(sizeof(bheap_t));
    h->max_n = n;
    h->n = 0; 
    h->a = (bheap_item_t *) malloc((n+1)*sizeof(bheap_item_t));
    h->p = (index_t *) malloc(n*sizeof(index_t));
#ifdef TRACK_BANDWIDTH
    h->key_comps  = 0; 
    h->mem = 0;
#endif
#ifdef TRACK_MEMORY
    inc_malloc_total(sizeof(bheap_t) + 
                     ((n+1)*sizeof(bheap_item_t)) +
                     (n*sizeof(index_t)));
#endif
    return h;
}

void bh_free(bheap_t *h)
{
#ifdef TRACK_MEMORY
    index_t n = h->max_n;
    dec_malloc_total(sizeof(bheap_t) + 
                     ((n+1)*sizeof(bheap_item_t)) +
                     (n*sizeof(index_t)));
#endif
    free(h->a);
    free(h->p);
    free(h);
}

#ifdef BHEAP_DUMP
// debug function
void bh_dump(bheap_t *h)  
{
    fprintf(stdout, "Binary heap: \n");
    for(index_t i = 1; i <= h->n; i++) 
        fprintf(stdout, " %ld(%ld)", h->a[i].item, h->a[i].key);
    fprintf(stdout, "\n");

    for(index_t i = 2; i <= h->n; i++)
    {
        if(h->a[i].key < h->a[i/2].key)
            ERROR("key error at entry %ld, value %ld\n", i, h->a[i].key);
    }
    for(index_t i = 1; i <= h->n; i++)
    {
        if(h->p[h->a[i].item] != i)
            ERROR("indexing error at entry %ld", i);
    }
}
#endif

/************************************************** Binary heap operations. */

static void bh_siftup(bheap_t *h, index_t p, index_t q)
{
    index_t j = p;
    index_t k = 2 * p;
    bheap_item_t y = h->a[p];
#ifdef TRACK_BANDWIDTH
    index_t mem = 0;
    index_t key_comps = 0;
#endif

    while(k <= q)
    {
        bheap_item_t z = h->a[k];
        if(k < q)
        {
#ifdef TRACK_BANDWIDTH
            mem++;
            key_comps++;
#endif
            if(z.key > h->a[k + 1].key) z = h->a[++k];
        }

#ifdef TRACK_BANDWIDTH
        mem += 2;
        key_comps++;
#endif
        if(y.key <= z.key) break;
        h->a[j] = z;
        h->p[z.item] = j;
        j = k;
        k = 2 * j;
    }

    h->a[j] = y;
    h->p[y.item] = j;

#ifdef TRACK_BANDWIDTH
    h->mem += mem;
    h->key_comps += key_comps;
#endif
}

bheap_item_t bh_min(bheap_t *h)
{
    return (bheap_item_t) h->a[1];
}


static void bh_insert(bheap_t *h, index_t item, index_t key)
{
    index_t i = ++(h->n);
#ifdef TRACK_BANDWIDTH
    index_t mem = 0;
    index_t key_comps = 0;
#endif

    while(i >= 2)
    {
        index_t j = i / 2;
        bheap_item_t y = h->a[j];

#ifdef TRACK_BANDWIDTH
        mem ++;
        key_comps++;
#endif
        if(key >= y.key) break;

        h->a[i] = y;
        h->p[y.item] = i;
        i = j;
    }

    h->a[i].item = item;
    h->a[i].key = key;
    h->p[item] = i;
#ifdef TRACK_BANDWIDTH
    h->mem += mem;
    h->key_comps += key_comps;
#endif
}

static void bh_delete(bheap_t *h, index_t item)
{
    index_t n = --(h->n);
    index_t p = h->p[item];
#ifdef TRACK_BANDWIDTH
    index_t mem = 0;
    index_t key_comps = 0;
#endif

    if(p <= n)
    {
#ifdef TRACK_BANDWIDTH
        key_comps++;
        mem += 2;
#endif
        if(h->a[p].key <= h->a[n + 1].key)
        {
            h->a[p] = h->a[n + 1];
            h->p[h->a[p].item] = p;
            bh_siftup(h, p, n);
        }
        else
        {
            h->n = p - 1;
            bh_insert(h, h->a[n + 1].item, h->a[n+1].key);
            h->n = n;
        }
    }
#ifdef TRACK_BANDWIDTH
    h->mem += mem;
    h->key_comps += key_comps;
#endif
}

static void bh_decrease_key(bheap_t *h, index_t item, index_t new_key)
{
    index_t n = h->n;
    h->n = h->p[item] - 1;
    index_t i = ++(h->n);
#ifdef TRACK_BANDWIDTH
    index_t mem = 1;
    index_t key_comps = 0;
#endif

    while(i >= 2)
    {
        index_t j = i / 2;
        bheap_item_t y = h->a[j];

#ifdef TRACK_BANDWIDTH
        mem ++;
        key_comps++;
#endif
        if(new_key >= y.key) break;

        h->a[i] = y;
        h->p[y.item] = i;
        i = j;
    }

    h->a[i].item = item;
    h->a[i].key = new_key;
    h->p[item] = i;
#ifdef TRACK_BANDWIDTH
    h->mem += mem;
    h->key_comps += key_comps;
#endif
    h->n = n;
}

static index_t bh_delete_min(bheap_t * h)
{    
    bheap_item_t min = (bheap_item_t) h->a[1];
    index_t u = min.item;
    bh_delete((bheap_t *)h, u);
    return u;
}
#endif

/********************************************* Fibonacci heap implementation. */

#ifdef FIB_HEAP
typedef struct fheap_node
{
    struct fheap_node *parent;
    struct fheap_node *left; 
    struct fheap_node *right;
    struct fheap_node *child;
    index_t rank;
    index_t marked;
    index_t key;
    index_t vertex_no;
} fheap_node_t;

typedef struct fheap 
{
    fheap_node_t **trees;
    fheap_node_t **nodes;
    index_t max_nodes; 
    index_t max_trees;
    index_t n;
    index_t value;
#ifdef TRACK_BANDWIDTH
    index_t key_comps;
    index_t mem;
#endif
} fheap_t;

fheap_t *fh_alloc(index_t max_nodes)
{
    fheap_t *h; 
 
    // Create the heap.
    h = (fheap_t *) malloc(sizeof(fheap_t));
 
    h->max_trees = 1.0 + 1.44 * log(max_nodes)/log(2.0);
    h->max_nodes = max_nodes;
    h->trees = (fheap_node_t **) calloc(h->max_trees, sizeof(fheap_node_t *));
    h->nodes = (fheap_node_t **) calloc(max_nodes, sizeof(fheap_node_t *));
    h->n = 0;
    h->value = 0;
#ifdef TRACK_BANDWIDTH
    h->mem = 0;
    h->key_comps = 0;
#endif

#ifdef TRACK_MEMORY
    inc_malloc_total(sizeof(fheap_t) +
                     (h->max_trees*sizeof(fheap_node_t *)) +
                     (max_nodes*sizeof(fheap_node_t *)) +
                     (max_nodes*sizeof(fheap_node_t)));
#endif
    return h;  
}

void fh_free(fheap_t *h)
{
#ifdef TRACK_MEMORY
    dec_malloc_total(sizeof(fheap_t) +
                     (h->max_trees*sizeof(fheap_node_t *)) +
                     (h->max_nodes*sizeof(fheap_node_t *)) +
                     (h->max_nodes*sizeof(fheap_node_t)));
#endif
    //for(index_t i = 0; i < h->max_nodes; i++)
    //    free(h->nodes[i]);
    free(h->nodes);
    free(h->trees);
    free(h);
}

/*********************************************** Fibonacci heap operations. */

void fh_meld(fheap_t *h, fheap_node_t *tree_list)
{
    fheap_node_t *first, *next, *node_ptr, *new_root, *temp, *temp2, *lc, *rc;
    index_t r;

#ifdef TRACK_BANDWIDTH
    index_t mem = 0;
    index_t key_comps = 0;
#endif
    node_ptr = first = tree_list;
    do 
    {
        next = node_ptr->right;
        node_ptr->right = node_ptr->left = node_ptr;
        node_ptr->parent = NULL;

        new_root = node_ptr;
        r = node_ptr->rank;
#ifdef TRACK_BANDWIDTH
        mem++;
#endif
        do 
        {
            if((temp = h->trees[r])) 
            {
                h->trees[r] = NULL;
                h->value -= (1 << r);
#ifdef TRACK_BANDWIDTH
                mem++;
                key_comps++;
#endif
                if(temp->key < new_root->key) 
                {
                    temp2 = new_root;
                    new_root = temp;
                    temp = temp2;
                }

                if(r++ > 0) 
                {
                    rc = new_root->child;
                    lc = rc->left;
                    temp->left = lc;
                    temp->right = rc;
                    lc->right = rc->left = temp;
                }
                new_root->child = temp;
                new_root->rank = r;
                temp->parent = new_root;
                temp->marked = 0;
            }
            else 
            {
                h->trees[r] = new_root;
                h->value += (1 << r);;
                new_root->marked = 1;
#ifdef TRACK_BANDWIDTH
                mem++;
#endif
            }

        } while(temp);

        node_ptr = next;
    } while(node_ptr != first);

#ifdef TRACK_BANDWIDTH
    h->mem += mem;
    h->key_comps += key_comps;
#endif
}

void fh_insert(fheap_t *h, index_t vertex_no, index_t k)
{
    fheap_node_t *new_node;
    new_node = (fheap_node_t *) malloc(sizeof(fheap_node_t));
    new_node->child = NULL;
    new_node->left = new_node->right = new_node;
    new_node->rank = 0;
    new_node->vertex_no = vertex_no;
    new_node->key = k;

    h->nodes[vertex_no] = new_node;
    fh_meld(h, new_node);

    h->n++;
#ifdef TRACK_BANDWIDTH
    h->mem++;
#endif
}

index_t fh_delete_min(fheap_t *h)
{
    fheap_node_t *min_node, *child, *next;
    index_t k, k2;
    index_t r, v, vertex_no;
#ifdef TRACK_BANDWIDTH
    index_t mem = 0;
    index_t key_comps = 0;
#endif
    v = h->value;
    r = -1;
    while(v) 
    {
        v = v >> 1;
        r++;
    };

    min_node = h->trees[r];
    k = min_node->key;
#ifdef TRACK_BANDWIDTH
    mem++;
#endif
    while(r > 0) 
    {
        r--;
        next = h->trees[r];
        if(next) {
            if((k2 = next->key) < k) {
                k = k2;
                min_node = next;
            }
#ifdef TRACK_BANDWIDTH
            mem++;
            key_comps++;
#endif
        }
    }

    r = min_node->rank;
    h->trees[r] = NULL;
    h->value -= (1 << r);

    child = min_node->child;
    if(child) fh_meld(h, child);

    vertex_no = min_node->vertex_no;
    h->nodes[vertex_no] = NULL;
    free(min_node);
    h->n--;

#ifdef TRACK_BANDWIDTH
    h->mem += mem;
    h->key_comps += key_comps;
#endif
    return vertex_no;
}

void fh_decrease_key(fheap_t *h, index_t vertex_no, index_t new_value)
{
    fheap_node_t *cut_node, *parent, *new_roots, *r, *l;
    index_t prev_rank;

    cut_node = h->nodes[vertex_no];
    parent = cut_node->parent;
    cut_node->key = new_value;
#ifdef TRACK_BANDWIDTH
    index_t mem = 1;
#endif

    if(!parent) 
        return;

    l = cut_node->left;
    r = cut_node->right;
    l->right = r;
    r->left = l;
    cut_node->left = cut_node->right = cut_node;

    new_roots = cut_node;

    while(parent && parent->marked) 
    {
        parent->rank--;
        if(parent->rank) {
            if(parent->child == cut_node) parent->child = r;
        }
        else {
            parent->child = NULL;
        }

        cut_node = parent;
        parent = cut_node->parent;

        l = cut_node->left;
        r = cut_node->right;
        l->right = r;
        r->left = l;

        l = new_roots->left;
        new_roots->left = l->right = cut_node;
        cut_node->left = l;
        cut_node->right = new_roots;
        new_roots = cut_node;
#ifdef TRACK_BANDWIDTH
        mem += 2;
#endif
    }

    if(!parent) {
        prev_rank = cut_node->rank + 1;
        h->trees[prev_rank] = NULL;
        h->value -= (1 << prev_rank);
    }
    else {
        parent->rank--;
        if(parent->rank) {
            if(parent->child == cut_node) parent->child = r;
        }
        else {
            parent->child = NULL;
        }

        parent->marked = 1;
#ifdef TRACK_BANDWIDTH
        mem++;
#endif
    }

    fh_meld(h, new_roots);
#ifdef TRACK_BANDWIDTH
    h->mem += mem;
#endif
}

/************************************************** Debugging functions. */

#if FHEAP_DUMP
void fh_dump_nodes(fheap_node_t *ptr, index_t level)
{
    fheap_node_t *child_ptr, *partner;
    index_t i, ch_count;

    for(i = 0; i < level; i++) fprintf(stderr,"   ");

    fprintf(stderr, "%d(%ld)[%d]\n", ptr->vertex_no, ptr->key, ptr->rank);
    
    if((child_ptr = ptr->child)) 
    {
        child_ptr = ptr->child->right;

        ch_count = 0;

        do 
        {
            fh_dump_nodes(child_ptr, level+1);
            if(child_ptr->dim > ptr->dim) 
            {
                for(i = 0; i < level+1; i++) fprintf(stderr,"   ");
                fprintf(stderr,"error(dim)\n");  exit(1);
            }
            if(child_ptr->parent != ptr) 
            {
                for(i = 0; i < level+1; i++) fprintf(stderr,"   ");
                fprintf(stderr,"error(parent)\n");
            }
            child_ptr = child_ptr->right;
            ch_count++;
        } while(child_ptr != ptr->child->right);
            
        if(ch_count != ptr->dim) 
        {
            for(i = 0; i < level; i++) fprintf(stderr,"   ");
            fprintf(stderr,"error(ch_count)\n");  exit(1);
        }
    }
    else 
    {   
        if(ptr->dim != 0) 
        {
            for(i = 0; i < level; i++) fprintf(stderr,"   ");
            fprintf(stderr,"error(dim)\n"); exit(1);
        }
    }

}
#endif

/******************************************* Print out a Fibonacci heap. */
#if FHEAP_DUMP
void fh_dump(fheap_t *h)
{
    fprintf(stderr, "Fibonacci heap: \n");

    index_t i;
    fheap_node_t *ptr;

    fprintf(stderr, "\n");
    fprintf(stderr, "value = %d\n", h->value);
    fprintf(stderr, "array entries 0..max_trees =");
    for(i=0; i<h->max_trees; i++) {
        fprintf(stderr, " %d", h->trees[i] ? 1 : 0 );
    }
    fprintf("\n\n");
    for(i=0; i<h->max_trees; i++) 
    {
        if((ptr = h->trees[i])) 
        {
            fprintf(stderr, "tree %d\n\n", i);
            fh_dump_nodes(ptr, 0);
            fprintf(stderr, "\n");
        }
    }
    fflush(stderr);
}
#endif

#endif

/************************************************** Heap wrapper functions. */

#ifdef BIN_HEAP
// allocation
#define heap_alloc(n) bh_alloc((n))
#define heap_free(h) bh_free((bheap_t *)(h));
// heap operations
#define heap_insert(h, v, k) bh_insert((h), (v), (k))
#define heap_delete_min(h) bh_delete_min((h));
#define heap_decrease_key(h, v, k) bh_decrease_key((h), (v), (k));
// fetch structure elements
#define heap_n(h) ((bheap_t *)h)->n;
#define heap_key_comps(h) ((bheap_t *)h)->key_comps;
#define heap_mem(h) (h)->mem;
// debug
#define heap_dump(h) bh_dump((bheap_t *)(h));

#define heap_node_t bheap_item_t
#define heap_t bheap_t
#endif

#ifdef FIB_HEAP
// allocation
#define heap_alloc(n) fh_alloc((n))
#define heap_free(h) fh_free((fheap_t *)(h));
// heap operations
#define heap_insert(h, v, k) fh_insert((h), (v), (k));
#define heap_delete_min(h) fh_delete_min((h));
#define heap_decrease_key(h, v, k) fh_decrease_key((h), (v), (k));
// fetch structure elements
#define heap_n(h) ((fheap_t *)h)->n;
#define heap_key_comps(h) ((fheap_t *)h)->key_comps;
#define heap_mem(h) h->mem;
// debug
#define heap_dump(h) fh_dump((fheap_t *)(h));

#define heap_node_t fheap_node_t
#define heap_t fheap_t
#endif

/************************************************** Dijkstra shortest path*/

void dijkstra(index_t n,
              index_t m, 
              index_t *pos, 
              index_t *adj, 
              index_t s, 
              index_t *d,
              index_t *visit
#ifdef TRACK_OPTIMAL
              ,index_t *p
#endif
#ifdef TRACK_BANDWIDTH
              ,index_t *heap_ops
#endif
             )
{
#ifdef DIJKSTRA_BENCHMARK
    push_time();
    fprintf(stdout, "dijkstra: ");
    fflush(stdout);
#endif

    heap_t *h = heap_alloc(n);

#ifdef DIJKSTRA_BENCHMARK
#ifdef TRACK_MEMORY
    push_memtrack();
#endif
    push_time();
#endif

    for(index_t v = 0; v < n; v++)
    {
        d[v]     = MAX_DISTANCE; // mem: n
        visit[v] = 0; // mem: n
#ifdef TRACK_OPTIMAL
        p[v] = UNDEFINED; // mem: n
#endif
    }
    d[s] = 0;

#ifdef DIJKSTRA_BENCHMARK
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    push_time();
    push_time();
#endif

    for(index_t v = 0; v < n; v++)
        heap_insert(h, v, d[v]);

#ifdef DIJKSTRA_BENCHMARK
    time = pop_time();
    fprintf(stdout, "[hinsert: %.2lf ms] ", time);
    fflush(stdout);
    push_time();
#endif

    //visit and label
    while(h->n > 0)
    {
        index_t u = heap_delete_min(h); 
        visit[u]  = 1;

        index_t pos_u  = pos[u];
        index_t *adj_u = adj + pos_u;
        index_t n_u  = adj_u[0];
        for(index_t i = 1; i <= 2*n_u; i += 2)
        {
            index_t v   = adj_u[i];
            index_t d_v = d[u] + adj_u[i+1];
            if(!visit[v] && d[v] > d_v)
            {
                d[v] = d_v;
                heap_decrease_key(h, v, d_v);
#ifdef TRACK_OPTIMAL
                p[v] = u;
#endif
            }
        }
        // mem: 2n+6m
    }

#ifdef DIJKSTRA_BENCHMARK
    time = pop_time();
    fprintf(stdout, "[visit: %.2lf ms] ", time);

    time = pop_time();
    double trans_rate = 0;

#ifdef TRACK_BANDWIDTH
#ifdef TRACK_OPTIMAL
    index_t mem_graph = 3*n+6*m; 
#else
    index_t mem_graph = 4*n+6*m;
#endif
    index_t mem_heap = heap_mem(h);
    index_t trans_bytes = (mem_graph * sizeof(index_t) +
                           mem_heap * sizeof(heap_node_t));
    trans_rate = trans_bytes / (time/1000.0);
#endif
    fprintf(stdout, "[total: %.2lf ms %.2lfGiB/s] ", 
                    time,
                    trans_rate/((double) (1<<30)));

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
#ifdef TRACK_MEMORY
    print_pop_memtrack();
#endif
    fprintf(stdout, "\n");
    fflush(stdout);
#endif

#ifdef TRACK_BANDWIDTH
    *heap_ops = heap_mem(h);
#endif
    heap_free(h);
}

/*************************************************** Traceback Steiner tree. */

#ifdef TRACK_OPTIMAL
void list_solution(graph_t *g)
{
    index_t m = g->num_edges;
    index_t *e = g->edges;
    index_t i = 0;

    fprintf(stdout, "solution: [");
    for(i = 0; i < (3*m-3); i+=3)
    {
        index_t u = e[i];
        index_t v = e[i+1];
        //index_t w = e[i+2];
        fprintf(stdout, "\"%ld %ld\", ", u+1, v+1);
    }
    index_t u = e[i];
    index_t v = e[i+1];
    //index_t w = e[i+2];
    fprintf(stdout, "\"%ld %ld\"", u+1, v+1);
    fprintf(stdout, "]\n");
    fflush(stdout);
}

void backtrack(index_t n, index_t k, index_t v, 
               index_t X, index_t *kk, index_t *b_v,
               graph_t *g)
{
    if(X == 0 || v == -1)
        return;

    index_t i_X = BV_INDEX(v, n, k, X);
    index_t u = b_v[i_X];

    if(v != u)
    {
        graph_add_edge(g, v, u, 1);
        index_t Xd = b_v[i_X+1];
        backtrack(n, k, u, Xd, kk, b_v, g);
    }
    else
    {
        index_t Xd = b_v[i_X+1];
        index_t X_Xd = (X & ~Xd);
        if(X == Xd)
            return;
        backtrack(n, k, u, Xd, kk, b_v, g);
        backtrack(n, k, u, X_Xd, kk, b_v, g);
    }
}

graph_t * build_tree(index_t n, index_t k, index_t *kk, index_t *b_v)
{
    index_t c = k-1;
    index_t C = (1<<c)-1;
    index_t q = kk[k-1];

    graph_t *g = graph_alloc();
    g->n = n;
    backtrack(n, k, q, C, kk, b_v, g);
    return g;
}

graph_t * tracepath(index_t n, index_t s, index_t v, index_t *p)
{
    index_t u = p[v];

    graph_t *g = graph_alloc();
    g->n = n;
    while(u != s)
    {
        graph_add_edge(g, v, u, 1);
        v = u;
        u = p[v];
    }
    graph_add_edge(g, v, u, 1);
    return g;
}

#endif

/**************************************************** Erickson Monma Veinott. */

index_t emv_kernel(index_t n, 
                    index_t m, 
                    index_t k, 
                    index_t c, 
                    index_t C, 
                    index_t q, 
                    index_t *kk, 
                    index_t *f_v, 
                    index_t *pos, 
                    index_t *adj, 
                    index_t *d,  
                    index_t *visit,
                    index_t nt
#ifdef TRACK_OPTIMAL
                    ,index_t *p
                    ,index_t *b_v 
#endif
#ifdef TRACK_BANDWIDTH
                    ,index_t *heap_ops
#endif
                    )
{
    // initialisation
    index_t block_size = k/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) // one thread per core
    {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? k-1 : (start+block_size-1);
        index_t *d_th = d + (n+nt)*th;
        index_t *visit_th = visit + (n+nt)*th;
#ifdef TRACK_OPTIMAL
        index_t *p_th = p + (n+nt)*th;
#endif
#ifdef TRACK_BANDWIDTH
        index_t *heap_ops_th = heap_ops + th;
#endif

        for(index_t t = start; t <= stop; t++) 
        {    
            dijkstra(n+1, m, pos, adj, kk[t], d_th, visit_th
#ifdef TRACK_OPTIMAL
                     ,p_th
#endif
#ifdef TRACK_BANDWIDTH
                     ,heap_ops_th
#endif
                     );
            index_t *f_t = f_v + FV_INDEX(0, n, k, 1<<t);
#ifdef TRACK_OPTIMAL
            index_t *b_t = b_v + BV_INDEX(0, n, k, 1<<t);
#endif

            for(index_t v = 0; v < n; v++) 
            {
                f_t[v] = d_th[v]; 
#ifdef TRACK_OPTIMAL
                b_t[2*v] = kk[t];
                b_t[2*v + 1] = (1<<t);
#endif
            }
            // mem: 2*k*n
        }    
    }

    for(index_t m = 2; m <= k; m++) // k-2
    {    
        index_t kCm = choose(k, m);
     
        index_t i = 0; 
        index_t *X_a = (index_t *) MALLOC(kCm * sizeof(index_t));

        index_t z = 0;
        for(index_t X = (1<<m)-1;
            X < (1<<k);
            z = X|(X-1), X = (z+1)|(((~z & -~z)-1) >> (__builtin_ctz(X) + 1))) // cCm
        {
            X_a[i++] = X;
        }

        index_t block_size = kCm/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) // one thread per core
        {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? kCm-1 : (start+block_size-1);
            index_t *d_th = d + (n+nt)*th;
            index_t *visit_th = visit + (n+nt)*th;
#ifdef TRACK_BANDWIDTH
            index_t *heap_ops_th = heap_ops + th;
#endif
            for(index_t i = start; i <= stop; i++)
            {
                index_t X = X_a[i]; // mem: 2^k
                index_t *f_X    = f_v + FV_INDEX(0, n, k, X);
#ifdef TRACK_OPTIMAL
                index_t *b_X  = b_v + BV_INDEX(0, n, k, X);
#endif
                // bit twiddling hacks: generating proper subsets of X 
                index_t Xd = 0;
                for(Xd = X & (Xd - X); Xd != X; Xd = X & (Xd - X)) // 2^m - 2
                {
                    index_t X_Xd = (X & ~Xd); // X - X' 
                    index_t *f_Xd   = f_v + FV_INDEX(0, n, k, Xd);
                    index_t *f_X_Xd = f_v + FV_INDEX(0, n, k, X_Xd);
                    for(index_t v = 0; v < n; v++)
                    {
                        index_t min_Xd = f_Xd[v] + f_X_Xd[v];
                        if(min_Xd < f_X[v])
                        {
                            f_X[v] = min_Xd;
#ifdef TRACK_OPTIMAL    
                            b_X[2*v] = v;
                            b_X[2*v + 1] = Xd;
#endif
                        }
                        // mem: 3^k*3n
                    }
                }

                index_t s = n + th;
                index_t ps = pos[s];
                index_t *adj_s = adj + (ps+1);
#ifdef TRACK_OPTIMAL
                index_t *p_th = p + (n+nt)*th;
#endif
                for(index_t u = 0; u < n; u++)
                    adj_s[2*u+1] = f_X[u]; // mem: 2^k * n

                for(index_t t = 0; t < k; t++)
                {
                    if(!(X & (1<<t)))
                        continue;
                    index_t u     = kk[t];
                    index_t X_u   = (X & ~(1<<t));
                    index_t i_X_u = FV_INDEX(u, n, k, X_u);
                    adj_s[2*u+1]  = f_v[i_X_u];
                }

                dijkstra(n+nt, m+n, pos, adj, s, d_th, visit_th
#ifdef TRACK_OPTIMAL
                         ,p_th
#endif
#ifdef TRACK_BANDWIDTH
                         ,heap_ops_th
#endif
                         );
                for(index_t v = 0; v < n; v++)
                {
                    f_X[v] = d_th[v];
                    // mem: 2^k * n 
#ifdef TRACK_OPTIMAL
                    index_t u = p_th[v];
                    if(u != s)
                    {
                        b_X[2*v] = u;
                        b_X[2*v + 1] = X;
                    }
                    // mem: 2^k * 2n 
#endif
                }
            }
        }
        FREE(X_a);
    }

    //print_b_v(n, k, b_v);
    index_t i_q_C  = FV_INDEX(q, n, k, C);
    return f_v[i_q_C];
}

index_t erickson_monma_veinott(steinerq_t *root, index_t list_soln)
{
#ifdef TRACK_MEMORY
    push_memtrack();
#endif
    push_time();

    double time;
    index_t n   = root->n;
    index_t m   = root->m;
    index_t k   = root->k;
    index_t *kk = root->kk;
    index_t min_cost = 0;

#ifdef TRACK_OPTIMAL
    graph_t *g = NULL;
#endif

    if(k == 2)
    {
        index_t u   = kk[0];
        index_t v   = kk[1];
        index_t *d  = (index_t *) MALLOC(n*sizeof(index_t));
        index_t *visit = (index_t *) MALLOC(n*sizeof(index_t));
#ifdef TRACK_OPTIMAL
        index_t *p  = (index_t *) MALLOC(n*sizeof(index_t));
#endif
#ifdef TRACK_BANDWIDTH
        index_t heap_ops = 0;
#endif
        fprintf(stdout, "erickson: ");
        push_time();
        dijkstra(n, m, root->pos, root->adj, u, d, visit
#ifdef TRACK_OPTIMAL
                ,p
#endif
#ifdef TRACK_BANDWIDTH
                ,&heap_ops
#endif
                );
        time = pop_time();

        // compute bandwidth
        double trans_rate = 0;
#ifdef TRACK_OPTIMAL
        index_t mem_graph = 5*n+6*m;
#else
        index_t mem_graph = 4*n+6*m;
#endif

#ifdef TRACK_BANDWIDTH
        index_t trans_bytes = mem_graph*sizeof(index_t) +
                              heap_ops*sizeof(heap_node_t);
        trans_rate = trans_bytes / (time / 1000.0);
#endif
        fprintf(stdout, "[kernel: %.2lf ms %.2lfGiB/s] ", 
                        time, trans_rate / (1<<30));

        min_cost = d[v];
#ifdef TRACK_OPTIMAL
        g = tracepath(n, u, v, p);
#endif

        FREE(d);
        FREE(visit);
#ifdef TRACK_OPTIMAL
        FREE(p);
#endif
    }
    else
    {
        index_t nt = num_threads();
        assert(nt < MAX_THREADS);

        index_t *f_v = (index_t *) MALLOC(n*(1<<k)*sizeof(index_t));

#ifdef BUILD_PARALLEL
        index_t *d     = (index_t *) MALLOC(nt*(n+nt)*sizeof(index_t));
        index_t *visit = (index_t *) MALLOC(nt*(n+nt)*sizeof(index_t));
#else
        index_t *d      = (index_t *) MALLOC((n+1)*sizeof(index_t));
        index_t *visit  = (index_t *) MALLOC((n+1)*sizeof(index_t));
#endif

#ifdef TRACK_OPTIMAL
        index_t *b_v = (index_t *) MALLOC(2*n*(1<<k)*sizeof(index_t));
#ifdef BUILD_PARALLEL
        index_t *p = (index_t *) MALLOC(nt*(n+nt)*sizeof(index_t));
#else
        index_t *p = (index_t *) MALLOC((n+1)*sizeof(index_t));
#endif
#endif

#ifdef TRACK_BANDWIDTH
#ifdef BUILD_PARALLEL
        index_t *heap_ops = (index_t *) MALLOC(nt*sizeof(index_t));
#else
        index_t *heap_ops = (index_t *) MALLOC(sizeof(index_t));
#endif
#endif

        // initialisation
        push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t i = 0; i < ((index_t)n*(1<<k)); i++)
            f_v[i] = MATH_INF;

#ifdef TRACK_OPTIMAL
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t i = 0; i < (index_t)(2*n*(1<<k)); i+=2)
        {
            b_v[i]   = -1;
            b_v[i+1] = 0;
        }
#endif
        time = pop_time();
        fprintf(stdout, "erickson: [zero: %.2lf ms] ", time);

        index_t q = kk[k-1];
        index_t c = k-1;
        index_t C = (1<<c)-1;

        // call kernel: do the hard work
        push_time();
        min_cost = emv_kernel(n, m, k, c, C, q, kk, f_v, root->pos,
                              root->adj, d, visit, nt
#ifdef TRACK_OPTIMAL
                              ,p
                              ,b_v
#endif
#ifdef TRACK_BANDWIDTH
                              ,heap_ops
#endif
                              );
        time = pop_time();

        double trans_rate = 0;
        // bandwidth calculation
#ifdef TRACK_BANDWIDTH
        index_t total_heap_ops = 0;
        for(index_t th = 0; th < nt; th++)
            total_heap_ops += heap_ops[th];
#ifdef TRACK_OPTIMAL
        index_t mem_graph = 5*n+6*m;
#else
        index_t mem_graph = 4*n+6*m;
#endif

        //mem: 3^{k+1}*n + 2^k*mem_graph*sizeof(index_t) + mem_heap*sizeof(heap_node_t)) 
        index_t trans_bytes = (((index_t)(pow(3,k+1))*n)+((index_t)(pow(2,k)*mem_graph)) 
                               *sizeof(index_t))+(total_heap_ops * sizeof(heap_node_t));
        trans_rate   = trans_bytes / (time / 1000.0);
#endif
        fprintf(stdout, "[kernel: %.2lf ms %.2lfGiB/s] ",
                        time, trans_rate/(1 << 30));

        // build a Steiner tree
#ifdef TRACK_OPTIMAL
        if(list_soln)
        {
            push_time();
            g = build_tree(n, k, kk, b_v);
            time = pop_time();
            fprintf(stdout, "[traceback: %.2lf ms] ", time);
        }
#endif

        FREE(d);
        FREE(f_v); 
        FREE(visit);
#ifdef TRACK_OPTIMAL
        FREE(p);
        FREE(b_v);
#endif
#ifdef TRACK_BANDWIDTH
        FREE(heap_ops);
#endif
    }

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] [cost: %ld] ", time, min_cost);
#ifdef TRACK_MEMORY
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
#endif
    fprintf(stdout, "\n");
    fflush(stdout);

    // list a solution
#ifdef TRACK_OPTIMAL
    if(list_soln)
    {
        list_solution(g);
        graph_free(g);
    }
#endif
    return min_cost;
}

/******************************************************* Program entry point. */

#define CMD_NOP                 0
#define CMD_DIJKSTRA            1
#define CMD_EDGE_LINEAR         2

char *cmd_legend[] = { "no operation", 
                       "Dijkstra Single-Source-Shortest-Path", 
                       "Erickson-Monma-Veinott"};

int main(int argc, char **argv)
{
    push_time();
#ifdef TRACK_MEMORY
    push_memtrack();
#endif

    index_t seed = 123456789;
    index_t have_seed = 0;
    index_t arg_cmd = CMD_NOP;
    index_t list_soln = 0;
    index_t file_input = 0; 
    char *filename = NULL;
    for(index_t f = 1; f < argc; f++) 
    {
        if(argv[f][0] == '-') 
        {
            if(!strcmp(argv[f], "-dijkstra"))
            {
                arg_cmd = CMD_DIJKSTRA;
            }
            if(!strcmp(argv[f], "-el") || !strcmp(argv[f], "-erickson"))
            {
                arg_cmd = CMD_EDGE_LINEAR; 
            }
            if(!strcmp(argv[f], "-list"))
            {
                list_soln = 1;
            }
            if(!strcmp(argv[f], "-in")) 
            {
                if(f == argc - 1) 
                    ERROR("file name missing from command line");
                filename = argv[++f];
                file_input = 1; 
            }
            if(!strcmp(argv[f], "-seed")) 
            {
                if(f == argc - 1) 
                    ERROR("random seed missing from command line");
                seed = atol(argv[++f]);
                have_seed = 1; 
            }
            if(!strcmp(argv[f], "-h") || !strcmp(argv[f], "-help"))
            {
                fprintf(stdout, "usage: %s -in <input graph> <arguments>\n"
                        "\n"
                        "arguments :\n"
                        "\t-seed : seed value\n"
                        "\t-el : Erickson-Monma-Veinott algorithm\n"
                        "\t-dijkstra : Dijkstra single source shortest path\n"
                        "\t-list : Output Steiner tree\n"
                        "\n",
                        argv[0]);
                return 0;
            }
        }
    }
 
    fprintf(stdout, "invoked as:");
    for(index_t f = 0; f < argc; f++) 
        fprintf(stdout, " %s", argv[f]);
    fprintf(stdout, "\n");

    FILE *in = stdin;
    if(file_input)
    {
        in = fopen(filename, "r");
        if(in == NULL)
            ERROR("unable to open file '%s'", filename);
    }
    else
    {
        fprintf(stdout, 
                "no input file specified, defaulting to stdin\n");
    }

    if(have_seed == 0) 
    { 
        fprintf(stdout, 
                "no random seed given, defaulting to %ld\n", seed);
    }    
    fprintf(stdout, "random seed = %ld\n", seed);

    
    graph_t *g = graph_load(in);
    steinerq_t *root = root_build(g);
    index_t min_cost  = g->cost;
    graph_free(g);

    fprintf(stdout, "command: %s\n", cmd_legend[arg_cmd]);
    fflush(stdout);
    push_time();
    switch(arg_cmd)
    {
        case CMD_NOP:
            steinerq_free(root);
            break;

        case CMD_DIJKSTRA:
            {
                srand(seed);
                index_t n = root->n;
                index_t m = root->m;
                index_t s = rand() % n; // source vertex
                index_t *d = (index_t *) MALLOC(n*sizeof(index_t));
                index_t *visit = (index_t *) MALLOC(n*sizeof(index_t));
#ifdef TRACK_OPTIMAL
                index_t *p = (index_t *) MALLOC(n*sizeof(index_t));
#endif
#ifdef TRACK_BANDWIDTH
                index_t trans_rate = 0;
#endif

                dijkstra(n, m, root->pos, root->adj, s, d, visit
#ifdef TRACK_OPTIMAL
                         ,p
#endif
#ifdef TRACK_BANDWIDTH
                         ,&trans_rate
#endif
                        );
                FREE(d);
                FREE(visit);
#ifdef TRACK_OPTIMAL
                FREE(p);
#endif
                steinerq_free(root);
            }
            break;

        case CMD_EDGE_LINEAR:
            {
                index_t cost = erickson_monma_veinott(root, list_soln);
                if(min_cost != -1 && min_cost != cost)
                    ERROR("min_cost != cost: minimum cost = %ld, cost = %ld", 
                           min_cost, cost);
                steinerq_free(root);
            }
            break;

        default:
            break;
    }

    double time = pop_time();
    fprintf(stdout, "command done [%.2lf ms]\n", time);
    time = pop_time();
    fprintf(stdout, "grand total [%.2lf ms] ", time);
#ifdef TRACK_MEMORY
    print_pop_memtrack();
#endif
    fprintf(stdout, "\n");
    fprintf(stdout, "host: %s\n", sysdep_hostname());
    fprintf(stdout, "build: %s, %s, %s\n",
                    "edge-linear kernel"
#ifdef BUILD_PARALLEL
                    ,"multi-threaded"
#else
                    ,"single thread"
#endif
#ifdef FIB_HEAP
                    ,"Fibonacci heap"
#else
                    ,"binary heap"
#endif
            );
    fprintf(stdout, "list solution: %s\n", (list_soln ? "true":"false"));
    fprintf(stdout, "num threads: %ld\n", num_threads());
    fprintf(stdout, 
            "compiler: gcc %d.%d.%d\n",
            __GNUC__,
            __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
    fflush(stdout);

#ifdef TRACK_MEMORY
    assert(malloc_balance == 0);
    assert(memtrack_stack_top < 0);
#endif
    return 0;
}
