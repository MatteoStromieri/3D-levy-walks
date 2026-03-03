#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helpers.h"

/* =========================
   Utility
   ========================= */

int count_items(const char *s) {
    int count = 1;
    for (; *s; s++)
        if (*s == ',') count++;
    return count;
}

/* =========================
   Parsing helpers
   ========================= */

double *parse_double_array(char *str, int *len) {
    *len = count_items(str);
    double *arr = malloc((*len) * sizeof(double));
    if (!arr) return NULL;

    char *token = strtok(str, ",");
    for (int i = 0; i < *len; i++) {
        arr[i] = atof(token);
        token = strtok(NULL, ",");
    }
    return arr;
}

int *parse_int_array(char *str, int *len) {
    *len = count_items(str);
    int *arr = malloc((*len) * sizeof(int));
    if (!arr) return NULL;

    char *token = strtok(str, ",");
    for (int i = 0; i < *len; i++) {
        arr[i] = atoi(token);
        token = strtok(NULL, ",");
    }
    return arr;
}

char **parse_string_array(char *str, int *len) {
    *len = count_items(str);
    char **arr = malloc((*len) * sizeof(char *));
    if (!arr) return NULL;

    char *token = strtok(str, ",");
    for (int i = 0; i < *len; i++) {
        arr[i] = strdup(token);
        token = strtok(NULL, ",");
    }
    return arr;
}

char *parse_string(char *str) {
    return strdup(str);
}

/* =========================
   Config loader
   ========================= */

void load_config(const char *path, Config *cfg) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("Error opening config file");
        exit(EXIT_FAILURE);
    }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || strlen(line) < 3)
            continue;

        char *key = strtok(line, "=");
        char *value = strtok(NULL, "\n");
        if (!key || !value)
            continue;

        while (*value == ' ') value++;

        printf("Parsing key: %s, value: %s\n", key, value);
        /* Scalars */
        if (!strcmp(key, "surface_selector"))
            cfg->surface_selector = atoi(value);
        else if (!strcmp(key, "delta_selector"))
            cfg->delta_selector = atoi(value);
        else if (!strcmp(key, "steps_between"))
            cfg->steps_between = atoi(value);
        else if (!strcmp(key, "max_touches"))
            cfg->max_touches = atoi(value);
        else if (!strcmp(key, "num_runs"))         /* NEW */
            cfg->num_runs = atoi(value);
        else if (!strcmp(key, "l_max"))         /* NEW */
            cfg->l_max = strtod(value, NULL);

        /* Ranges */
        else if (!strcmp(key, "rangemu_LevyDistrib"))
            cfg->rangemu_LevyDistrib =
                parse_double_array(value, &cfg->len_rangemu_LevyDistrib);

        else if (!strcmp(key, "range_diam"))
            cfg->range_diam =
                parse_double_array(value, &cfg->len_range_diam);

        else if (!strcmp(key, "range_disk_diameter"))   /* NEW */
            cfg->range_disk_diameter =
                parse_double_array(value, &cfg->len_range_disk_diameter);

        else if (!strcmp(key, "range_side"))
            cfg->range_side =
                parse_int_array(value, &cfg->len_range_side);

        else if (!strcmp(key, "range_delta"))
            cfg->range_delta =
                parse_double_array(value, &cfg->len_range_delta);

        else if (!strcmp(key, "list_shapes"))
            cfg->list_shapes =
                parse_string_array(value, &cfg->len_list_shapes);

        else if (!strcmp(key, "range_nwalkers"))
            cfg->range_nwalkers =
                parse_int_array(value, &cfg->len_range_nwalkers);

        else if (!strcmp(key, "range_ntargets"))
            cfg->range_ntargets =
                parse_int_array(value, &cfg->len_range_ntargets);

        else if (!strcmp(key, "range_probability"))
            cfg->range_probability =
                parse_double_array(value, &cfg->len_range_probability);
        
        else if (!strcmp(key, "reference_shape")) { 
            cfg->reference_shape = parse_string(value);
        }
        
        /* Output */
        else if (!strcmp(key, "save_directory")) {
            printf("Parsing save_directory: %s\n", value);
            cfg->save_directory = parse_string(value);
        }

        else if (!strcmp(key, "file_name"))            /* NEW */
            cfg->file_name = parse_string(value);
    }

    fclose(f);
}

/* =========================
   Cleanup
   ========================= */

void free_config(Config *cfg) {
    free(cfg->rangemu_LevyDistrib);
    free(cfg->range_diam);
    free(cfg->range_disk_diameter);   /* NEW */
    free(cfg->range_side);
    free(cfg->range_delta);
    free(cfg->range_nwalkers);
    free(cfg->range_ntargets);
    free(cfg->range_probability);

    if (cfg->list_shapes) {
        for (int i = 0; i < cfg->len_list_shapes; i++)
            free(cfg->list_shapes[i]);
        free(cfg->list_shapes);
    }

    free(cfg->save_directory);
    free(cfg->file_name);             /* NEW */
}

char *build_output_path(const char *save_dir, const char *file_name) {
    if (!save_dir || !file_name) return NULL;

    // +2 for possible '/' and null terminator
    size_t len = strlen(save_dir) + strlen(file_name) + 2;
    char *full_path = malloc(len);
    if (!full_path) return NULL;

    // Add '/' if save_dir does not already end with '/'
    if (save_dir[strlen(save_dir) - 1] == '/')
        snprintf(full_path, len, "%s%s", save_dir, file_name);
    else
        snprintf(full_path, len, "%s/%s", save_dir, file_name);

    return full_path;
}

