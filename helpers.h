

typedef struct {
    /* Scalars */
    int surface_selector;
    int delta_selector;
    int steps_between;
    int max_touches;
    int num_runs;         
    double l_max;
    /* Ranges */
    double *rangemu_LevyDistrib;
    int len_rangemu_LevyDistrib;

    double *range_diam;
    int len_range_diam;

    double *range_disk_diameter; 
    int len_range_disk_diameter;

    int *range_side;
    int len_range_side;

    double *range_delta;
    int len_range_delta;

    char **list_shapes;
    int len_list_shapes;

    int *range_nwalkers;
    int len_range_nwalkers;

    int *range_ntargets;
    int len_range_ntargets;

    double *range_probability;
    int len_range_probability;

    char* reference_shape;

    /* Output */
    char *save_directory;
    char *file_name;       /* NEW: output / reference filename */

} Config;

/* =========================
   Utility
   ========================= */

int count_items(const char *s);

/* =========================
   Parsing helpers
   ========================= */

double *parse_double_array(char *str, int *len);

int *parse_int_array(char *str, int *len);

char **parse_string_array(char *str, int *len);

char *parse_string(char *str);

/* =========================
   Config loader
   ========================= */

void load_config(const char *path, Config *cfg);

/* =========================
   Cleanup
   ========================= */

void free_config(Config *cfg);

char *build_output_path(const char *save_dir, const char *file_name);

