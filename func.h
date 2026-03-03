// Max values for array sizing (adjust as per your actual max mu/lmax)
#define MAX_MU_INDEX 11
#define MAX_LMAX_VALUE 1200

typedef struct {
    double detection_time;
    int first_touch_steps;
    int second_touch_steps;
} Result;


extern double a_coeffs[MAX_MU_INDEX][MAX_LMAX_VALUE + 1]; // Store 'a' values
extern double expected_lengths[MAX_MU_INDEX][MAX_LMAX_VALUE + 1]; // Store ExpectedLength values

double get_normalization_constant(double mu, int lmax);

double get_diameter_from_surface(double surface, const char* shape);

double get_diameter_from_projected_surface(double surface, const char* shape);

Result LevySearch3D_MultiWalker(int, const char*, double , double, int,
                                double , const char* , int ,
                                double, double, double, int, int, int, double);

double LevySearch3D_MultiWalker_jump_and_random_dir(int, const char*, double, double, int,
                                int, const char*, int,
                                double, double, int*, double*, double);

int get_mu_index(double);

double toroidal_distance_squared(double p1[3], double p2[3], double side_length);

double toroidal_distance_squared_2D(double p1[2], double p2[2], double side_length);

double toroidal_distance(double p1[3], double p2[3], double side_length);

double Levy(double, int, double);

int get_mu_index(double);

double LevySearch3D_SingleWalker_boundary_detection(const char* , double , double , int ,int, const char*,int, double, double , int*, double);
double LevySearch3D_SingleWalker_distance(const char* , double , double , int ,int, const char*,int, double, double , double*, double);
double LevySearch3D_SingleWalker_boundary(const char*, double, double, int, int, const char*, int, double, double, double);