#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // Per strcmp e per gestire le stringhe
#include "func.h"

int get_mu_index(double mu_val) {
    return (int)round((mu_val - 1.0) / 0.2);
}

double get_diameter_from_surface(double surface, const char* shape){
    // + 0.5 is for the approximation
    if (strcmp(shape, "Ball") == 0 || strcmp(shape, "Ball_boundary") == 0)
        return (sqrt((double)surface / M_PI) - 2.0);
    if (strcmp(shape, "Ball_no_boundary") == 0)
        return sqrt((double)surface / M_PI);
    if (strcmp(shape, "Line") == 0) 
        return (((double) surface / (2.0*M_PI) - 2.0));
    if (strcmp(shape, "Disk") == 0)
        return (-2.0*M_PI + sqrt(4.0 * M_PI * M_PI + 2.0 * (double) surface)); 
    return -1; // Error case
}


double get_normalization_constant(double mu, int lmax){
    if (mu == 1.0){
        return 1 / (1 + log((double)lmax));
    } else {
        return 1 / (1 - 1 / (mu - 1.0) * (pow((double)lmax, 1.0-mu) -1.0) );
    }
}

double toroidal_distance_squared(double p1[3], double p2[3], double side_length) {
    double delta[3];
    for (int i = 0; i < 3; i++) {
        delta[i] = fabs(p1[i] - p2[i]);
        delta[i] = fmin(delta[i], side_length - delta[i]);
    }
    return delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
}

double toroidal_distance_squared_2D(double p1[2], double p2[2], double side_length) {
    double delta[2];
    for (int i = 0; i < 2; ++i) {
        delta[i] = fabs(p1[i] - p2[i]);
        delta[i] = fmin(delta[i], side_length - delta[i]);
    }
    return delta[0]*delta[0] + delta[1]*delta[1];
}

double toroidal_distance(double p1[3], double p2[3], double side_length) {
    return sqrt(toroidal_distance_squared(p1, p2, side_length));
}

// --- Funzione Levy originale ---
double Levy(double mu, int lmax, double normalization_constant) {
    // flip a coin with probability "normalization constant"
    double toss = (double)rand() / RAND_MAX;
    double unif = (double)rand() / RAND_MAX;
    if (toss <= normalization_constant){
        return unif;
    } else {
        if (mu == 1){
            return exp((1 - normalization_constant)/normalization_constant*unif);
        } else {
            return pow((1 - normalization_constant)/normalization_constant*(1-mu)*unif + 1,1/(1-mu));
        }
    }
}

// --- Rotazione per disco ---
void rotate_point(double point[3], double normal[3], double rotated[3]) {
    double z_axis[3] = {0, 0, 1};
    double v[3] = {
        z_axis[1] * normal[2] - z_axis[2] * normal[1],
        z_axis[2] * normal[0] - z_axis[0] * normal[2],
        z_axis[0] * normal[1] - z_axis[1] * normal[0]
    };
    double s = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    double c = z_axis[0]*normal[0] + z_axis[1]*normal[1] + z_axis[2]*normal[2];

    if (s == 0) {
        for (int i = 0; i < 3; i++) rotated[i] = point[i];
        return;
    }

    double vx = v[0], vy = v[1], vz = v[2];
    double R[3][3] = {
        {c + vx*vx*(1-c),     vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s},
        {vy*vx*(1-c) + vz*s,  c + vy*vy*(1-c),    vy*vz*(1-c) - vx*s},
        {vz*vx*(1-c) - vy*s,  vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)}
    };

    for (int i = 0; i < 3; i++) {
        rotated[i] = 0;
        for (int j = 0; j < 3; j++) {
            rotated[i] += R[i][j] * point[j];
        }
    }
}



Result LevySearch3D_MultiWalker(int n_walkers, const char* initialization, double n_volume, double mu, int lmax,
                                double D, const char* TargetShape, int num_targets_to_generate,
                                double target_distance_from_origin, double p, double normalization_constant, int steps_between, int max_touches, int delta_selector, double delta) {
    double cube_side = cbrt(n_volume);

    // Allocate memory for walkers
    double (*walkers)[3] = (double (*)[3])malloc(n_walkers * sizeof(double[3]));
    if (walkers == NULL) {
        fprintf(stderr, "Memory allocation failed for walkers\n");
        exit(EXIT_FAILURE);
    }

    // Initialize walkers
    if (strcmp(initialization, "independent") == 0) {
        for (int i = 0; i < n_walkers; ++i) {
            walkers[i][0] = (double)rand() / RAND_MAX * cube_side;
            walkers[i][1] = (double)rand() / RAND_MAX * cube_side;
            walkers[i][2] = (double)rand() / RAND_MAX * cube_side;
        }
    } else if (strcmp(initialization, "nest") == 0) {
        double random_point[3];
        random_point[0] = (double)rand() / RAND_MAX * cube_side;
        random_point[1] = (double)rand() / RAND_MAX * cube_side;
        random_point[2] = (double)rand() / RAND_MAX * cube_side;
        for (int i = 0; i < n_walkers; ++i) {
            walkers[i][0] = random_point[0];
            walkers[i][1] = random_point[1];
            walkers[i][2] = random_point[2];
        }
    }

    // Allocate memory for target positions
    double (*target_positions)[3] = (double (*)[3])malloc(num_targets_to_generate * sizeof(double[3]));
    if (target_positions == NULL) {
        fprintf(stderr, "Memory allocation failed for target_positions\n");
        exit(EXIT_FAILURE);
    }

    // Initialize target positions
    if (target_distance_from_origin >= 0) { // Fixed distance
        for (int i = 0; i < num_targets_to_generate; ++i) {
            double theta = (double)rand() / RAND_MAX * 2 * M_PI;
            double phi = (double)rand() / RAND_MAX * M_PI;

            double x = target_distance_from_origin * sin(phi) * cos(theta);
            double y = target_distance_from_origin * sin(phi) * sin(theta);
            double z = target_distance_from_origin * cos(phi);

            target_positions[i][0] = fmod(x + cube_side / 2.0, cube_side);
            target_positions[i][1] = fmod(y + cube_side / 2.0, cube_side);
            target_positions[i][2] = fmod(z + cube_side / 2.0, cube_side);
            if (target_positions[i][0] < 0) target_positions[i][0] += cube_side;
            if (target_positions[i][1] < 0) target_positions[i][1] += cube_side;
            if (target_positions[i][2] < 0) target_positions[i][2] += cube_side;
        }
    } else { // Random placement
        for (int i = 0; i < num_targets_to_generate; ++i) {
            target_positions[i][0] = (double)rand() / RAND_MAX * cube_side;
            target_positions[i][1] = (double)rand() / RAND_MAX * cube_side;
            target_positions[i][2] = (double)rand() / RAND_MAX * cube_side;
        }
    }

    double* walker_times = (double*)calloc(n_walkers, sizeof(double)); // Initialize to 0.0
    double* discovery_times = (double*)malloc(n_walkers * sizeof(double));
    int* first_touch_nsteps = (int*)malloc(n_walkers * sizeof(int));
    int* second_touch_nsteps = (int*)malloc(n_walkers * sizeof(int));
    int* n_touches = (int*)malloc(n_walkers * sizeof(int));
    int n_steps = 0;
    int winning_index = -1;
    Result result = {0.0, 0, 0};
    if (walker_times == NULL || discovery_times == NULL) {
        fprintf(stderr, "Memory allocation failed for times arrays\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n_walkers; ++i) {
        discovery_times[i] = HUGE_VAL; // Initialize to infinity
        first_touch_nsteps[i] = -1;
        second_touch_nsteps[i] = -1;
        n_touches[i] = 0;
    }

    int any_walker_found_target = 0;

    while (1) {
        double min_overall_discovery_time = HUGE_VAL;
        n_steps += 1;
        for (int i = 0; i < n_walkers; ++i) {
            if (discovery_times[i] < min_overall_discovery_time) {
                min_overall_discovery_time = discovery_times[i];
            }
        }

        if (any_walker_found_target) {
            int all_remaining_walkers_past_min_time = 1;
            for (int i = 0; i < n_walkers; ++i) {
                if (discovery_times[i] == HUGE_VAL && walker_times[i] < min_overall_discovery_time) {
                    all_remaining_walkers_past_min_time = 0;
                    break;
                }
            }
            if (all_remaining_walkers_past_min_time) {
                result.detection_time = min_overall_discovery_time;
                result.first_touch_steps = first_touch_nsteps[winning_index];
                result.second_touch_steps = second_touch_nsteps[winning_index];
                free(walkers);
                free(target_positions);
                free(walker_times);
                free(discovery_times);
                free(first_touch_nsteps);
                free(second_touch_nsteps);
                free(n_touches);
                return result;
            }
        }

        for (int i = 0; i < n_walkers; ++i) {
            if (discovery_times[i] != HUGE_VAL && walker_times[i] >= min_overall_discovery_time) {
                continue;
            }

            double l = Levy(mu, lmax, normalization_constant);
            walker_times[i] += l;

            double theta = (double)rand() / RAND_MAX * 2 * M_PI;
            double phi = (double)rand() / RAND_MAX * M_PI;
            double direction[3];
            direction[0] = sin(phi) * cos(theta);
            direction[1] = sin(phi) * sin(theta);
            direction[2] = cos(phi);

            walkers[i][0] += direction[0] * l;
            walkers[i][1] += direction[1] * l;
            walkers[i][2] += direction[2] * l;

            // Apply toroidal boundary conditions
            walkers[i][0] = fmod(walkers[i][0], cube_side);
            walkers[i][1] = fmod(walkers[i][1], cube_side);
            walkers[i][2] = fmod(walkers[i][2], cube_side);
            if (walkers[i][0] < 0) walkers[i][0] += cube_side;
            if (walkers[i][1] < 0) walkers[i][1] += cube_side;
            if (walkers[i][2] < 0) walkers[i][2] += cube_side;

            int found_this_walker_in_this_step = 0;

            for (int j = 0; j < num_targets_to_generate; ++j) {
                double *target_center = target_positions[j];

                int inside_target = 0;

                if (strcmp(TargetShape, "Ball") == 0) {
                    if (toroidal_distance(walkers[i], target_center, cube_side) <= 0.5 * D + 1) {
                        inside_target = 1;
                    }
                } else if (strcmp(TargetShape, "Line") == 0) {
                    double dx_torus = fmin(fabs(walkers[i][0] - target_center[0]), cube_side - fabs(walkers[i][0] - target_center[0]));
                    double dz_torus = fmin(fabs(walkers[i][2] - target_center[2]), cube_side - fabs(walkers[i][2] - target_center[2]));

                    double p1_line_y = target_center[1] - D / 2.0;
                    double p2_line_y = target_center[1] + D / 2.0;

                    double closest_y = fmax(p1_line_y, fmin(walkers[i][1], p2_line_y));
                    double dy = walkers[i][1] - closest_y;

                    double dist_to_line_torus_squared = dx_torus*dx_torus + dy*dy + dz_torus*dz_torus;

                    if (sqrt(dist_to_line_torus_squared) <= 1) {
                        inside_target = 1;
                    }

                } else if (strcmp(TargetShape, "3Line") == 0) {
                    // step 1: compute vector from the center to the searcher point
                    double delta_x = fmin(fabs(target_center[0] - walkers[i][0]), cube_side - fabs(target_center[0] - walkers[i][0]));
                    double delta_y = fmin(fabs(target_center[1] - walkers[i][1]), cube_side - fabs(target_center[1] - walkers[i][1]));
                    double delta_z = fmin(fabs(target_center[2] - walkers[i][2]), cube_side - fabs(target_center[2] - walkers[i][2]));
                    // Observe that for this shape, the parameter D does not represent the diameter but the length of a single line 
                    // check lines
                    if ((delta_x * delta_x + delta_y * delta_y <= 1 && delta_z <= D + 1) || 
                        (delta_y * delta_y + delta_z * delta_z <= 1 && delta_x <= D + 1) || 
                        (delta_z * delta_z + delta_x * delta_x <= 1 && delta_y <= D + 1)){
                            inside_target = 1;
                    }
                } else if (strcmp(TargetShape, "3Line_convex_hull") == 0) {
                    // step 1: compute vector from the center to the searcher point
                    double delta_x = fmin(fabs(target_center[0] - walkers[i][0]), cube_side - fabs(target_center[0] - walkers[i][0]));
                    double delta_y = fmin(fabs(target_center[1] - walkers[i][1]), cube_side - fabs(target_center[1] - walkers[i][1]));
                    double delta_z = fmin(fabs(target_center[2] - walkers[i][2]), cube_side - fabs(target_center[2] - walkers[i][2]));
                    // Observe that for this shape, the parameter D does not represent the diameter but the length of a single line 
                    // check convex hull 
                    if ((delta_x >= -1 && delta_y >= -1 && delta_z >= -1) && // -1 because of the detection radius 
                        (fabs(delta_x) + fabs(delta_y) + fabs(delta_z) <= D + 1)) 
                        {
                            inside_target = 1;
                        }
                } else if (strcmp(TargetShape, "Square") == 0) { // Cube
                    double half_side_target = D / 2.0 + 1;

                    double delta_x = fmin(fabs(target_center[0] - walkers[i][0]), cube_side - fabs(target_center[0] - walkers[i][0]));
                    double delta_y = fmin(fabs(target_center[1] - walkers[i][1]), cube_side - fabs(target_center[1] - walkers[i][1]));
                    double delta_z = fmin(fabs(target_center[2] - walkers[i][2]), cube_side - fabs(target_center[2] - walkers[i][2]));

                    if (delta_x < half_side_target &&
                        delta_y < half_side_target &&
                        delta_z < half_side_target) {
                        inside_target = 1;
                    }
                } else if (strcmp(TargetShape, "Disk") == 0) {
                    double walker_xy[2] = {walkers[i][0], walkers[i][1]};
                    double target_xy[2] = {target_center[0], target_center[1]};
                    double dist_xy_squared = toroidal_distance_squared_2D(walker_xy, target_xy, cube_side);
                    double dist_xy = sqrt(dist_xy_squared);

                    double dist_z = fmin(fabs(walkers[i][2] - target_center[2]), cube_side - fabs(walkers[i][2] - target_center[2]));

                    if (dist_xy <= 0.5 * D + 1 && dist_z <= 1) {
                        inside_target = 1;
                    }
                } else if (strcmp(TargetShape, "Ball_boundary") == 0) {
                    double dist = toroidal_distance(walkers[i], target_center, cube_side);
                    if (dist <= 0.5 * D + 1 && dist >= 0.5 * D) {
                        inside_target = 1;
                    }
                } else if (strcmp(TargetShape, "Ball_no_boundary") == 0) {
                    double dist = toroidal_distance(walkers[i], target_center, cube_side);
                    if (dist <= 0.5 * D) {
                        inside_target = 1;
                    }
                } else if (strcmp(TargetShape, "2D_rectangle") == 0) {
                    // in this setting, the diameter is the surface and we have to use the delta parameter
                    double side_x = (pow(D,delta))/2; // longest side + detection radius
                    double side_y = (pow(D,1-delta))/2; // shortest side + detection radius
                    
                    double dist_x = fmin(fabs(walkers[i][0] - target_center[0]), cube_side - fabs(walkers[i][0] - target_center[0]));
                    double dist_y = fmin(fabs(walkers[i][1] - target_center[1]), cube_side - fabs(walkers[i][1] - target_center[1]));
                    double dist_z = fmin(fabs(walkers[i][2] - target_center[2]), cube_side - fabs(walkers[i][2] - target_center[2]));

                    if (dist_x < side_x &&
                        dist_y < side_y &&
                        dist_z < 1) {
                        inside_target = 1;
                    }

                }else {
                    fprintf(stderr, "Unknown TargetShape: %s\n", TargetShape);
                    free(walkers);
                    free(target_positions);
                    free(walker_times);
                    free(discovery_times);
                    exit(EXIT_FAILURE);
                }

                if (inside_target) {
                    if (n_touches[i] == 0){
                        first_touch_nsteps[i] = n_steps;
                    } else if (n_touches[i] == 1){
                        second_touch_nsteps[i] = n_steps;
                    }
                    if (n_touches[i] < max_touches - 1){
                        n_touches[i] += 1;
                    } else {
                        found_this_walker_in_this_step = 1;
                        winning_index = i;
                        break;
                    }
                    double r = (double)rand() / RAND_MAX;
                    if (r <= p) {
                        found_this_walker_in_this_step = 1;
                        break;
                    }
                }
            }

            if (found_this_walker_in_this_step) {
                if (discovery_times[i] == HUGE_VAL) {
                    discovery_times[i] = walker_times[i];
                    any_walker_found_target = 1;
                }
            }
        }
    }
}

double LevySearch3D_MultiWalker_boundary_detection(int n_walkers, const char* initialization, double n_volume, double mu, int lmax,
                                int D, const char* TargetShape, int num_targets_to_generate,
                                double target_distance_from_origin, double p, int* total_points_inside_target, double normalization_constant) {
    double cube_side = cbrt(n_volume);

    // Allocate memory for walkers
    double (*walkers)[3] = (double (*)[3])malloc(n_walkers * sizeof(double[3]));
    int* walkers_steps_inside_target = (int*)malloc(n_walkers * sizeof(int));
    if (walkers == NULL) {
        fprintf(stderr, "Memory allocation failed for walkers\n");
        exit(EXIT_FAILURE);
    }

    // Initialize walkers
    if (strcmp(initialization, "independent") == 0) {
        for (int i = 0; i < n_walkers; ++i) {
            walkers[i][0] = (double)rand() / RAND_MAX * cube_side;
            walkers[i][1] = (double)rand() / RAND_MAX * cube_side;
            walkers[i][2] = (double)rand() / RAND_MAX * cube_side;
        }
    } else if (strcmp(initialization, "nest") == 0) {
        double random_point[3];
        random_point[0] = (double) cube_side / 2.0;
        random_point[1] = (double) cube_side / 2.0;
        random_point[2] = (double) cube_side / 2.0;
        for (int i = 0; i < n_walkers; ++i) {
            walkers[i][0] = random_point[0];
            walkers[i][1] = random_point[1];
            walkers[i][2] = random_point[2];
        }
    }

    for (int i = 0; i < n_walkers; ++i) {
        walkers_steps_inside_target[i] = 0; // Initialize steps inside target to 0
    }

    // Allocate memory for target positions
    double (*target_positions)[3] = (double (*)[3])malloc(num_targets_to_generate * sizeof(double[3]));
    if (target_positions == NULL) {
        fprintf(stderr, "Memory allocation failed for target_positions\n");
        exit(EXIT_FAILURE);
    }

    // Initialize target positions
    if (target_distance_from_origin >= 0) { // Fixed distance
        for (int i = 0; i < num_targets_to_generate; ++i) {
            double theta = (double)rand() / RAND_MAX * 2 * M_PI;
            double phi = (double)rand() / RAND_MAX * M_PI;

            double x = target_distance_from_origin * sin(phi) * cos(theta);
            double y = target_distance_from_origin * sin(phi) * sin(theta);
            double z = target_distance_from_origin * cos(phi);

            target_positions[i][0] = fmod(x + cube_side / 2.0, cube_side);
            target_positions[i][1] = fmod(y + cube_side / 2.0, cube_side);
            target_positions[i][2] = fmod(z + cube_side / 2.0, cube_side);
            if (target_positions[i][0] < 0) target_positions[i][0] += cube_side;
            if (target_positions[i][1] < 0) target_positions[i][1] += cube_side;
            if (target_positions[i][2] < 0) target_positions[i][2] += cube_side;
        }
    } else { // Random placement
        for (int i = 0; i < num_targets_to_generate; ++i) {
            target_positions[i][0] = (double)rand() / RAND_MAX * cube_side;
            target_positions[i][1] = (double)rand() / RAND_MAX * cube_side;
            target_positions[i][2] = (double)rand() / RAND_MAX * cube_side;
        }
    }

    double* walker_times = (double*)calloc(n_walkers, sizeof(double)); // Initialize to 0.0
    double* discovery_times = (double*)malloc(n_walkers * sizeof(double));
    int jumps_inside = 0;
    double dist = 0.0;
    if (walker_times == NULL || discovery_times == NULL) {
        fprintf(stderr, "Memory allocation failed for times arrays\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n_walkers; ++i) {
        discovery_times[i] = HUGE_VAL; // Initialize to infinity
    }

    int any_walker_found_target_boundary = 0;

    while (1) {
        double min_overall_discovery_time = HUGE_VAL;
        for (int i = 0; i < n_walkers; ++i) {
            if (discovery_times[i] < min_overall_discovery_time) {
                min_overall_discovery_time = discovery_times[i];
                jumps_inside = walkers_steps_inside_target[i];
            }
        }

        if (any_walker_found_target_boundary) {
            int all_remaining_walkers_past_min_time = 1;
            for (int i = 0; i < n_walkers; ++i) {
                if (discovery_times[i] == HUGE_VAL && walker_times[i] < min_overall_discovery_time) {
                    all_remaining_walkers_past_min_time = 0;
                    break;
                }
            }
            if (all_remaining_walkers_past_min_time) {
                *total_points_inside_target = jumps_inside;
                free(walkers);
                free(target_positions);
                free(walker_times);
                free(discovery_times);
                free(walkers_steps_inside_target);
                return min_overall_discovery_time;
            }
        }

        for (int i = 0; i < n_walkers; ++i) {
            if (discovery_times[i] != HUGE_VAL && walker_times[i] >= min_overall_discovery_time) {
                continue;
            }

            int l = Levy(mu, lmax, normalization_constant);
            walker_times[i] += l;

            double theta = (double)rand() / RAND_MAX * 2 * M_PI;
            double phi = (double)rand() / RAND_MAX * M_PI;
            double direction[3];
            direction[0] = sin(phi) * cos(theta);
            direction[1] = sin(phi) * sin(theta);
            direction[2] = cos(phi);

            walkers[i][0] += direction[0] * l;
            walkers[i][1] += direction[1] * l;
            walkers[i][2] += direction[2] * l;

            // Apply toroidal boundary conditions
            walkers[i][0] = fmod(walkers[i][0], cube_side);
            walkers[i][1] = fmod(walkers[i][1], cube_side);
            walkers[i][2] = fmod(walkers[i][2], cube_side);
            if (walkers[i][0] < 0) walkers[i][0] += cube_side;
            if (walkers[i][1] < 0) walkers[i][1] += cube_side;
            if (walkers[i][2] < 0) walkers[i][2] += cube_side;

            int found_this_walker_in_this_step = 0;

            for (int j = 0; j < num_targets_to_generate; ++j) {
                double *target_center = target_positions[j];

                int inside_target = 0;
                // here we check the distance between the walker and the target 
                if (strcmp(TargetShape, "Ball") == 0) {
                    dist = toroidal_distance(walkers[i], target_center, cube_side);
                    if (dist <= 0.5 * D + 1) {
                        walkers_steps_inside_target[i]++;
                    }
                    if (dist >= D/2 && dist <= D/2 + 1) {
                        inside_target = 1;
                    }
                } else {
                    fprintf(stderr, "Error: The input shape is not valid.");
                }   
                
                if (inside_target) {
                    double r = (double)rand() / RAND_MAX;
                    if (r <= p) {
                        found_this_walker_in_this_step = 1;
                        break;
                    }
                }
            }

            if (found_this_walker_in_this_step) {
                if (discovery_times[i] == HUGE_VAL) {
                    discovery_times[i] = walker_times[i];
                    any_walker_found_target_boundary = 1;
                }
            }
        }
    }
}

double LevySearch3D_SingleWalker_distance(const char* initialization, double cube_side, double mu, int lmax,
                                int D, const char* TargetShape, int num_targets_to_generate,
                                double target_distance_from_origin, double p, double* distance, double normalization_constant) {
    // Allocate memory for walkers
    double walkers[3];
    int walkers_steps_inside_target;

    // Initialize walkers
    if (strcmp(initialization, "independent") == 0) {
        walkers[0] = (double)rand() / RAND_MAX * cube_side;
        walkers[1] = (double)rand() / RAND_MAX * cube_side;
        walkers[2] = (double)rand() / RAND_MAX * cube_side;
    } else if (strcmp(initialization, "nest") == 0) {
        double random_point[3];
        random_point[0] = (double) cube_side / 2.0;
        random_point[1] = (double) cube_side / 2.0;
        random_point[2] = (double) cube_side / 2.0;
        walkers[0] = random_point[0];
        walkers[1] = random_point[1];
        walkers[2] = random_point[2];
    }

    
    walkers_steps_inside_target = 0; // Initialize steps inside target to 0

    // Allocate memory for target positions
    double (*target_positions)[3] = (double (*)[3])malloc(num_targets_to_generate * sizeof(double[3]));
    if (target_positions == NULL) {
        fprintf(stderr, "Memory allocation failed for target_positions\n");
        exit(EXIT_FAILURE);
    }

    // Initialize target positions
    if (target_distance_from_origin >= 0) { // Fixed distance
        for (int i = 0; i < num_targets_to_generate; ++i) {
            double theta = (double)rand() / RAND_MAX * 2 * M_PI;
            double phi = (double)rand() / RAND_MAX * M_PI;

            double x = target_distance_from_origin * sin(phi) * cos(theta);
            double y = target_distance_from_origin * sin(phi) * sin(theta);
            double z = target_distance_from_origin * cos(phi);

            target_positions[i][0] = fmod(x + cube_side / 2.0, cube_side);
            target_positions[i][1] = fmod(y + cube_side / 2.0, cube_side);
            target_positions[i][2] = fmod(z + cube_side / 2.0, cube_side);
            if (target_positions[i][0] < 0) target_positions[i][0] += cube_side;
            if (target_positions[i][1] < 0) target_positions[i][1] += cube_side;
            if (target_positions[i][2] < 0) target_positions[i][2] += cube_side;
        }
    } else { // Random placement
        for (int i = 0; i < num_targets_to_generate; ++i) {
            target_positions[i][0] = (double)rand() / RAND_MAX * cube_side;
            target_positions[i][1] = (double)rand() / RAND_MAX * cube_side;
            target_positions[i][2] = (double)rand() / RAND_MAX * cube_side;
        }
    }

    double walker_times = 0.0;
    double dist = 0.0;
    int temp = 0;
    
    while (1) {
            temp++;
            double l = Levy(mu, lmax, normalization_constant);
            printf("Step length: %0.1f\n", l);
            walker_times+= l;

            double theta = (double)rand() / RAND_MAX * 2 * M_PI;
            double phi = (double)rand() / RAND_MAX * M_PI;
            double direction[3];
            direction[0] = sin(phi) * cos(theta);
            direction[1] = sin(phi) * sin(theta);
            direction[2] = cos(phi);

            walkers[0] += direction[0] * l;
            walkers[1] += direction[1] * l;
            walkers[2] += direction[2] * l;

            // Apply toroidal boundary conditions
            walkers[0] = fmod(walkers[0], cube_side);
            walkers[1] = fmod(walkers[1], cube_side);
            walkers[2] = fmod(walkers[2], cube_side);
            if (walkers[0] < 0) walkers[0] += cube_side;
            if (walkers[1] < 0) walkers[1] += cube_side;
            if (walkers[2] < 0) walkers[2] += cube_side;

            for (int j = 0; j < num_targets_to_generate; ++j) {
                double *target_center = target_positions[j];

                int inside_target = 0;
                // here we check the distance between the walker and the target 
                if (strcmp(TargetShape, "Ball") == 0) {
                    dist = toroidal_distance(walkers, target_center, cube_side);
                    if (dist <= 0.5 * D + 1) {
                        walkers_steps_inside_target++;
                        *distance = dist;
                        //printf("Average step length: %f\n", walker_times / (double)temp);
                        return walker_times;
                    }
                } else {
                    fprintf(stderr, "Error: The input shape is not valid.");
                }   
            }
    }
}

