#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

// Simple row container
struct GyroRow {
    double t_sec{};
    double x{};
    double y{};
    double z{};
};

static std::vector<std::string> split_csv_line(const std::string& line) {
    // Minimal CSV split (works for files: no quoted commas)
    std::vector<std::string> out;
    std::string token;
    std::stringstream ss(line);
    while (std::getline(ss, token, ',')) out.push_back(token);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/Gyroscope.csv\n";
        return 1;
    }

    const std::string in_path = argv[1];
    std::ifstream fin(in_path);
    if (!fin) {
        std::cerr << "Failed to open: " << in_path << "\n";
        return 1;
    }

    std::string header;
    if (!std::getline(fin, header)) {
        std::cerr << "Empty file.\n";
        return 1;
    }

    // Expect: time,seconds_elapsed,z,y,x  
    // Read by index to avoid dependency on exact header spelling.
    std::vector<GyroRow> rows;
    rows.reserve(100000);

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto c = split_csv_line(line);
        if (c.size() < 5) continue;

        GyroRow r;
        // c[0]=time, c[1]=seconds_elapsed, c[2]=z, c[3]=y, c[4]=x
        r.t_sec = std::stod(c[1]);
        r.z = std::stod(c[2]);
        r.y = std::stod(c[3]);
        r.x = std::stod(c[4]);

        rows.push_back(r);
    }

    if (rows.empty()) {
        std::cerr << "No data rows parsed.\n";
        return 1;
    }

    // Gnuplot-friendly data file: t x y z
    const std::string dat_path = "gyro_plot.dat";
    {
        std::ofstream fout(dat_path);
        fout << "# t_sec x y z\n";
        for (const auto& r : rows) {
            fout << r.t_sec << " " << r.x << " " << r.y << " " << r.z << "\n";
        }
    }

    // Gnuplot script that outputs a PNG
    const std::string gp_path = "plot_gyro.gp";
    {
        std::ofstream gp(gp_path);
        gp <<
            "set terminal pngcairo size 1400,800\n"
            "set output 'gyro_plot.png'\n"
            "set grid\n"
            "set key left top\n"
            "set xlabel 'seconds_elapsed (s)'\n"
            "set ylabel 'angular velocity (units in file)'\n"
            "set title 'Gyroscope: x/y/z vs time'\n"
            "plot "
            "'gyro_plot.dat' using 1:2 with lines title 'wx (x)', "
            "'gyro_plot.dat' using 1:3 with lines title 'wy (y)', "
            "'gyro_plot.dat' using 1:4 with lines title 'wz (z)'\n";
    }

    // Run gnuplot
    const std::string cmd = "gnuplot " + gp_path;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "gnuplot failed. Is it installed?\n";
        return 1;
    }

    std::cout << "Parsed " << rows.size() << " samples.\n";
    std::cout << "Wrote: " << dat_path << ", " << gp_path << ", gyro_plot.png\n";
    return 0;
}